#!/usr/bin/env python3
"""
mine_praise_prompt_bank.py

Build a label-free transferable pedestrian attribute prompt bank for PRAISE-style
Attribute-Conditioned Subspace (ACS) retrieval adaptation.

The script can:
  1. discover images from a dataset root, an image list, or a pid->images mapping,
  2. sample pedestrian images,
  3. ask a VLM to extract compact appearance attributes,
  4. aggregate recurring attributes into a cleaned vocabulary,
  5. export prompts in JSON/TXT/PTH formats.

Default categories intentionally avoid age/gender. To reproduce a bank that uses
sensitive demographic-like prompts, pass --include_sensitive, but this is not
recommended for deployment or fairness-sensitive analysis.

Example:
  python mine_praise_prompt_bank.py \
    --image_root /path/to/Market-1501-v15.09.15/bounding_box_train \
    --dataset market1501 \
    --out_dir ./prompt_banks/market \
    --sample_per_pid 4 \
    --max_images 3000 \
    --device cuda:0 \
    --min_count 5

Resume:
  python mine_praise_prompt_bank.py --image_root ... --out_dir ... --resume
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_TEMPLATE = "A photo of a person with {attribute}."
DEFAULT_NEUTRAL_PROMPT = "A photo of a person."

CATEGORY_TEMPLATES = {
    "upper_color": "A photo of a person wearing a {attribute} upper-body garment.",
    "upper_type": "A photo of a person wearing {attribute}.",
    "lower_color": "A photo of a person wearing a {attribute} lower-body garment.",
    "lower_type": "A photo of a person wearing {attribute}.",
    "shoes": "A photo of a person wearing {attribute} shoes.",
    "bag": "A photo of a person carrying {attribute}.",
    "accessory": "A photo of a person with {attribute}.",
    "hair": "A photo of a person with {attribute} hair.",
    "gender": "A photo of a {attribute}.",
    "age": "A photo of a {attribute} person.",
}

DEFAULT_CATEGORIES = [
    "upper_color",
    "upper_type",
    "lower_color",
    "lower_type",
    "shoes",
    "bag",
    "accessory",
    "hair",
]
SENSITIVE_CATEGORIES = ["gender", "age"]

UNKNOWN_VALUES = {
    "",
    "none",
    "no",
    "n/a",
    "na",
    "null",
    "unknown",
    "unclear",
    "not visible",
    "not clear",
    "not shown",
    "not specified",
    "cannot determine",
}

COLOR_SYNONYMS = {
    "grey": "gray",
    "dark grey": "dark gray",
    "light grey": "light gray",
    "navy blue": "navy",
    "navy-colored": "navy",
    "bluish": "blue",
    "reddish": "red",
    "maroon": "dark red",
    "burgundy": "dark red",
    "beige": "tan",
    "khaki": "tan",
    "cream": "white",
    "off white": "white",
    "off-white": "white",
}
TYPE_SYNONYMS = {
    "t shirt": "t-shirt",
    "tee shirt": "t-shirt",
    "tee": "t-shirt",
    "trousers": "pants",
    "jeans pants": "jeans",
    "denim pants": "jeans",
    "shorts pants": "shorts",
    "back pack": "backpack",
    "rucksack": "backpack",
    "hand bag": "handbag",
    "shoulderbag": "shoulder bag",
    "sneaker": "sneakers",
    "trainer": "sneakers",
    "trainers": "sneakers",
}


def make_extraction_prompt(include_sensitive: bool = False) -> str:
    categories = list(DEFAULT_CATEGORIES)
    if include_sensitive:
        categories += SENSITIVE_CATEGORIES

    schema = {cat: "string or list of short strings; use unknown if not visible" for cat in categories}
    return f"""
Extract only visible pedestrian appearance attributes from this cropped person image.

Return ONLY valid JSON with these keys:
{json.dumps(schema, indent=2)}

Rules:
- Use lowercase.
- Keep values short: 1-4 words.
- Do not mention background, pose, action, camera, brands, identity, or image quality.
- Use "unknown" if a cue is absent or unclear.
- For bag/accessory, use "none" only if clearly absent; otherwise "unknown".
- For colors, use simple color names such as black, white, red, blue, green, gray, yellow, pink, purple, orange, brown, tan, navy.
- For clothing type, use simple nouns such as t-shirt, shirt, jacket, hoodie, coat, pants, jeans, shorts, skirt, dress.
- For shoes, include color/type only if visible.
- Output JSON only. No prose.
""".strip()


@dataclass
class ImageRecord:
    path: str
    pid: str = "unknown"
    dataset: str = "generic"


@dataclass
class PromptRecord:
    category: str
    attribute: str
    prompt: str
    count: int


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def infer_pid_from_filename(path: str, dataset: str = "generic") -> str:
    """Infer PID from common ReID filenames.

    Market/Duke/most ReID folders: 0001_c1s1_...jpg -> 0001
    MSMT-like: 0001_00_...jpg -> 0001
    Generic fallback: first integer-like token.
    """
    stem = Path(path).stem
    if dataset.lower() in {"market", "market1501", "duke", "dukemtmc", "msmt", "msmt17", "generic"}:
        m = re.match(r"(-?\d+)", stem)
        if m:
            return m.group(1)

    if "_" in stem:
        return stem.split("_", 1)[0]

    m = re.search(r"\d+", stem)
    return m.group(0) if m else "unknown"


def is_junk_pid(pid: str) -> bool:
    try:
        return int(pid) < 0 or int(pid) == 0
    except Exception:
        return pid.lower() in {"junk", "background", "unknown"}


def discover_images(
    image_root: Optional[str],
    dataset: str,
    include_junk: bool = False,
) -> List[ImageRecord]:
    if not image_root:
        return []

    root = Path(image_root)
    if not root.exists():
        raise FileNotFoundError(f"image_root does not exist: {image_root}")

    records: List[ImageRecord] = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        pid = infer_pid_from_filename(str(p), dataset=dataset)
        if not include_junk and is_junk_pid(pid):
            continue
        records.append(ImageRecord(path=str(p), pid=str(pid), dataset=dataset))
    return records


def load_records_from_pid_mapping(path: Optional[str], dataset: str, include_junk: bool = False) -> List[ImageRecord]:
    if not path:
        return []

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"pid mapping does not exist: {path}")

    if path_obj.suffix == ".pth":
        obj = torch.load(str(path_obj), map_location="cpu")
    else:
        with open(path_obj, "r", encoding="utf-8") as f:
            obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError("Expected mapping {pid: [image_path, ...]} in .pth/.json")

    records: List[ImageRecord] = []
    for pid, paths in obj.items():
        pid = str(pid)
        if not include_junk and is_junk_pid(pid):
            continue
        if not isinstance(paths, (list, tuple)):
            continue
        for p in paths:
            records.append(ImageRecord(path=str(p), pid=pid, dataset=dataset))
    return records


def load_records_from_image_list(path: Optional[str], dataset: str, include_junk: bool = False) -> List[ImageRecord]:
    if not path:
        return []
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"image list does not exist: {path}")

    records: List[ImageRecord] = []
    with open(path_obj, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if "," in line:
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 2 and Path(parts[1]).suffix.lower() in IMAGE_EXTS:
                    pid, img_path = parts[0], parts[1]
                else:
                    img_path, pid = parts[0], parts[1]
            else:
                parts = line.split()
                img_path = parts[0]
                pid = parts[1] if len(parts) > 1 else infer_pid_from_filename(img_path, dataset)

            if not include_junk and is_junk_pid(str(pid)):
                continue
            records.append(ImageRecord(path=img_path, pid=str(pid), dataset=dataset))
    return records


def sample_records(
    records: Sequence[ImageRecord],
    sample_per_pid: int = 4,
    max_images: Optional[int] = None,
    seed: int = 0,
) -> List[ImageRecord]:
    rng = random.Random(seed)
    by_pid: DefaultDict[str, List[ImageRecord]] = collections.defaultdict(list)
    for rec in records:
        by_pid[rec.pid].append(rec)

    sampled: List[ImageRecord] = []
    for _, items in by_pid.items():
        items = list(items)
        rng.shuffle(items)
        sampled.extend(items[:sample_per_pid] if sample_per_pid > 0 else items)

    rng.shuffle(sampled)
    if max_images is not None and max_images > 0:
        sampled = sampled[:max_images]
    return sampled


def load_processor(model_id: str):
    return AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


def load_model(model_id: str, device_map: str, dtype: str):
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]

    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    ).eval()
    return model


def first_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def open_rgb(path: str) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[warn] failed to open image {path}: {e}", file=sys.stderr)
        return None


def generate_attribute_json_text(
    processor,
    model,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = 192,
) -> str:
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
    ]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[image], return_tensors="pt")

    device = first_model_device(model)
    inputs = inputs.to(device)

    tokenizer = processor.tokenizer
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None) or eos_id

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )

    return processor.batch_decode(out, skip_special_tokens=True)[0].strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    candidates = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    if not candidates and text.startswith("{") and text.endswith("}"):
        candidates = [text]
    if not candidates:
        raise ValueError(f"No JSON object found. Raw output begins: {text[:300]}")

    last_error: Optional[Exception] = None
    for candidate in reversed(candidates):
        try:
            return json.loads(candidate)
        except Exception as e:
            last_error = e

    raise ValueError(f"Failed to parse JSON: {last_error}. Raw output begins: {text[:300]}")


def normalize_text_value(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_values = value
    else:
        raw_values = [value]

    out: List[str] = []
    for raw in raw_values:
        if raw is None:
            continue
        s = str(raw).lower().strip()
        s = s.replace("_", " ")
        s = re.sub(r"[\.;:\(\)\[\]\{\}\"']", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        s = s.strip(" ,-/")

        pieces = re.split(r"\s*(?:,|/|\band\b|\bwith\b)\s*", s)
        for piece in pieces:
            piece = piece.strip(" ,-/")
            if not piece or piece in UNKNOWN_VALUES:
                continue
            out.append(piece)

    return out


def canonicalize_attribute(category: str, value: str) -> Optional[str]:
    value = value.lower().strip()
    value = re.sub(r"\b(person|pedestrian|wearing|with|carrying|visible|appears|looks)\b", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    if value in UNKNOWN_VALUES:
        return None

    if len(value.split()) > 5:
        return None

    if "color" in category:
        value = COLOR_SYNONYMS.get(value, value)
        value = re.sub(r"\b(colored|colour|color)\b", "", value).strip()
    else:
        value = TYPE_SYNONYMS.get(value, value)

    value = re.sub(r"\s+", " ", value).strip()
    if value in UNKNOWN_VALUES:
        return None

    return value


def collect_vocabulary(
    parsed_records: Iterable[Dict[str, Any]],
    categories: Sequence[str],
    min_count: int,
    max_per_category: int,
) -> Dict[str, List[Tuple[str, int]]]:
    counts: Dict[str, collections.Counter] = {cat: collections.Counter() for cat in categories}

    for rec in parsed_records:
        attrs = rec.get("attributes", {})
        if not isinstance(attrs, dict):
            continue
        for cat in categories:
            values = normalize_text_value(attrs.get(cat))
            for value in values:
                canon = canonicalize_attribute(cat, value)
                if canon:
                    counts[cat][canon] += 1

    vocab: Dict[str, List[Tuple[str, int]]] = {}
    for cat in categories:
        items = [(v, c) for v, c in counts[cat].most_common() if c >= min_count]
        if max_per_category and max_per_category > 0:
            items = items[:max_per_category]
        vocab[cat] = items
    return vocab


def prompt_from_category_value(category: str, value: str) -> str:
    tmpl = CATEGORY_TEMPLATES.get(category, DEFAULT_TEMPLATE)

    if category == "shoes" and value.endswith("shoes"):
        return f"A photo of a person wearing {value}."

    if value in {"none", "no bag", "no accessory"}:
        return ""

    return tmpl.format(attribute=value)


def build_prompt_records(vocab: Dict[str, List[Tuple[str, int]]]) -> List[PromptRecord]:
    prompt_records: List[PromptRecord] = []
    seen_prompts = set()

    for category, items in vocab.items():
        for value, count in items:
            prompt = prompt_from_category_value(category, value)
            if not prompt:
                continue
            key = prompt.lower()
            if key in seen_prompts:
                continue
            seen_prompts.add(key)
            prompt_records.append(PromptRecord(category=category, attribute=value, prompt=prompt, count=count))

    return prompt_records


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_txt(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    categories = list(DEFAULT_CATEGORIES)
    if args.include_sensitive:
        categories += SENSITIVE_CATEGORIES

    if args.categories:
        categories = [x.strip() for x in args.categories.split(",") if x.strip()]

    records: List[ImageRecord] = []
    records += discover_images(args.image_root, args.dataset, include_junk=args.include_junk)
    records += load_records_from_pid_mapping(args.pid_to_images, args.dataset, include_junk=args.include_junk)
    records += load_records_from_image_list(args.image_list, args.dataset, include_junk=args.include_junk)

    dedup: Dict[str, ImageRecord] = {}
    for rec in records:
        dedup[os.path.abspath(rec.path)] = rec
    records = list(dedup.values())

    if not records:
        raise RuntimeError("No images found. Provide --image_root, --image_list, or --pid_to_images.")

    sampled = sample_records(
        records,
        sample_per_pid=args.sample_per_pid,
        max_images=args.max_images,
        seed=args.seed,
    )

    metadata_path = out_dir / "sampled_images.json"
    write_json(metadata_path, [asdict(r) for r in sampled])
    print(f"[info] discovered {len(records)} images; sampled {len(sampled)} images")
    print(f"[saved] {metadata_path}")

    parsed_path = out_dir / "image_attributes.jsonl"
    raw_path = out_dir / "raw_model_outputs.jsonl"

    done_paths = set()
    if args.resume:
        for row in read_jsonl(parsed_path):
            done_paths.add(os.path.abspath(row.get("path", "")))
        print(f"[info] resume enabled; found {len(done_paths)} already parsed images")

    prompt = make_extraction_prompt(include_sensitive=args.include_sensitive)

    if not args.skip_vlm:
        processor = load_processor(args.model_id)
        model = load_model(args.model_id, device_map=args.device, dtype=args.dtype)

        new_parsed_rows: List[Dict[str, Any]] = []
        new_raw_rows: List[Dict[str, Any]] = []

        for rec in tqdm(sampled, desc="extract attributes"):
            abs_path = os.path.abspath(rec.path)
            if args.resume and abs_path in done_paths:
                continue

            img = open_rgb(rec.path)
            if img is None:
                continue

            raw_text = ""
            attrs: Dict[str, Any] = {}
            error: Optional[str] = None
            try:
                raw_text = generate_attribute_json_text(
                    processor=processor,
                    model=model,
                    image=img,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                )
                attrs = extract_json_object(raw_text)
            except Exception as e:
                error = repr(e)

            raw_row = {
                "path": rec.path,
                "pid": rec.pid,
                "dataset": rec.dataset,
                "raw_output": raw_text,
                "error": error,
            }
            parsed_row = {
                "path": rec.path,
                "pid": rec.pid,
                "dataset": rec.dataset,
                "attributes": attrs,
                "error": error,
            }

            new_raw_rows.append(raw_row)
            new_parsed_rows.append(parsed_row)

            if len(new_parsed_rows) >= args.flush_every:
                append_jsonl(raw_path, new_raw_rows)
                append_jsonl(parsed_path, new_parsed_rows)
                new_raw_rows.clear()
                new_parsed_rows.clear()

        if new_parsed_rows:
            append_jsonl(raw_path, new_raw_rows)
            append_jsonl(parsed_path, new_parsed_rows)

        print(f"[saved] {parsed_path}")
        print(f"[saved] {raw_path}")

    parsed_records = read_jsonl(parsed_path)
    if not parsed_records:
        raise RuntimeError(
            f"No parsed records found at {parsed_path}. "
            "Run without --skip_vlm first, or provide an existing image_attributes.jsonl."
        )

    vocab = collect_vocabulary(
        parsed_records=parsed_records,
        categories=categories,
        min_count=args.min_count,
        max_per_category=args.max_per_category,
    )
    prompt_records = build_prompt_records(vocab)

    vocab_json = {
        cat: [{"attribute": value, "count": count} for value, count in items]
        for cat, items in vocab.items()
    }
    prompts_json = [asdict(p) for p in prompt_records]
    prompts_txt = [p.prompt for p in prompt_records]

    write_json(out_dir / "attribute_vocab.json", vocab_json)
    write_json(out_dir / "prompt_bank.json", {
        "neutral_prompt": args.neutral_prompt,
        "categories": categories,
        "prompts": prompts_json,
        "notes": {
            "include_sensitive": args.include_sensitive,
            "min_count": args.min_count,
            "max_per_category": args.max_per_category,
            "template_default": DEFAULT_TEMPLATE,
        },
    })
    write_txt(out_dir / "prompt_bank.txt", prompts_txt)

    torch.save(
        {
            "neutral_prompt": args.neutral_prompt,
            "categories": categories,
            "prompts": prompts_txt,
            "prompt_records": prompts_json,
            "vocab": vocab_json,
        },
        out_dir / "prompt_bank.pth",
    )

    with open(out_dir / "prompt_bank_stats.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "attribute", "count", "prompt"])
        writer.writeheader()
        for pr in prompt_records:
            writer.writerow(asdict(pr))

    print(f"[done] categories: {categories}")
    print(f"[done] number of prompts: {len(prompt_records)}")
    print(f"[saved] {out_dir / 'attribute_vocab.json'}")
    print(f"[saved] {out_dir / 'prompt_bank.json'}")
    print(f"[saved] {out_dir / 'prompt_bank.txt'}")
    print(f"[saved] {out_dir / 'prompt_bank.pth'}")
    print(f"[saved] {out_dir / 'prompt_bank_stats.csv'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mine a PRAISE-style transferable pedestrian attribute prompt bank from ReID/person images."
    )

    parser.add_argument("--image_root", type=str, default=None, help="Root folder containing pedestrian images.")
    parser.add_argument("--image_list", type=str, default=None, help="Optional text/CSV list of images, optionally with PIDs.")
    parser.add_argument("--pid_to_images", type=str, default=None, help=".json/.pth mapping {pid: [image paths]}.")
    parser.add_argument("--dataset", type=str, default="generic", help="Dataset name for PID parsing, e.g. market1501/msmt17/duke/generic.")
    parser.add_argument("--include_junk", action="store_true", help="Do not filter junk PID 0/-1.")

    parser.add_argument("--sample_per_pid", type=int, default=4, help="Number of images sampled per PID. Use 0 for all.")
    parser.add_argument("--max_images", type=int, default=3000, help="Max total sampled images. Use 0 for no limit.")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", type=str, default="cuda:0", help='Device map, e.g. "cuda:0", "auto", "cpu".')
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--skip_vlm", action="store_true", help="Skip VLM extraction and rebuild bank from existing image_attributes.jsonl.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing JSONL outputs.")
    parser.add_argument("--flush_every", type=int, default=20)

    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--min_count", type=int, default=5, help="Minimum occurrence count to keep an attribute.")
    parser.add_argument("--max_per_category", type=int, default=40, help="Max attributes per category. Use 0 for no cap.")
    parser.add_argument("--include_sensitive", action="store_true", help="Include age/gender categories. Not recommended by default.")
    parser.add_argument("--categories", type=str, default=None, help="Comma-separated category override.")
    parser.add_argument("--neutral_prompt", type=str, default=DEFAULT_NEUTRAL_PROMPT)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.max_images == 0:
        args.max_images = None
    run(args)
