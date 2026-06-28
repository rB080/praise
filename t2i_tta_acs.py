# t2i_tta_acs.py
import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.utils import Test_parse_args, test_map  # keep same as i2i runner
from utils.metrics import eval_func, cosine_similarity, euclidean_distance  # keep same as i2i runner

from dataloaders.loader_base import ImageDataset_T2I, TextDataset, build_transforms
from dataloaders.ds_assets import RSTPReid, CUHKPEDES

# ---------------------------------------------------------
# Choose the right TTA implementations (T2I versions)
# ---------------------------------------------------------
# IMPORTANT: change these imports to where your T2I ATTR_TTA lives.
# e.g. if you saved my refactor as TTA_t2i/attr_plip_ortho.py:
from TTA_t2i import attr_plip_ortho as attr_plip
from TTA_t2i.attr_plip_ortho import disable_dropout
from TTA_t2i import attr_clipreid_ortho as attr_clip


# ---------------------------------------------------------
# Optional cache dir (same concept as i2i_tta_acs.py)
# ---------------------------------------------------------
CACHE_DIR = "/export/livia/home/vision/Rbhattacharya/work/reid_sandbox/PLIP/saved_features"


# =========================================================
# 1) Dataloaders
# =========================================================
def build_t2i_dataloaders(
    dataset_name: str,
    root: str,
    batch_size: int,
    text_length: int = 250,
    img_size=(256, 128),
    num_workers: int = 4,
):
    """
    T2I: gallery = images, query = captions
    """
    assert dataset_name in ["rstpreid", "cuhkpedes"], "Currently implemented: rstpreid, cuhkpedes. Adjust build_t2i_dataloaders() if using a different dataset."

    if dataset_name == "rstpreid": dataset = RSTPReid(root=root, verbose=True)
    elif dataset_name == "cuhkpedes": dataset = CUHKPEDES(root=root, verbose=True)
    ds = dataset.test

    Gdataset = ImageDataset_T2I(
        ds["image_pids"],
        ds["img_paths"],
        transform=build_transforms(img_size=img_size)
    )
    Qdataset = TextDataset(
        ds["caption_pids"],
        ds["captions"],
        text_length=text_length
    )

    Gloader = DataLoader(
        Gdataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers
    )
    Qloader = DataLoader(
        Qdataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return {"gallery": Gloader, "query": Qloader}


# =========================================================
# 2) Feature extraction helpers
# =========================================================
@torch.no_grad()
def encode_gallery_images(TTA_model, imgs: torch.Tensor, backbone: str):
    """
    Return (B,d) image embeddings. Adjust only here if API differs.
    """
    if backbone == "plip":
        feats = TTA_model.model.get_image_embeddings(imgs)
    elif backbone == "clipreid":
        # clipreid wrapper from your old script
        _, image_features, image_features_proj = TTA_model.model.image_encoder(imgs)
        feats = image_features_proj[:, 0]
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return feats


@torch.no_grad()
def encode_text_noadapt(TTA_model, captions, backbone: str, device):
    """
    Return (B,d) text embeddings for no-adapt baseline.
    (Your PLIP text encoder expects token+mask.)
    """
    if backbone == "plip":
        tokens, mask = TTA_model.tokenize(captions)
        feats, _ = TTA_model.model.text_encoder(tokens.to(device), mask.to(device))
        return feats
    elif backbone == "clipreid":
        # if your clipreid T2I variant supports text, adjust here
        tokens = TTA_model.tokenize(captions)
        feats = TTA_model.clip.encode_text(tokens.to(device))
        return feats
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


@torch.no_grad()
def compute_gallery_features(TTA_model, Gloader, backbone: str, device):
    G_FEATS = []
    G_PIDS = []
    G_PATHS = []
    if TTA_model.training:
        TTA_model.train()
        disable_dropout(TTA_model.model)
        was_training = True
    iterator = tqdm(Gloader, total=len(Gloader), desc="Gallery Precompute (T2I)")
    with torch.no_grad():
        for batch in iterator:
            # ImageDataset_T2I returns (pids, imgs)
            pids, imgs, paths = batch
            imgs = imgs.to(device)

            feats = encode_gallery_images(TTA_model, imgs, backbone=backbone)
            G_FEATS.append(feats.detach().cpu())
            G_PATHS.extend(paths)
            if torch.is_tensor(pids):
                G_PIDS.extend(pids.cpu().tolist())
            else:
                G_PIDS.extend([int(x) for x in pids])
        
        G_FEATS = torch.cat(G_FEATS, dim=0).to(device)
        G_PIDS = np.array(G_PIDS)

    if was_training: TTA_model.train()
    return G_FEATS, G_PIDS, G_PATHS


def maybe_load_cached_t2i(dataset: str, backbone: str, device):
    """
    Expected cache dict keys:
      gallery_features, gallery_pids, query_features, query_pids
    """
    path = os.path.join(CACHE_DIR, f"{dataset}_{backbone}_t2i_test_features.pth")
    if not os.path.exists(path):
        return None

    saved = torch.load(path, map_location="cpu")
    out = {
        "G_FEATS": saved["gallery_features"].to(device),
        "G_PIDS": np.array(saved["gallery_pids"]),
        "Q_FEATS": saved.get("query_features", None),
        "Q_PIDS": np.array(saved.get("query_pids", [])),
    }
    print(f"Loaded cached T2I features from: {path}")
    return out


# =========================================================
# 3) Query loop (T2I)
# =========================================================
def run_query_loop_t2i(
    TTA_model,
    Qloader,
    G_FEATS,
    backbone: str,
    device,
    no_adapt: bool,
    hybrid_eval: bool,
):
    Q_FEATS = []
    Q_PIDS = []
    CAPTIONS = []
    DISTMAT = []
    disable_dropout(TTA_model.model)
    iterator = tqdm(Qloader, total=len(Qloader), desc="Query TTA (T2I)")
    for batch in iterator:
        # TextDataset returns (pids, captions)
        pids, captions = batch

        if no_adapt:
            feats = encode_text_noadapt(TTA_model, captions, backbone=backbone, device=device)
            batch_loss = 0.0
        else:
            if not hybrid_eval:
                feats, batch_loss = TTA_model.forward(captions, G_FEATS)
            else:
                preds, feats, batch_loss = TTA_model.forward(captions, G_FEATS, hybrid_eval=True)
                # breakpoint()
                DISTMAT.append(preds.detach().cpu())

        Q_FEATS.append(feats.detach().cpu())
        CAPTIONS.extend(captions)
        if torch.is_tensor(pids):
            Q_PIDS.extend(pids.cpu().tolist())
        else:
            Q_PIDS.extend([int(x) for x in pids])

        iterator.set_postfix({"loss": batch_loss})

    Q_FEATS = torch.cat(Q_FEATS, dim=0).to(device)
    Q_PIDS = np.array(Q_PIDS)
    return Q_FEATS, Q_PIDS, DISTMAT, CAPTIONS


# =========================================================
# 4) Evaluation (T2I)
# =========================================================
@torch.no_grad()
def evaluate_t2i(Q_FEATS, G_FEATS, Q_PIDS, G_PIDS):
    """
    RSTP-style benchmarks often ignore camera IDs; eval_func requires them.
    We set camids to zeros (no cross-camera filtering).
    """
    Q_FEATS = Q_FEATS.to(torch.float32)
    G_FEATS = G_FEATS.to(torch.float32)

    Q_CAMIDS = np.zeros(len(Q_PIDS), dtype=np.int64)
    G_CAMIDS = np.zeros(len(G_PIDS), dtype=np.int64)

    # Euclidean dist
    dist_euc = euclidean_distance(Q_FEATS, G_FEATS)
    cmc_e, mAP_e = eval_func(dist_euc, Q_PIDS, G_PIDS, Q_CAMIDS, G_CAMIDS)

    # Cosine dist (your utils.metrics.cosine_similarity returns a distance matrix in i2i runner)
    dist_cos = cosine_similarity(Q_FEATS, G_FEATS)
    cmc_c, mAP_c = eval_func(dist_cos, Q_PIDS, G_PIDS, Q_CAMIDS, G_CAMIDS)

    return (cmc_e, mAP_e), (cmc_c, mAP_c)


@torch.no_grad()
def evaluate_hybrid_t2i(DISTMAT_list, Q_PIDS, G_PIDS):
    """
    If hybrid returns similarity/probabilities, we convert to a ranking distance.
    In i2i runner we used distance = -preds (monotone).
    """
    DISTMAT = torch.cat(DISTMAT_list, dim=0).detach().cpu().numpy()
    DISTMAT = -DISTMAT

    Q_CAMIDS = np.zeros(len(Q_PIDS), dtype=np.int64)
    G_CAMIDS = np.zeros(len(G_PIDS), dtype=np.int64)
    cmc_h, mAP_h = eval_func(DISTMAT, Q_PIDS, G_PIDS, Q_CAMIDS, G_CAMIDS)
    return cmc_h, mAP_h


# =========================================================
# 5) Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Run T2I TTA (ACS-style runner)")
    parser.add_argument("--dataset", type=str, default="rstpreid", choices=["rstpreid", "cuhkpedes"])
    parser.add_argument("--root", type=str, default="/export/livia/home/vision/Rbhattacharya/work/data/data/")
    parser.add_argument("--backbone", type=str, default="plip", choices=["plip", "clipreid"])
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--no_adapt", action="store_true")
    parser.add_argument("--hybrid_eval", action="store_true")
    parser.add_argument("--load_gallery", action="store_true")
    parser.add_argument("--savefeats", action="store_true")

    # ---- ACS flags (match i2i runner style) ----
    parser.add_argument("--use_acs", action="store_true", help="Use ACS similarity inside T2I TTA")
    parser.add_argument("--acs_groups", type=str, default="gender,hair,ubcc,lbcc,accessories")
    parser.add_argument("--acs_kmax", type=int, default=10)
    parser.add_argument("--acs_alpha", type=float, default=0.4)
    parser.add_argument("--acs_conf_thresh", type=float, default=0.1)
    parser.add_argument("--acs_use_residual", action="store_true")
    parser.add_argument("--neutral_text", type=str, default="A photo of a person.")

    # ---- core TTA knobs ----
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--id_temp", type=float, default=1.0)

    testargs = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = Test_parse_args(notebook=True)
    print(args)
    print("Device:", device)

    tta_consoles = {
        "plip": attr_plip.ATTR_TTA,
        "clipreid": attr_clip.ATTR_TTA,
    }

    # -------------------------
    # Build loaders
    # -------------------------
    loaders = build_t2i_dataloaders(
        dataset_name=testargs.dataset,
        root=testargs.root,
        batch_size=testargs.batch_size,
        text_length=250,
        img_size=(256, 128),
        num_workers=4,
    )
    Gloader = loaders["gallery"]
    Qloader = loaders["query"]

    # -------------------------
    # Instantiate + init model
    # -------------------------
    tta_args = dict(
        steps=testargs.steps,
        topk=testargs.topk,
        id_temp=testargs.id_temp,
        device=str(device),
        lr=testargs.lr,
        episodic=True,

        # ACS knobs (only if your T2I ATTR_TTA supports them)
        use_acs=testargs.use_acs,
        acs_groups=[g.strip() for g in testargs.acs_groups.split(",") if g.strip()],
        acs_kmax=testargs.acs_kmax,
        acs_alpha=testargs.acs_alpha,
        acs_conf_thresh=testargs.acs_conf_thresh,
        acs_use_residual=testargs.acs_use_residual,
        neutral_text=testargs.neutral_text,
    )

    TTA_model = tta_consoles[testargs.backbone](**tta_args).to(device)
    TTA_model.init_model()
    # TTA_model.init_model("/export/livia/home/vision/Rbhattacharya/work/reid_sandbox/CLIP-ReID/outputs/downloaded_models/MSMT17_clipreid_ViT-B-16_60.pth")

    # Precompute ACS attribute bank ONCE (atomic prompts)
    if testargs.use_acs:
        if hasattr(TTA_model, "precompute_attr_bank"):
            TTA_model.precompute_attr_bank()
        else:
            raise AttributeError("use_acs=True but ATTR_TTA lacks precompute_attr_bank().")

    # -------------------------
    # Load or compute gallery feats
    # -------------------------
    cached = None
    if testargs.load_gallery:
        cached = maybe_load_cached_t2i(testargs.dataset, testargs.backbone, device)

    if cached is not None:
        G_FEATS = cached["G_FEATS"]
        G_PIDS = cached["G_PIDS"]
        print(f"Loaded gallery feats: {G_FEATS.shape}")
    else:
        print("Computing gallery features...")
        G_FEATS, G_PIDS, gimgpaths = compute_gallery_features(TTA_model, Gloader, testargs.backbone, device)
        print(f"Computed gallery feats: {G_FEATS.shape}")

    # Initialize TTA optimizer (norm affine params)
    if hasattr(TTA_model, "initialize"):
        TTA_model.initialize(gallery_feats=G_FEATS)

    # -------------------------
    # Run query loop
    # -------------------------
    if testargs.no_adapt:
        print("Running NO-ADAPT baseline (no TTA).")
    else:
        print("Running T2I TTA.")

    Q_FEATS, Q_PIDS, DISTMAT_list, CAPTIONS = run_query_loop_t2i(
        TTA_model=TTA_model,
        Qloader=Qloader,
        G_FEATS=G_FEATS,
        backbone=testargs.backbone,
        device=device,
        no_adapt=testargs.no_adapt,
        hybrid_eval=testargs.hybrid_eval,
    )

    # =========================
    # Evals (MATCH ORIGINAL T2I)
    # =========================
    with torch.no_grad():
        Q_FEATS = Q_FEATS.to(torch.float32)
        G_FEATS = G_FEATS.to(torch.float32)

        # In your original script you used test_map() as the main eval.
        # Keep Q_PIDS/G_PIDS as torch tensors like before for maximum compatibility.
        if not torch.is_tensor(Q_PIDS):
            Q_PIDS_t = torch.tensor(Q_PIDS)
        else:
            Q_PIDS_t = Q_PIDS

        if not torch.is_tensor(G_PIDS):
            G_PIDS_t = torch.tensor(G_PIDS)
        else:
            G_PIDS_t = G_PIDS

        print(f"Evaluation results for {testargs.dataset} dataset with Cosine distance:")
        scores = test_map(Q_FEATS, Q_PIDS_t, G_FEATS, G_PIDS_t)

        # =========================
        # Hybrid eval (MATCH ORIGINAL)
        # =========================
        if testargs.hybrid_eval:
            print("Hybrid eval enabled: saving preds/distmat, but evaluation still uses test_map().")

            DM = torch.cat(DISTMAT_list, dim=0).to(device)
            # Evaluate same as original (feature-based)
            with torch.no_grad():
                Q_FEATS= Q_FEATS.to(torch.float32)
                G_FEATS = G_FEATS.to(torch.float32)
                print(f"Evaluation results for {testargs.dataset} dataset (Hybrid flag ON) with Cosine distance:")
                scores = test_map(Q_FEATS, Q_PIDS_t, G_FEATS, G_PIDS_t, DM)


    if testargs.savefeats:
        save_path = os.path.join(CACHE_DIR, f"{testargs.dataset}{'_noadapt' if testargs.no_adapt else ''}_test_features_t2i.pth")
        torch.save({
            "gallery_features": G_FEATS.detach().cpu(),
            "gallery_pids": G_PIDS,
            "gimgpaths": gimgpaths,
            "query_features": Q_FEATS.detach().cpu(),
            "query_pids": Q_PIDS,
            "captions": CAPTIONS,
        }, save_path)
        print(f"Saved features to: {save_path}")


if __name__ == "__main__":
    main()
