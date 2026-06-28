import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision
from PIL import Image

from utils.utils import Test_parse_args
from utils.metrics import euclidean_distance, cosine_similarity, eval_func

from dataloaders.loader_base import ImageDataset, val_collate_fn
from dataloaders.ds_assets import _process_dir_duke, _process_dir_market, _process_dir_msmt

# ---------------------------------------------------------
# Choose the right TTA implementations
# ---------------------------------------------------------
# If you implemented ACS inside attr_plip_ortho.py, keep this import.
# If you made a new file, change to: from TTA import attr_plip_acs as attr_plip
from TTA import attr_plip_ortho as attr_plip
from TTA import attr_dino_ortho
from TTA import attr_clipreid_ortho as attr_clip  # if you also add ACS there later


# ---------------------------------------------------------
# Paths (keep your existing ones)
# ---------------------------------------------------------
MARKET_ROOT = "/export/livia/home/vision/Rbhattacharya/work/data/data/market1501/Market-1501-v15.09.15"
DUKE_ROOT   = "/export/livia/home/vision/Rbhattacharya/work/data/data/dukemtmcreid/DukeMTMC-reID"
MSMT_ROOT   = "/export/livia/home/vision/Rbhattacharya/work/data/data/msmt17/MSMT17"

# Optional: cached gallery/query features
CACHE_DIR = "/export/livia/home/vision/Rbhattacharya/work/tdop/saved_features"


# =========================================================
# 1) Dataloaders: build ONLY the requested dataset
# =========================================================
def build_dataloaders(dataset_name: str, batch_size: int, backbone: str):
    assert dataset_name in ["market", "duke", "msmt"]
    path_only = (backbone == "qwen")  # you used this logic

    if dataset_name == "market":
        g_dir = os.path.join(MARKET_ROOT, "bounding_box_test")
        q_dir = os.path.join(MARKET_ROOT, "query")

        G_ds = _process_dir_market(g_dir)  # or _process_dir_market() if default points to bbox_test
        Q_ds = _process_dir_market(q_dir)

    elif dataset_name == "duke":
        g_dir = os.path.join(DUKE_ROOT, "bounding_box_test")
        q_dir = os.path.join(DUKE_ROOT, "query")

        G_ds = _process_dir_duke(g_dir)
        Q_ds = _process_dir_duke(q_dir)

    else:  # msmt
        g_list = os.path.join(MSMT_ROOT, "list_gallery.txt")
        q_list = os.path.join(MSMT_ROOT, "list_query.txt")
        test_root = os.path.join(MSMT_ROOT, "test")

        G_ds = _process_dir_msmt(test_root, g_list)
        Q_ds = _process_dir_msmt(test_root, q_list)

    if backbone == "dinotxt":
        TF = torchvision.transforms.Compose([
            torchvision.transforms.Resize(518, interpolation=Image.BICUBIC),
            torchvision.transforms.CenterCrop(518),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])
    else:
        TF = None

    Gdataset = ImageDataset(G_ds, transform=TF, path_only=path_only)
    Qdataset = ImageDataset(Q_ds, transform=TF, path_only=path_only)

    Gloader = DataLoader(
        Gdataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
        collate_fn=val_collate_fn
    )
    Qloader = DataLoader(
        Qdataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=val_collate_fn
    )

    return {"gallery": Gloader, "query": Qloader}


# =========================================================
# 2) Feature extraction helpers (gallery/query)
# =========================================================
@torch.no_grad()
def encode_images(TTA_model, imgs, backbone: str):
    """
    Return (B,d) features for imgs.
    Adjust only here if your backbone APIs differ.
    """
    if backbone == "plip":
        feats = TTA_model.model.get_image_embeddings(imgs)

    elif backbone == "clipreid":
        _, image_features, image_features_proj = TTA_model.model.image_encoder(imgs)
        feats = image_features_proj[:, 0]

    elif backbone == "qwen":
        feats = TTA_model.model.encode_image(imgs)
    elif backbone == "dinotxt":
        feats = TTA_model.model.encode_image(imgs)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    return feats


@torch.no_grad()
def compute_gallery_features(TTA_model, Gloader, backbone: str, device):
    G_FEATS, G_PIDS, G_CAMIDS, gimgpaths = [], [], [], []
    iterator = tqdm(Gloader, total=len(Gloader), desc="Gallery Precompute")
    for batch in iterator:
        imgs, pids, camids, camids_batch, viewids, img_paths = batch
        if backbone != "qwen":
            imgs = imgs.to(device)

        feats = encode_images(TTA_model, imgs, backbone=backbone)
        G_FEATS.append(feats.detach().cpu())
        G_PIDS.extend(pids)
        G_CAMIDS.extend(camids)
        gimgpaths.extend(img_paths)

    G_FEATS = torch.cat(G_FEATS, dim=0).to(device)
    G_PIDS = np.array(G_PIDS)
    G_CAMIDS = np.array(G_CAMIDS)
    return G_FEATS, G_PIDS, G_CAMIDS, gimgpaths


def maybe_load_cached_features(dataset: str, backbone: str, device):
    """
    Optional: you already have a cache format in your older scripts.
    Adjust filename to your convention.
    """
    path = os.path.join(CACHE_DIR, f"{dataset}_{backbone}_test_features.pth")
    if not os.path.exists(path):
        return None

    saved = torch.load(path, map_location="cpu")
    out = {
        "G_FEATS": saved["gallery_features"].to(device),
        "G_PIDS": np.array(saved["gallery_pids"]),
        "G_CAMIDS": np.array(saved["gallery_camids"]),
        "gimgpaths": saved.get("gimgpaths", []),
        "Q_FEATS": saved.get("query_features", None),
        "Q_PIDS": np.array(saved.get("query_pids", [])),
        "Q_CAMIDS": np.array(saved.get("query_camids", [])),
        "qimgpaths": saved.get("qimgpaths", []),
    }
    print(f"Loaded cached features from: {path}")
    return out


# =========================================================
# 3) Query loop: with or without adaptation
# =========================================================
def run_query_loop(
    TTA_model,
    Qloader,
    G_FEATS,
    G_PIDS,
    backbone: str,
    device,
    no_adapt: bool,
    hybrid_eval: bool,
    use_oracle: bool,
    use_PL: bool,
):
    Q_FEATS, Q_PIDS, Q_CAMIDS, qimgpaths = [], [], [], []
    DISTMAT = []
    
    iterator = tqdm(Qloader, total=len(Qloader), desc="Query TTA")
    for batch in iterator:
        imgs, pids, camids, camids_batch, viewids, img_paths = batch
        if backbone != "qwen":
            imgs = imgs.to(device)

        if no_adapt:
            feats = encode_images(TTA_model, imgs, backbone=backbone)
            batch_loss = 0.0
            preds = None
        else:
            if not hybrid_eval:
                if (use_PL or use_oracle) is False:
                    feats, batch_loss = TTA_model.forward(imgs, G_FEATS)
                else:
                    # same contract you already used:
                    # no_oracle=False => oracle_mode, no_oracle=True => PL mode (mask_using_id)
                    feats, batch_loss = TTA_model.forward(
                        imgs, G_FEATS,
                        batch_pids=pids,
                        gpids=G_PIDS,
                        no_oracle=use_PL
                    )
            else:
                preds, feats, batch_loss = TTA_model.forward(imgs, G_FEATS, hybrid_eval=True)
                DISTMAT.append(preds.detach().cpu())

        Q_FEATS.append(feats.detach().cpu())
        Q_PIDS.extend(pids)
        Q_CAMIDS.extend(camids)
        qimgpaths.extend(img_paths)
        iterator.set_postfix({"loss": batch_loss})

    Q_FEATS = torch.cat(Q_FEATS, dim=0).to(device)
    return Q_FEATS, np.array(Q_PIDS), np.array(Q_CAMIDS), qimgpaths, DISTMAT


# =========================================================
# 4) Evaluation helpers
# =========================================================
@torch.no_grad()
def evaluate(Q_FEATS, G_FEATS, Q_PIDS, G_PIDS, Q_CAMIDS, G_CAMIDS):
    Q_FEATS = Q_FEATS.to(torch.float32)
    G_FEATS = G_FEATS.to(torch.float32)

    dist_euc = euclidean_distance(Q_FEATS, G_FEATS)
    cmc_e, mAP_e = eval_func(dist_euc, Q_PIDS, G_PIDS, Q_CAMIDS, G_CAMIDS)
    del dist_euc  # free memory
    dist_cos = cosine_similarity(Q_FEATS, G_FEATS)
    cmc_c, mAP_c = eval_func(dist_cos, Q_PIDS, G_PIDS, Q_CAMIDS, G_CAMIDS)
    del dist_cos  # free memory
    return (cmc_e, mAP_e), (cmc_c, mAP_c)


@torch.no_grad()
def evaluate_hybrid(DISTMAT_list, Q_PIDS, G_PIDS, Q_CAMIDS, G_CAMIDS):
    """
    Your old code did: DISTMAT = 1 - preds
    That only makes sense if preds is a similarity in [0,1].
    If your ACS hybrid returns a probability distribution, it is NOT a distance.
    Recommended:
      - if hybrid returns similarities S in [-1,1], set distance = -S
      - if hybrid returns probs p, set distance = -p (monotone for ranking)
    Here we keep your original behavior but safer: distance = -preds.
    """
    # DISTMAT = torch.cat(DISTMAT_list, dim=0).detach().cpu().numpy() # do this only if distmat_list is list
    DISTMAT = torch.cat(DISTMAT_list, dim=0).detach().cpu().numpy() if isinstance(DISTMAT_list, list) else DISTMAT_list.detach().cpu().numpy()
    # safer monotone distance for ranking:
    DISTMAT = -DISTMAT
    cmc_h, mAP_h = eval_func(DISTMAT, Q_PIDS, G_PIDS, Q_CAMIDS, G_CAMIDS)
    return cmc_h, mAP_h


# =========================================================
# 5) Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Run TTA with ACS restructuring")
    parser.add_argument("--dataset", type=str, default="market", choices=["market", "duke", "msmt"])
    parser.add_argument("--backbone", type=str, default="plip", choices=["plip", "clipreid", "qwen", "dinotxt"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--no_adapt", action="store_true")
    parser.add_argument("--use_oracle", action="store_true")
    parser.add_argument("--use_PL", action="store_true")
    parser.add_argument("--hybrid_eval", action="store_true")
    parser.add_argument("--load_gallery", action="store_true")
    parser.add_argument("--savefeats", action="store_true")
    parser.add_argument("--ablation_mode", action="store_true")
    parser.add_argument("--learn_text", action="store_true")  # keep for non-ACS variants
    parser.add_argument("--model_path", type=str, default=None, help="Path to pre-trained model checkpoint (optional)")

    # ---- ACS flags (new) ----
    parser.add_argument("--use_acs", action="store_true", help="Use ACS similarity inside TTA")
    parser.add_argument("--acs_groups", type=str, default="gender,hair,ubcc,lbcc,accessories")
    parser.add_argument("--acs_kmax", type=int, default=10)
    parser.add_argument("--acs_alpha", type=float, default=0.4)
    parser.add_argument("--acs_conf_thresh", type=float, default=0.1)
    parser.add_argument("--acs_use_residual", action="store_true")

    # ---- core TTA knobs ----
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--id_temp", type=float, default=10.0)

    testargs = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = Test_parse_args(notebook=True)
    print(args)
    print("Device:", device)

    # Choose TTA class
    tta_consoles = {
        "plip": attr_plip.ATTR_TTA,
        "clipreid": attr_clip.ATTR_TTA,
        "qwen": None,  # placeholder if you add ACS to your Qwen TTA later
        "dinotxt": attr_dino_ortho.ATTR_TTA,
    }
    assert testargs.backbone in tta_consoles, "Only plip/clipreid supported by this runner."

    # Build dataloaders only for the chosen dataset
    loaders = build_dataloaders(testargs.dataset, testargs.batch_size, testargs.backbone)
    Gloader = loaders["gallery"]
    Qloader = loaders["query"]

    # -------------------------
    # Instantiate and init model
    # -------------------------
    tta_args = dict(
        steps=testargs.steps,
        acs_alpha=0.0,           # not used in ACS; keep for compatibility
        topk=testargs.topk, # was 6
        id_temp=testargs.id_temp,
        # att_temp=0.005,      # not used in ACS; keep for compatibility
        device=device,
        lr=testargs.lr,
        episodic=True, 
        # vectorize_alpha=False,
    )

    # If your ATTR_TTA constructor supports ACS args, pass them here.
    # If not, you can set them as attributes after construction.
    if testargs.use_acs:
        tta_args.update(dict(
            use_acs=True,
            acs_groups=[g.strip() for g in testargs.acs_groups.split(",") if g.strip()],
            acs_kmax=testargs.acs_kmax,
            acs_alpha=testargs.acs_alpha,
            acs_conf_thresh=testargs.acs_conf_thresh,
            acs_use_residual=testargs.acs_use_residual,
            neutral_text="A photo of a person.",
        ))

    TTA_model = tta_consoles[testargs.backbone](**tta_args, num_class=751 if "Market" in testargs.model_path else 1041).to(device)
    TTA_model.init_model(model_path=testargs.model_path)  # your existing init_model should accept a path argument
    # TTA_model.init_model("/export/livia/home/vision/Rbhattacharya/work/reid_sandbox/CLIP-ReID/outputs/downloaded_models/Market1501_clipreid_ViT-B-16_60.pth")

    # ACS bank precompute (atomic prompts) — do this ONCE
    if testargs.use_acs:
        # this method must exist in your updated class
        TTA_model.precompute_attr_bank()

    # -------------------------
    # Load or compute gallery feats
    # -------------------------
    cached = None
    if testargs.load_gallery:
        cached = maybe_load_cached_features(testargs.dataset, testargs.backbone, device)

    if cached is not None:
        G_FEATS = cached["G_FEATS"]
        G_PIDS = cached["G_PIDS"]
        G_CAMIDS = cached["G_CAMIDS"]
        gimgpaths = cached["gimgpaths"]
        print(f"Loaded gallery feats: {G_FEATS.shape}")
    else:
        print("Computing gallery features...")
        G_FEATS, G_PIDS, G_CAMIDS, gimgpaths = compute_gallery_features(
            TTA_model, Gloader, backbone=testargs.backbone, device=device
        )
        print(f"Computed gallery feats: {G_FEATS.shape}")

    # Initialize TTA optimizer (LN/BN params)
    # For ACS, DO NOT set_learnable_text=True (unless you intentionally want that).
    if hasattr(TTA_model, "initialize"):
        if testargs.use_acs:
            TTA_model.initialize(gallery_feats=G_FEATS, set_learnable_text=False)
        else:
            TTA_model.initialize(gallery_feats=G_FEATS, set_learnable_text=testargs.learn_text)

    # -------------------------
    # Run query loop
    # -------------------------
    if not testargs.ablation_mode:
        Q_FEATS, Q_PIDS, Q_CAMIDS, qimgpaths, DISTMAT_list = run_query_loop(
            TTA_model=TTA_model,
            Qloader=Qloader,
            G_FEATS=G_FEATS,
            G_PIDS=G_PIDS,
            backbone=testargs.backbone,
            device=device,
            no_adapt=testargs.no_adapt,
            hybrid_eval=testargs.hybrid_eval,
            use_oracle=testargs.use_oracle,
            use_PL=testargs.use_PL,
        )
        
        gset = set(gimgpaths)
        qset = set(qimgpaths)
        overlap = len(gset & qset)
        print("G/Q overlap:", overlap)
        assert overlap == 0, "Leakage: some query images are in the gallery."
        # Evals
        (cmc_e, mAP_e), (cmc_c, mAP_c) = evaluate(Q_FEATS, G_FEATS, Q_PIDS, G_PIDS, Q_CAMIDS, G_CAMIDS)
        print(f"\n[{testargs.dataset}] Euclidean: mAP={mAP_e:.2%}, R1={cmc_e[0]:.2%}, R5={cmc_e[4]:.2%}, R10={cmc_e[9]:.2%}")
        print(f"[{testargs.dataset}] Cosine:    mAP={mAP_c:.2%}, R1={cmc_c[0]:.2%}, R5={cmc_c[4]:.2%}, R10={cmc_c[9]:.2%}")

        if testargs.hybrid_eval and len(DISTMAT_list) > 0:
            cmc_h, mAP_h = evaluate_hybrid(DISTMAT_list, Q_PIDS, G_PIDS, Q_CAMIDS, G_CAMIDS)
            print(f"[{testargs.dataset}] Hybrid:   mAP={mAP_h:.2%}, R1={cmc_h[0]:.2%}, R5={cmc_h[4]:.2%}, R10={cmc_h[9]:.2%}")

    if testargs.savefeats:
        save_path = os.path.join(CACHE_DIR, f"{testargs.dataset}_{testargs.backbone}{'_noadapt' if testargs.no_adapt else ''}_test_features.pth")
        torch.save({
            "gallery_features": G_FEATS.detach().cpu(),
            "gallery_pids": G_PIDS,
            "gallery_camids": G_CAMIDS,
            "gimgpaths": gimgpaths,
            "query_features": Q_FEATS.detach().cpu(),
            "query_pids": Q_PIDS,
            "query_camids": Q_CAMIDS,
            "qimgpaths": qimgpaths,
        }, save_path)
        print(f"Saved features to: {save_path}")


if __name__ == "__main__":
    main()
