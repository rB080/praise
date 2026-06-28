import torch
import numpy as np
import os
# from utils.reranking import re_ranking
from tqdm import tqdm


def euclidean_distance(qf, gf, modified=False, return_tensor=False):
    if modified:
        qf = qf.unsqueeze(0)
    m = qf.shape[0]
    n = gf.shape[0]
    #breakpoint()
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    if return_tensor:
        return dist_mat
    return dist_mat.detach().cpu().numpy()

def euclidean_distance_with_descriptors(qf, gf, textfeats, pids):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = []
    with torch.no_grad():
        for i in tqdm(range(m), total=m, desc="Computing Euclidean distances with text"):
            #breakpoint()
            q = qf[i].unsqueeze(0).cuda()
            q *= textfeats[pids[i]].unsqueeze(0).cuda()
            tf = textfeats[pids[i]].repeat(n, 1).cuda()
            g = gf.cuda() * tf
            dist_mat.append(euclidean_distance(q, g)[0])
        dist_mat = np.array(dist_mat)
    return dist_mat

# def euclidean_distance_with_descriptors(qf, gf, textfeats, pids, batch_size=32):
#     """
#     Computes pairwise Euclidean distance matrix with text-based descriptors in batches.
    
#     Args:
#         qf (torch.Tensor): Query features, shape (m, d).
#         gf (torch.Tensor): Gallery features, shape (n, d).
#         textfeats (torch.Tensor): Text feature descriptors, shape (k, d).
#         pids (list): List of person IDs for each query, length m.
#         batch_size (int): Batch size for query features.

#     Returns:
#         numpy.ndarray: Pairwise Euclidean distance matrix, shape (m, n).
#     """
#     m, d = qf.shape
#     n = gf.shape[0]
#     dist_mat = []

#     # Precompute modified gallery features based on text features
#     with torch.no_grad():
#         gf = gf.cuda()
#         # Batch process the queries
#         for start in tqdm(range(0, m, batch_size), desc="Processing batches: Computing Euclidean distances with text"):
#             end = min(start + batch_size, m)
#             batch_qf = qf[start:end].cuda()  # Load batch of queries
#             batch_pids = pids[start:end]
#             #breakpoint()
#             # Scale query features with corresponding text descriptors
#             q_text_scaled = torch.stack(
#                 [batch_qf[i] * textfeats[batch_pids[i]].cuda() for i in range(len(batch_pids))]
#             )
            
#             # Repeat text descriptors for gallery features
#             gf_text_scaled = torch.stack(
#                 [gf * textfeats[pid].cuda() for pid in batch_pids]
#             )
            
#             # Compute Euclidean distances batch-wise
#             batch_distances = []
#             for bq, gf_scaled in zip(q_text_scaled, gf_text_scaled):
#                 dist = euclidean_distance(bq, gf_scaled, True)  # Adjusted distance computation
#                 batch_distances.append(dist)
            
#             dist_mat.append(np.stack(batch_distances))

#     dist_mat = np.vstack(dist_mat)[:, 0, :]
    
#     return dist_mat




def cosine_similarity(qf, gf, modified=False, return_tensor=False, invert=True):
    if modified:
        qf = qf.unsqueeze(0)
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot)#.cpu().numpy()
    dist_mat = torch.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    if invert: dist_mat = torch.arccos(dist_mat)
    if return_tensor:
        return dist_mat
    else:
        return dist_mat.detach().cpu().numpy()
    

def mahalanobis_distance(qf, gf, modified=False, return_tensor=False):
    """
    Compute Mahalanobis distance matrix between qf (B, D) and gf (N, D) using inverse covariance matrix S_inv (D, D).

    Returns:
        Tensor of shape (B, N) where entry (i, j) is the Mahalanobis distance between qf[i] and gf[j].
    """
    if modified:
        qf = qf.unsqueeze(0)
    # Center the data
    G_centered = gf - gf.mean(dim=0, keepdim=True)

    # Compute covariance matrix
    cov = (G_centered.T @ G_centered) / (qf.shape[0] - 1)

    # Invert the covariance matrix
    S_inv = torch.linalg.pinv(cov)  # Use pinv for numerical stability

    # Compute projections
    Q_proj = qf @ S_inv          # shape: (B, D)
    G_proj = gf @ S_inv          # shape: (N, D)

    # Compute squared terms
    Q_sq = (Q_proj * qf).sum(dim=1)       # shape: (B,)
    G_sq = (G_proj * gf).sum(dim=1)       # shape: (N,)

    # Cross term
    cross_term = Q_proj @ gf.T            # shape: (B, N)

    # Compute squared Mahalanobis distances
    D_sq = Q_sq[:, None] - 2 * cross_term + G_sq[None, :]

    # Numerical stability and final distance
    dist_mat =  torch.sqrt(torch.clamp(D_sq, min=1e-8))
    if return_tensor:
        return dist_mat
    else:
        return dist_mat.detach().cpu().numpy()



def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, exclude_cam=None):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # breakpoint()
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    #breakpoint()
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    removed = 0
    
    for q_idx in tqdm(range(num_q), total=num_q):
        #breakpoint()
        if exclude_cam is not None:
            q_camids_ignore = [1 if x == exclude_cam else 0 for x in q_camids]
            if q_camids_ignore[q_idx] != 1: continue
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        removed += remove.sum()
    print(f"Removed: {removed}/{num_q}; Total valid queries: {num_valid_q}/{num_q}")

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    #print("CAM, mAP, R1, R5, R10: ", exclude_cam, mAP, all_cmc[0], all_cmc[4], all_cmc[9])
    return all_cmc, mAP


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False, dist="euclidean"):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.dist_type = dist

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.textfeats = []

    def update(self, output):  # called once for each batch
        if len(output) == 3: 
            feat, pid, camid = output
        else: 
            feat, pid, camid, tfeat = output
            self.textfeats.append(tfeat.cpu())
        
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self, textfeats=None):  # called after each epoch
        # breakpoint()
        if not torch.is_tensor(self.feats): feats = torch.cat(self.feats, dim=0)
        else: feats = self.feats
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])
        if self.reranking:
            print('=> Enter reranking')
        #     # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        #     distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            if textfeats is None:
                if self.dist_type == "euclidean": distmat = euclidean_distance(qf, gf)
                elif self.dist_type == "cosine": distmat = cosine_similarity(qf, gf)
            else:
                distmat = euclidean_distance_with_descriptors(qf, gf, textfeats, q_pids)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, exclude_cam=None)
        # cmcs, mAPs = [], []
        # for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
        #     cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, exclude_cam=i)
        #     cmcs.append(cmc)
        #     mAPs.append(mAP)
        return cmc, mAP, distmat, self.pids, self.camids, qf, gf



