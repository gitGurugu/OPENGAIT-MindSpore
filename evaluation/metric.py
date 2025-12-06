import mindspore as ms
import numpy as np
import mindspore.ops as ops

from utils import is_tensor


def cuda_dist(x, y, metric='euc'):
    # Convert numpy to MindSpore Tensor
    if isinstance(x, np.ndarray):
        x = ms.Tensor(x, dtype=ms.float32)
    if isinstance(y, np.ndarray):
        y = ms.Tensor(y, dtype=ms.float32)
    
    # Set device context (MindSpore handles this automatically in graph mode)
    if metric == 'cos':
        x = ops.L2Normalize(axis=1)(x)  # n c p
        y = ops.L2Normalize(axis=1)(y)  # n c p
    num_bin = x.shape[2]
    n_x = x.shape[0]
    n_y = y.shape[0]
    dist = ops.Zeros()((n_x, n_y), ms.float32)
    for i in range(num_bin):
        _x = x[:, :, i]
        _y = y[:, :, i]
        if metric == 'cos':
            dist = dist + ops.BatchMatMul()(_x, _y.transpose(1, 0))
        else:
            _dist = (_x ** 2).sum(axis=1, keepdims=True) + (_y ** 2).sum(axis=1, keepdims=True).transpose(1, 0) - 2 * ops.BatchMatMul()(_x, _y.transpose(1, 0))
            dist = dist + ops.Sqrt()(ops.ReLU()(_dist))
    if metric == 'cos':
        return 1 - dist/num_bin
    else:
        return dist / num_bin


def mean_iou(msk1, msk2, eps=1.0e-9):
    if not is_tensor(msk1):
        msk1 = ms.Tensor(msk1, dtype=ms.float32)
    if not is_tensor(msk2):
        msk2 = ms.Tensor(msk2, dtype=ms.float32)
    n = msk1.shape[0]
    inter = msk1 * msk2
    union = ((msk1 + msk2) > 0.).astype(ms.float32)
    miou = inter.reshape(n, -1).sum(axis=-1) / (union.reshape(n, -1).sum(axis=-1) + eps)
    return miou.asnumpy()


def compute_ACC_mAP(distmat, q_pids, g_pids, q_views=None, g_views=None, rank=1):
    num_q, _ = distmat.shape
    # Convert to numpy if needed
    if isinstance(distmat, ms.Tensor):
        distmat = distmat.asnumpy()
    if isinstance(q_pids, ms.Tensor):
        q_pids = q_pids.asnumpy()
    if isinstance(g_pids, ms.Tensor):
        g_pids = g_pids.asnumpy()

    all_ACC = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        q_idx_dist = distmat[q_idx]
        q_idx_glabels = g_pids
        if q_views is not None and g_views is not None:
            q_idx_mask = np.isin(g_views, q_views[q_idx], invert=True) | np.isin(
                g_pids, q_pids[q_idx], invert=True)
            q_idx_dist = q_idx_dist[q_idx_mask]
            q_idx_glabels = q_idx_glabels[q_idx_mask]

        assert(len(q_idx_glabels) >
               0), "No gallery after excluding identical-view cases!"
        q_idx_indices = np.argsort(q_idx_dist)
        q_idx_matches = (q_idx_glabels[q_idx_indices]
                         == q_pids[q_idx]).astype(np.int32)

        # binary vector, positions with value 1 are correct matches
        orig_cmc = q_idx_matches
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_ACC.append(cmc[rank-1])

        # compute average precision
        num_rel = orig_cmc.sum()

        if num_rel > 0:
            num_valid_q += 1.
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

    ACC = np.mean(all_ACC)
    mAP = np.mean(all_AP)

    return ACC, mAP


def evaluate_rank(distmat, p_lbls, g_lbls, max_rank=50):
    '''
    Copy from https://github.com/Gait3D/Gait3D-Benchmark/blob/72beab994c137b902d826f4b9f9e95b107bebd78/lib/utils/rank.py#L12-L63
    '''
    # Convert to numpy if needed
    if isinstance(distmat, ms.Tensor):
        distmat = distmat.asnumpy()
    if isinstance(p_lbls, ms.Tensor):
        p_lbls = p_lbls.asnumpy()
    if isinstance(g_lbls, ms.Tensor):
        g_lbls = g_lbls.asnumpy()
        
    num_p, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    indices = np.argsort(distmat, axis=1)

    matches = (g_lbls[indices] == p_lbls[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each probe
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_p = 0.  # number of valid probe

    for p_idx in range(num_p):
        # compute cmc curve
        raw_cmc = matches[p_idx]
        if not np.any(raw_cmc):
            continue

        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_p += 1.

        # compute average precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_p > 0, 'Error: all probe identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_p

    return all_cmc, all_AP, all_INP


def evaluate_many(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    # Convert to numpy if needed
    if isinstance(distmat, ms.Tensor):
        distmat = distmat.asnumpy()
    if isinstance(q_pids, ms.Tensor):
        q_pids = q_pids.asnumpy()
    if isinstance(g_pids, ms.Tensor):
        g_pids = g_pids.asnumpy()
    if isinstance(q_camids, ms.Tensor):
        q_camids = q_camids.asnumpy()
    if isinstance(g_camids, ms.Tensor):
        g_camids = g_camids.asnumpy()
        
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(
        np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()

        pos_idx = np.where(orig_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    return all_cmc, mAP, mINP

