"""
Barnes-Hut法 O(N log N) — フラット八分木 + ベクトル化BFS
"""

from collections import deque

import numpy as np
from ..utils.core import Particles, G, SOFTENING

# child octant offset table: shape (8, 3), values in {-1, +1}
_OI = np.arange(8)
_OFFSETS = np.stack([
    (_OI >> 2 & 1) * 2 - 1,
    (_OI >> 1 & 1) * 2 - 1,
    (_OI      & 1) * 2 - 1,
], axis=1).astype(np.float64)  # (8, 3)


def _build_flat_octree(pos: np.ndarray, mass: np.ndarray):
    """
    八分木をフラット配列として構築する（BFS）。

    Returns
    -------
    com        : (M, 3)  各ノードの質量中心
    node_mass  : (M,)    各ノードの総質量
    node_half  : (M,)    各ノードの半辺長
    children   : (M, 8)  子ノードインデックス（存在しない場合は -1）
    leaf_idx   : (M,)    葉ノードの粒子インデックス（内部ノードは -1）
    """
    N = len(mass)
    max_nodes = max(4 * N, 64)

    com       = np.zeros((max_nodes, 3))
    node_mass = np.zeros(max_nodes)
    node_half = np.zeros(max_nodes)
    node_ctr  = np.zeros((max_nodes, 3))
    children  = np.full((max_nodes, 8), -1, dtype=np.int32)
    leaf_idx  = np.full(max_nodes, -1, dtype=np.int32)

    root_ctr  = (pos.max(axis=0) + pos.min(axis=0)) * 0.5
    root_half = (pos.max(axis=0) - pos.min(axis=0)).max() * 0.5 * 1.01 + 1e-8
    node_ctr[0]  = root_ctr
    node_half[0] = root_half
    n = 1

    queue = deque([(0, np.arange(N, dtype=np.int32))])

    while queue:
        ni, idx = queue.popleft()
        sub_pos  = pos[idx]
        sub_mass = mass[idx]

        tm = sub_mass.sum()
        node_mass[ni] = tm
        com[ni] = (sub_pos * sub_mass[:, None]).sum(axis=0) / tm

        if len(idx) == 1:
            leaf_idx[ni] = idx[0]
            continue

        c = node_ctr[ni]
        q = node_half[ni] * 0.5

        # 全粒子のオクタントをベクトル化で一括計算
        bits    = (sub_pos > c).astype(np.int32)           # (K, 3)
        octants = bits[:, 0] * 4 + bits[:, 1] * 2 + bits[:, 2]  # (K,)
        child_ctrs = c + _OFFSETS * q                      # (8, 3)

        for oi in range(8):
            mask = octants == oi
            if not mask.any():
                continue
            if n >= max_nodes:
                extra = max_nodes
                com       = np.vstack([com,      np.zeros((extra, 3))])
                node_mass = np.concatenate([node_mass, np.zeros(extra)])
                node_half = np.concatenate([node_half, np.zeros(extra)])
                node_ctr  = np.vstack([node_ctr, np.zeros((extra, 3))])
                children  = np.vstack([children, np.full((extra, 8), -1, dtype=np.int32)])
                leaf_idx  = np.concatenate([leaf_idx, np.full(extra, -1, dtype=np.int32)])
                max_nodes *= 2
            ci = n; n += 1
            node_ctr[ci]  = child_ctrs[oi]
            node_half[ci] = q
            children[ni, oi] = ci
            queue.append((ci, idx[mask]))

    return com[:n], node_mass[:n], node_half[:n], children[:n], leaf_idx[:n]


def compute_acceleration_barneshut(p: Particles, theta: float = 0.5) -> np.ndarray:
    """
    Barnes-Hut法で全粒子の加速度を計算。

    ベクトル化BFS: 全 (粒子, ノード) ペアを一括処理し、
    - 開き角基準を満たすペア → 質量中心近似で力を計算
    - 満たさないペア → 子ノードへ展開
    Python ループは木の深さ O(log N) 回のみ。

    Parameters
    ----------
    theta : float
        開き角パラメータ（小さいほど精度高・速度低）
    """
    com, node_mass, node_half, node_children, node_leaf_idx = \
        _build_flat_octree(p.pos, p.mass)

    N   = p.N
    acc = np.zeros((N, 3))

    # 全粒子がルート（ノード0）からスタート
    act_par = np.arange(N, dtype=np.int32)
    act_nod = np.zeros(N, dtype=np.int32)

    while len(act_par) > 0:
        nod_com  = com[act_nod]                       # (K, 3)
        nod_mass = node_mass[act_nod]                 # (K,)
        nod_half = node_half[act_nod]                 # (K,)
        leaf_par = node_leaf_idx[act_nod]             # (K,)  -1 if internal
        is_leaf  = leaf_par >= 0                      # (K,)

        dr    = nod_com - p.pos[act_par]              # (K, 3)
        dist2 = np.einsum('ij,ij->i', dr, dr) + SOFTENING**2  # (K,)
        dist  = np.sqrt(dist2)                        # (K,)

        is_self    = is_leaf & (leaf_par == act_par)
        open_angle = (2.0 * nod_half) / dist          # s/d
        use_approx = ((open_angle < theta) | is_leaf) & ~is_self

        # 力を蓄積（COM近似 or 葉ノード）
        if use_approx.any():
            ap = act_par[use_approx]
            m_ = nod_mass[use_approx]
            dp = dr[use_approx]
            d2 = dist2[use_approx]
            d  = dist[use_approx]

            f = G * m_[:, None] * dp / (d2 * d)[:, None]  # (K', 3)

            # bincount による高速 scatter-add（np.add.at より速い）
            for dim in range(3):
                acc[:, dim] += np.bincount(ap, weights=f[:, dim], minlength=N)

        # 子ノードへ展開（再帰が必要なペアのみ）
        recurse = ~use_approx & ~is_self
        if not recurse.any():
            break

        rec_par    = act_par[recurse]
        rec_nod    = act_nod[recurse]
        child_nods = node_children[rec_nod]   # (K_rec, 8)
        valid      = child_nods >= 0          # (K_rec, 8)

        act_par = np.repeat(rec_par, 8)[valid.ravel()]
        act_nod = child_nods.ravel()[valid.ravel()].astype(np.int32)

    return acc
