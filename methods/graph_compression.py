"""
グラフ圧縮法による N 体問題の近似加速度計算

グラフ構築:
  scipy.spatial.KDTree で k 近傍のみエッジ化 → O(N log N + N·k)

圧縮アルゴリズム:
  次数ベースレバレッジスコア近似 — R_e ≈ 1/deg[i] + 1/deg[j]
  k-NN グラフで有効な局所近似。O(E) で全 N スケールに対応。
"""

import numpy as np
from scipy.spatial import KDTree
from ..utils.core import Particles, G, SOFTENING


# ─────────────────────────────────────────────────────────────
# Step 1: スパース相互作用グラフの構築（kd-tree k近傍）
# ─────────────────────────────────────────────────────────────

def build_sparse_interaction_graph(p: Particles, k_neighbors: int = 32):
    """
    kd-tree を使って k 近傍グラフを構築する。O(N log N + N·k)

    全粒子ペア O(N²/2) ではなく各粒子の上位 k 近傍のみをエッジ化する。
    双方向の近傍関係を考慮し、undirected edge (i < j) として重複なく返す。

    Parameters
    ----------
    p           : Particles
    k_neighbors : 各粒子あたりの近傍数（N-1 でクリップ）

    Returns
    -------
    ii, jj   : ndarray of int, shape (E,)   — E ≦ N·k
    weights  : ndarray of float, shape (E,) — G * m_i * m_j / r²
    """
    k = min(k_neighbors, p.N - 1)
    tree = KDTree(p.pos)
    dists, idxs = tree.query(p.pos, k=k + 1)   # index 0 は自身（距離 0）

    src = np.repeat(np.arange(p.N), k)
    dst = idxs[:, 1:].ravel()                   # 自身を除いた k 近傍

    # undirected pair (i < j) に統一 → 重複除去
    ii_raw = np.minimum(src, dst)
    jj_raw = np.maximum(src, dst)
    pairs = np.unique(np.stack([ii_raw, jj_raw], axis=1), axis=0)
    ii, jj = pairs[:, 0], pairs[:, 1]

    dr    = p.pos[jj] - p.pos[ii]
    dist2 = np.sum(dr**2, axis=1) + SOFTENING**2
    weights = G * p.mass[ii] * p.mass[jj] / dist2
    return ii, jj, weights


# ─────────────────────────────────────────────────────────────
# Step 2: スペクトルスパース化（Spielman-Srivastava 型）
# ─────────────────────────────────────────────────────────────

def compress_graph_spectral(ii, jj, weights, keep_ratio: float = 0.5, seed=None):
    """
    次数ベースレバレッジスコアによるグラフスパース化。

    R_e ≈ 1/deg[i] + 1/deg[j] を有効抵抗の近似として使用。
    k-NN グラフで孤立粒子（低次数）を優先保持する物理的に妥当な近似。
    O(E) で全 N スケールに対応。
    """
    if len(weights) == 0 or keep_ratio >= 1.0:
        return ii, jj, weights.copy()
    rng = np.random.default_rng(seed)
    N = int(max(ii.max(), jj.max())) + 1
    n_keep = max(1, int(len(weights) * keep_ratio))

    deg = np.zeros(N)
    np.add.at(deg, ii, weights)
    np.add.at(deg, jj, weights)

    R = 1.0 / np.maximum(deg[ii], 1e-30) + 1.0 / np.maximum(deg[jj], 1e-30)
    leverage = weights * R
    l_sum = leverage.sum()
    if l_sum < 1e-30:
        idx = rng.choice(len(weights), size=n_keep, replace=False)
        mask = np.zeros(len(weights), dtype=bool)
        mask[idx] = True
        return ii[mask], jj[mask], weights[mask]

    p = np.minimum(1.0, n_keep * leverage / l_sum)
    mask = rng.uniform(size=len(weights)) < p
    if not mask.any():
        mask[np.argmax(leverage)] = True

    return ii[mask], jj[mask], weights[mask] / p[mask]


# ─────────────────────────────────────────────────────────────
# Step 3: 圧縮グラフ上での力計算
# ─────────────────────────────────────────────────────────────

def compute_acceleration_graph(
    p: Particles,
    keep_ratio: float = 0.5,
    k_neighbors: int = 32,
) -> tuple:
    """
    スペクトルスパースグラフによる加速度計算。

    Parameters
    ----------
    keep_ratio  : スペクトル圧縮後に保持するエッジの割合
    k_neighbors : kd-tree で構築する近傍数

    Returns
    -------
    acc          : ndarray (N, 3)
    n_kept_edges : int
    n_total_edges: int  — スパースグラフのエッジ数
    """
    ii_all, jj_all, w_all = build_sparse_interaction_graph(p, k_neighbors)
    ii, jj, w = compress_graph_spectral(ii_all, jj_all, w_all, keep_ratio)

    acc = np.zeros((p.N, 3))
    if len(ii) > 0:
        dr    = p.pos[jj] - p.pos[ii]
        dist2 = np.sum(dr**2, axis=1) + SOFTENING**2
        dist  = np.sqrt(dist2)

        # w̃ = G·m_i·m_j/r² · (1/p_e)  — 不偏推定
        f_ij = dr * (w / (p.mass[ii] * dist))[:, None]
        f_ji = dr * (w / (p.mass[jj] * dist))[:, None]

        np.add.at(acc, ii,  f_ij)
        np.add.at(acc, jj, -f_ji)

    return acc, len(ii), len(ii_all)


def graph_compression_stats(p: Particles, keep_ratio: float = 0.5, k_neighbors: int = 32):
    """スパースグラフの圧縮率・保持エッジ数などの統計を返す"""
    ii_all, jj_all, w_all = build_sparse_interaction_graph(p, k_neighbors)
    ii, jj, _ = compress_graph_spectral(ii_all, jj_all, w_all, keep_ratio)
    total = len(ii_all)
    kept  = len(ii)
    return {
        "total_edges": total,
        "kept_edges":  kept,
        "compression_ratio": kept / total if total > 0 else 1.0,
    }
