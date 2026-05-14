"""
グラフ圧縮法による N 体問題の近似加速度計算

圧縮アルゴリズム一覧
--------------------
1. compress_graph_threshold   — 閾値圧縮（決定論的、ベースライン）
2. compress_graph_importance  — 重み比例サンプリング（確率的、不偏推定）
3. compress_graph_spectral    — スペクトルスパース化（有効抵抗ベース）

設計メモ: threshold は「エッジを落として正確な力を計算」する偏った推定。
importance と spectral は「採択エッジの重みを 1/p でスケーリング」することで
E[w̃_e] = w_e を保ち、不偏推定量を得る。
"""

import numpy as np
from scipy.linalg import eigh
from ..utils.core import Particles, G, SOFTENING


# ─────────────────────────────────────────────────────────────
# Step 1: 相互作用グラフの構築
# ─────────────────────────────────────────────────────────────

def build_interaction_graph(p: Particles, cutoff: float = None):
    """
    粒子間の相互作用グラフを構築する。

    Returns
    -------
    ii, jj : ndarray of int, shape (E,)
    weights : ndarray of float, shape (E,)  — G * m_i * m_j / r²
    """
    ii, jj = np.triu_indices(p.N, k=1)

    dr    = p.pos[jj] - p.pos[ii]
    dist2 = np.sum(dr**2, axis=1) + SOFTENING**2   # 力計算と同じソフトニング
    dist  = np.sqrt(dist2)

    if cutoff is not None:
        mask = dist < cutoff
        ii, jj, dist = ii[mask], jj[mask], dist[mask]

    weights = G * p.mass[ii] * p.mass[jj] / dist**2
    return ii, jj, weights


# ─────────────────────────────────────────────────────────────
# Step 2a: 閾値圧縮（ベースライン）
# ─────────────────────────────────────────────────────────────

def compress_graph_threshold(ii, jj, weights, keep_ratio: float = 0.5):
    """
    重み上位 keep_ratio のエッジのみ残す決定論的圧縮。

    推定の性質: 偏り有り（弱い相互作用を系統的に脱落させる）。
    速度: O(E log E)（ソート相当）。
    """
    if len(weights) == 0:
        return ii, jj, weights
    threshold = np.quantile(weights, 1 - keep_ratio)
    mask = weights >= threshold
    return ii[mask], jj[mask], weights[mask]


# ─────────────────────────────────────────────────────────────
# Step 2b: 重み比例サンプリング（不偏推定）
# ─────────────────────────────────────────────────────────────

def compress_graph_importance(
    ii, jj, weights, keep_ratio: float = 0.5, seed=None
):
    """
    重み比例確率サンプリング＋リスケーリング（不偏推定量）

    各エッジ e をサンプリング確率 p_e = min(1, n_keep · w_e / Σw) で採択し、
    採択された重みを w̃_e = w_e / p_e にスケーリングする。
    このとき E[w̃_e] = w_e が保証され、どの粒子への合力も不偏に推定できる。

    推定の性質: 不偏（期待値 = 真値）。
    速度: O(E)。
    """
    if len(weights) == 0 or keep_ratio >= 1.0:
        return ii, jj, weights.copy()
    rng = np.random.default_rng(seed)

    n_keep = max(1, int(len(weights) * keep_ratio))
    p = np.minimum(1.0, n_keep * weights / weights.sum())

    mask = rng.uniform(size=len(weights)) < p
    if not mask.any():
        mask[np.argmax(weights)] = True

    return ii[mask], jj[mask], weights[mask] / p[mask]


# ─────────────────────────────────────────────────────────────
# Step 2c: スペクトルスパース化（Spielman-Srivastava 型）
# ─────────────────────────────────────────────────────────────

def compress_graph_spectral(
    ii, jj, weights, keep_ratio: float = 0.5, seed=None
):
    """
    有効抵抗に基づくスペクトルスパース化。

    重み付きラプラシアン L = D − A の固有分解から有効抵抗
        R_e = (e_i − e_j)ᵀ L⁺ (e_i − e_j)
    を計算し、レバレッジスコア l_e = w_e · R_e に比例する確率でサンプリング。

    性質:
    - Σ l_e = n−1（連結グラフ）— 接続に不可欠なエッジほど高スコア
    - keep_ratio = (n−1)/E のとき、スパニングツリー構造が必ず残る
    - 採択後の重みスケーリングで不偏推定を維持

    計算コスト: O(N³)（密行列の固有分解）。N ≲ 1000 なら実用範囲。
    """
    if len(weights) == 0 or keep_ratio >= 1.0:
        return ii, jj, weights.copy()
    rng = np.random.default_rng(seed)
    N = int(max(ii.max(), jj.max())) + 1
    n_keep = max(1, int(len(weights) * keep_ratio))

    # 重み付きラプラシアン L = D − A（密行列）
    deg = np.zeros(N)
    np.add.at(deg, ii, weights)
    np.add.at(deg, jj, weights)
    L = np.diag(deg)
    L[ii, jj] -= weights
    L[jj, ii] -= weights

    # 固有分解 L = U Λ Uᵀ（対称行列なので eigh が使える）
    eigvals, U = eigh(L)
    tol = eigvals[-1] * 1e-10
    nonzero = eigvals > tol

    # V[i, :] ≡ (Λ^{1/2+} Uᵀ)[:, i]  →  R_ij = ‖V[i] − V[j]‖²
    sqrt_inv = np.where(nonzero, 1.0 / np.sqrt(np.maximum(eigvals, tol)), 0.0)
    V = U * sqrt_inv[None, :]            # (N, N)

    diff_V = V[ii] - V[jj]              # (E, N)
    R = np.sum(diff_V**2, axis=1)        # (E,)  有効抵抗

    leverage = weights * R               # l_e = w_e · R_e
    l_sum = leverage.sum()
    if l_sum < 1e-30:
        return compress_graph_threshold(ii, jj, weights, keep_ratio)

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
    cutoff: float = None,
    method: str = "threshold",
) -> tuple:
    """
    グラフ圧縮法による加速度計算。

    Parameters
    ----------
    keep_ratio : float
        保持するエッジの割合（1.0 = 直接法と同等）
    cutoff : float or None
        相互作用の距離カットオフ
    method : {"threshold", "importance", "spectral"}
        圧縮アルゴリズムの選択

    Returns
    -------
    acc          : ndarray (N, 3)
    n_kept_edges : int
    n_total_edges: int
    """
    ii_all, jj_all, w_all = build_interaction_graph(p, cutoff=cutoff)

    if method == "importance":
        ii, jj, w = compress_graph_importance(ii_all, jj_all, w_all, keep_ratio)
        rescaled = True
    elif method == "spectral":
        ii, jj, w = compress_graph_spectral(ii_all, jj_all, w_all, keep_ratio)
        rescaled = True
    else:  # threshold（デフォルト・後方互換）
        ii, jj, _ = compress_graph_threshold(ii_all, jj_all, w_all, keep_ratio)
        rescaled = False

    acc = np.zeros((p.N, 3))
    if len(ii) > 0:
        dr    = p.pos[jj] - p.pos[ii]
        dist2 = np.sum(dr**2, axis=1) + SOFTENING**2
        dist  = np.sqrt(dist2)

        if rescaled:
            # w̃ = G·m_i·m_j/r² · (1/p_e)
            # 粒子 i の加速度寄与: a_ij = w̃ / (m_i · r) · dr
            # 不偏性: E[a_ij] = (p·w/p) / (m_i·r) · dr = G·m_j/r³ · dr ✓
            f_ij = dr * (w / (p.mass[ii] * dist))[:, None]
            f_ji = dr * (w / (p.mass[jj] * dist))[:, None]
        else:
            # 閾値法: 採択エッジに正確な力を使う（偏りあり）
            scale = G / (dist2 * dist)
            f_ij = dr * (p.mass[jj] * scale)[:, None]
            f_ji = dr * (p.mass[ii] * scale)[:, None]

        np.add.at(acc, ii,  f_ij)
        np.add.at(acc, jj, -f_ji)

    return acc, len(ii), len(ii_all)


def graph_compression_stats(p: Particles, keep_ratio: float = 0.5):
    """圧縮率・保持エッジ数などの統計を返す"""
    ii_all, jj_all, w_all = build_interaction_graph(p)
    ii, _, _ = compress_graph_threshold(ii_all, jj_all, w_all, keep_ratio)
    total = len(ii_all)
    kept  = len(ii)
    return {
        "total_edges": total,
        "kept_edges": kept,
        "compression_ratio": kept / total if total > 0 else 1.0,
    }
