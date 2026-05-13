"""
グラフ圧縮法（あなたの手法）— ここに実装を追加していく

設計思想
--------
粒子間相互作用をグラフ G = (V, E) として表現し、
グラフ圧縮（スパース化・クラスタリング・コアセット等）を用いて
計算量を削減しながら加速度を近似する。

実装ステップ（コメントで示す）
------------------------------
Step 1: 相互作用グラフの構築
Step 2: グラフ圧縮（スパース化・クラスタリング等）
Step 3: 圧縮グラフ上での力計算
Step 4: 直接法・Barnes-Hutとの精度/速度比較
"""

import numpy as np
from ..utils.core import Particles, G, SOFTENING


# ─────────────────────────────────────────────────────
# Step 1: 相互作用グラフの構築
# ─────────────────────────────────────────────────────

def build_interaction_graph(p: Particles, cutoff: float = None):
    """
    粒子間の相互作用グラフを構築する。

    Parameters
    ----------
    cutoff : float or None
        Noneなら全ペア（完全グラフ）。
        値を指定するとその距離以内のみエッジを張る。

    Returns
    -------
    edges : list of (i, j, weight)
        weight = G * m_i * m_j / r²  （相互作用の強さ）
    """
    edges = []
    for i in range(p.N):
        for j in range(i + 1, p.N):
            dr   = p.pos[j] - p.pos[i]
            dist = np.linalg.norm(dr) + SOFTENING
            if cutoff is None or dist < cutoff:
                weight = G * p.mass[i] * p.mass[j] / dist**2
                edges.append((i, j, weight))
    return edges


# ─────────────────────────────────────────────────────
# Step 2: グラフ圧縮（ここが研究のコア）
# ─────────────────────────────────────────────────────

def compress_graph_threshold(edges, keep_ratio: float = 0.5):
    """
    重み上位 keep_ratio のエッジのみ残すシンプルな閾値圧縮。
    （ベースラインとして実装。より高度な手法をここに追加する）

    拡張候補
    --------
    - スペクトルスパース化（Spielman-Srivastava）
    - ランダムウォークベースのクラスタリング
    - Louvain / Leiden コミュニティ検出
    - コアセット近似
    """
    if not edges:
        return edges
    weights = np.array([e[2] for e in edges])
    threshold = np.quantile(weights, 1 - keep_ratio)
    return [e for e in edges if e[2] >= threshold]


# ─────────────────────────────────────────────────────
# Step 3: 圧縮グラフ上での力計算
# ─────────────────────────────────────────────────────

def compute_acceleration_graph(
    p: Particles,
    keep_ratio: float = 0.5,
    cutoff: float = None,
) -> np.ndarray:
    """
    グラフ圧縮法による加速度計算。

    Parameters
    ----------
    keep_ratio : float
        保持するエッジの割合（1.0 = 直接法と同等）
    cutoff : float or None
        相互作用の距離カットオフ
    """
    edges = build_interaction_graph(p, cutoff=cutoff)
    compressed = compress_graph_threshold(edges, keep_ratio=keep_ratio)

    acc = np.zeros((p.N, 3))
    edge_set = set()

    for (i, j, _) in compressed:
        edge_set.add((i, j))
        dr    = p.pos[j] - p.pos[i]
        dist2 = np.dot(dr, dr) + SOFTENING**2
        dist  = np.sqrt(dist2)
        f_ij  = G * p.mass[j] * dr / (dist2 * dist)
        f_ji  = G * p.mass[i] * (-dr) / (dist2 * dist)
        acc[i] += f_ij
        acc[j] += f_ji

    return acc, len(compressed), len(edges)


def graph_compression_stats(p: Particles, keep_ratio: float = 0.5):
    """圧縮率・保持エッジ数などの統計を返す"""
    edges = build_interaction_graph(p)
    compressed = compress_graph_threshold(edges, keep_ratio)
    total = len(edges)
    kept  = len(compressed)
    return {
        "total_edges": total,
        "kept_edges": kept,
        "compression_ratio": kept / total if total > 0 else 1.0,
    }
