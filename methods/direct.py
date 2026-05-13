"""
直接法 O(N²) — 全ペアの相互作用を計算する基準実装
"""

import numpy as np
from ..utils.core import Particles, G, SOFTENING


def compute_acceleration_direct(p: Particles) -> np.ndarray:
    """
    全粒子ペアに対してニュートン重力加速度を計算。
    精度が最も高い基準実装（reference solution）。

    Returns
    -------
    acc : ndarray, shape (N, 3)
    """
    N = p.N
    acc = np.zeros((N, 3))

    # ベクトル化: 差分行列を一括計算
    # dr[i, j] = pos[j] - pos[i]
    dr = p.pos[np.newaxis, :, :] - p.pos[:, np.newaxis, :]   # (N, N, 3)
    dist2 = np.sum(dr**2, axis=-1) + SOFTENING**2              # (N, N)
    dist3 = dist2**1.5                                          # (N, N)

    # 対角成分（自己相互作用）を除く
    np.fill_diagonal(dist3, np.inf)

    # a_i = G * sum_j m_j * (r_j - r_i) / |r_j - r_i|^3
    acc = G * np.sum(
        p.mass[np.newaxis, :, np.newaxis] * dr / dist3[:, :, np.newaxis],
        axis=1
    )
    return acc
