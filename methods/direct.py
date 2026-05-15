"""
直接法 O(N²) — 全ペアの相互作用を計算する基準実装
"""

import numpy as np
from ..utils.core import Particles, G, SOFTENING

_CHUNK = 500  # 一度に処理する行数 (N×CHUNK×3×8 bytes がピークメモリ)


def compute_acceleration_direct(p: Particles) -> np.ndarray:
    """
    全粒子ペアに対してニュートン重力加速度を計算。
    精度が最も高い基準実装（reference solution）。

    N > _CHUNK の場合はチャンク処理してピークメモリを抑制する。

    Returns
    -------
    acc : ndarray, shape (N, 3)
    """
    N = p.N

    if N <= _CHUNK:
        dr = p.pos[np.newaxis, :, :] - p.pos[:, np.newaxis, :]   # (N, N, 3)
        dist2 = np.sum(dr**2, axis=-1) + SOFTENING**2
        dist3 = dist2**1.5
        np.fill_diagonal(dist3, np.inf)
        return G * np.sum(
            p.mass[np.newaxis, :, np.newaxis] * dr / dist3[:, :, np.newaxis],
            axis=1,
        )

    # チャンク処理: ピーク使用量 = CHUNK×N×3×8 bytes ≈ 120 MB (N=10000)
    acc = np.zeros((N, 3))
    for start in range(0, N, _CHUNK):
        end = min(start + _CHUNK, N)
        chunk = end - start
        dr_c = p.pos[np.newaxis, :, :] - p.pos[start:end, np.newaxis, :]  # (chunk, N, 3)
        dist2_c = np.sum(dr_c**2, axis=-1) + SOFTENING**2                  # (chunk, N)
        dist3_c = dist2_c**1.5
        # 自己相互作用（対角成分）を除外
        k = np.arange(chunk)
        dist3_c[k, start + k] = np.inf
        acc[start:end] = G * np.sum(
            p.mass[np.newaxis, :, np.newaxis] * dr_c / dist3_c[:, :, np.newaxis],
            axis=1,
        )
    return acc
