"""
高速多重極展開法（FMM）— 簡略版（2D多重極、教育・比較用途）

完全な3D FMMは非常に複雑なため、ここでは:
  - 粒子群をセルに分割
  - 各セルの多重極モーメント（0次=質量, 1次=質量×位置）を計算
  - 遠方セルは多重極展開で近似、近傍セルは直接計算

という「Particle-Mesh的な簡略FMM」を実装する。
完全なFMM（p次展開、M2L変換）への拡張ポイントにコメントを付記。
"""

import numpy as np
from ..utils.core import Particles, G, SOFTENING


def _assign_cells(pos: np.ndarray, n_cells_per_dim: int):
    """粒子を3D格子セルに割り当てる"""
    pmin = pos.min(axis=0) - 1e-8
    pmax = pos.max(axis=0) + 1e-8
    cell_size = (pmax - pmin) / n_cells_per_dim

    idx = ((pos - pmin) / cell_size).astype(int)
    idx = np.clip(idx, 0, n_cells_per_dim - 1)

    # セルIDを1次元に変換
    C = n_cells_per_dim
    cell_id = idx[:, 0] * C * C + idx[:, 1] * C + idx[:, 2]
    return cell_id, cell_size, pmin


def _multipole_moments(pos, mass):
    """
    0次 (total mass) と 1次 (mass-weighted center) の多重極モーメントを計算。
    拡張ポイント: ここに2次・高次モーメントを追加するとFMMの精度が上がる。
    """
    M = mass.sum()
    if M == 0:
        return 0.0, np.zeros(3)
    com = (pos * mass[:, None]).sum(axis=0) / M
    return M, com


def compute_acceleration_fmm(
    p: Particles,
    n_cells_per_dim: int = 4,
    near_threshold: int = 1,
) -> np.ndarray:
    """
    簡略FMMによる加速度計算。

    Parameters
    ----------
    n_cells_per_dim : int
        各次元のセル分割数（合計 C³ セル）
    near_threshold : int
        この距離（セル数）以内は直接法を使う近傍領域

    計算フロー
    ----------
    1. 粒子をセルに割り当て
    2. 各セルの多重極モーメント（質量・質量中心）を計算  [Upward pass]
    3. 各粒子 i に対して:
        - 遠方セル: 多重極展開で加速度を近似             [M2L相当]
        - 近傍セル: 直接法                               [Near-field]
    """
    C = n_cells_per_dim
    cell_id, cell_size, pmin = _assign_cells(p.pos, C)

    # --- Upward pass: セルの多重極モーメントを計算 ---
    unique_cells = np.unique(cell_id)
    cell_data = {}  # cell_id -> (total_mass, com, [particle_indices])
    for cid in unique_cells:
        members = np.where(cell_id == cid)[0]
        M, com  = _multipole_moments(p.pos[members], p.mass[members])
        cell_data[cid] = (M, com, members)

    # --- 各粒子への加速度計算 ---
    acc = np.zeros((p.N, 3))

    # セルの3D座標
    def cell_coord(cid):
        return np.array([cid // (C * C), (cid // C) % C, cid % C])

    for i in range(p.N):
        ci = cell_coord(cell_id[i])

        for cid, (M, com, members) in cell_data.items():
            if M == 0:
                continue
            cj = cell_coord(cid)
            # チェビシェフ距離でセル間距離を測る
            cell_dist = np.max(np.abs(ci - cj))

            if cell_dist <= near_threshold:
                # 近傍セル → 直接法（自己を除く）
                for j in members:
                    if j == i:
                        continue
                    dr = p.pos[j] - p.pos[i]
                    dist2 = np.dot(dr, dr) + SOFTENING**2
                    acc[i] += G * p.mass[j] * dr / (dist2 * dist2**0.5)
            else:
                # 遠方セル → 多重極近似（0次 + 1次モーメント）
                dr    = com - p.pos[i]
                dist2 = np.dot(dr, dr) + SOFTENING**2
                dist  = np.sqrt(dist2)
                # 拡張ポイント: 高次モーメント補正をここに追加
                acc[i] += G * M * dr / (dist2 * dist)

    return acc
