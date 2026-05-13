"""
Barnes-Hut法 O(N log N) — 八分木（Octree）による遠方粒子の質量中心近似
"""

import numpy as np
from ..utils.core import Particles, G, SOFTENING


class OctreeNode:
    """八分木の1ノード。葉ノードは1粒子、内部ノードは質量中心を保持。"""

    __slots__ = (
        "center", "half", "mass", "com", "idx",
        "children", "is_leaf",
    )

    def __init__(self, center: np.ndarray, half: float):
        self.center   = center      # このノードのAABBの中心
        self.half     = half        # 半辺長
        self.mass     = 0.0
        self.com      = np.zeros(3) # 質量中心
        self.idx      = -1          # 葉ノードの粒子インデックス
        self.children = [None] * 8  # 8つの子ノード
        self.is_leaf  = True

    def _child_idx(self, pos: np.ndarray) -> int:
        """どの八分体に属するかを3ビットで決定"""
        bit = (pos > self.center).astype(int)
        return bit[0] * 4 + bit[1] * 2 + bit[2]

    def _child_center(self, octant: int) -> np.ndarray:
        q = self.half / 2
        offsets = np.array([
            ((octant >> 2) & 1) * 2 - 1,
            ((octant >> 1) & 1) * 2 - 1,
            ((octant >> 0) & 1) * 2 - 1,
        ], dtype=float)
        return self.center + offsets * q

    def insert(self, pos: np.ndarray, mass: float, idx: int):
        """粒子を木に挿入し、質量・質量中心を更新"""
        self.com = (self.com * self.mass + pos * mass) / (self.mass + mass)
        self.mass += mass

        if self.is_leaf:
            if self.idx == -1:
                # 空の葉 → 直接格納
                self.idx = idx
                return
            # 既存粒子と衝突 → 分割して内部ノード化
            self.is_leaf = False
            old_idx = self.idx
            self.idx = -1
            # 既存粒子を子ノードへ再挿入（後で再帰するので再計算は不要）
            self._insert_to_child(pos - (pos - self.center),  # dummy
                                   old_idx, mass=0.0)           # massは既反映
            # ここでは old_pos が必要なので、別途管理が必要
            # → シンプル実装のため _insert_to_child はスキップし、
            #    build() 時に順番に insert する設計を採用
            # (実用上は pos リストを保持する実装が多い)
            return

        oi = self._child_idx(pos)
        if self.children[oi] is None:
            self.children[oi] = OctreeNode(
                self._child_center(oi), self.half / 2
            )
        self.children[oi].insert(pos, mass, idx)

    def calc_acc(
        self,
        pos: np.ndarray,
        theta: float,
        particle_pos: np.ndarray,
        particle_mass: np.ndarray,
    ) -> np.ndarray:
        """
        Barnes-Hut 基準で加速度を計算。
        theta: 開き角（小さいほど精度高・速度低）
        """
        if self.mass == 0.0:
            return np.zeros(3)

        dr = self.com - pos
        dist2 = np.dot(dr, dr) + SOFTENING**2
        dist = np.sqrt(dist2)

        if self.is_leaf:
            if self.idx >= 0 and np.allclose(particle_pos[self.idx], pos):
                return np.zeros(3)  # 自己相互作用を除外
            return G * self.mass * dr / (dist2 * dist)

        # 開き角判定: s/d < theta なら質量中心近似を使う
        s = 2 * self.half
        if s / dist < theta:
            return G * self.mass * dr / (dist2 * dist)

        # 精度不足 → 子ノードへ再帰
        acc = np.zeros(3)
        for child in self.children:
            if child is not None:
                acc += child.calc_acc(pos, theta, particle_pos, particle_mass)
        return acc


def _build_tree(p: Particles) -> OctreeNode:
    """粒子群から八分木を構築"""
    # ルートノードのサイズを粒子の最大範囲に合わせる
    center = (p.pos.max(axis=0) + p.pos.min(axis=0)) / 2
    half   = (p.pos.max(axis=0) - p.pos.min(axis=0)).max() / 2 * 1.01 + 1e-8

    root = OctreeNode(center, half)

    # 簡略実装: 全粒子を順番に挿入
    # (既存粒子の位置保持が必要なため、シンプルに全粒子ループで insert)
    # ここでは内部ノード化時の再挿入問題を回避するため、
    # 葉の分割を実装したフルバージョンを使う
    root2 = _FullOctree(center, half, p.pos, p.mass)
    return root2


class _FullOctree:
    """シンプルかつ正確な八分木実装（粒子配列を直接保持）"""

    def __init__(self, center, half, pos, mass):
        self.center = center
        self.half   = half
        self.pos    = pos
        self.mass   = mass
        self.total_mass = mass.sum()
        self.com = (pos * mass[:, None]).sum(axis=0) / self.total_mass
        self.children  = [None] * 8
        self.leaf_idx  = None

        indices = np.arange(len(mass))
        self._build(indices)

    def _octant(self, p):
        bit = (p > self.center).astype(int)
        return bit[0] * 4 + bit[1] * 2 + bit[2]

    def _child_center(self, oi):
        q = self.half / 2
        offsets = np.array([
            ((oi >> 2) & 1) * 2 - 1,
            ((oi >> 1) & 1) * 2 - 1,
            ((oi >> 0) & 1) * 2 - 1,
        ], dtype=float)
        return self.center + offsets * q

    def _build(self, indices):
        if len(indices) == 1:
            self.leaf_idx = indices[0]
            return
        # 各粒子をどの八分体に割り当てるか
        octants = np.array([self._octant(self.pos[i]) for i in indices])
        for oi in range(8):
            sub = indices[octants == oi]
            if len(sub) == 0:
                continue
            child = _FullOctree.__new__(_FullOctree)
            child.center = self._child_center(oi)
            child.half   = self.half / 2
            child.pos    = self.pos
            child.mass   = self.mass
            child.children = [None] * 8
            child.leaf_idx = None
            sub_mass = self.mass[sub]
            child.total_mass = sub_mass.sum()
            child.com = (self.pos[sub] * sub_mass[:, None]).sum(axis=0) / child.total_mass
            child._build(sub)
            self.children[oi] = child

    def calc_acc(self, pos, theta, src_idx=None):
        if self.total_mass == 0.0:
            return np.zeros(3)
        dr = self.com - pos
        dist2 = np.dot(dr, dr) + SOFTENING**2
        dist  = np.sqrt(dist2)

        if self.leaf_idx is not None:
            if self.leaf_idx == src_idx:
                return np.zeros(3)
            return G * self.total_mass * dr / (dist2 * dist)

        s = 2 * self.half
        if s / dist < theta:
            return G * self.total_mass * dr / (dist2 * dist)

        acc = np.zeros(3)
        for child in self.children:
            if child is not None:
                acc += child.calc_acc(pos, theta, src_idx)
        return acc


def compute_acceleration_barneshut(p: Particles, theta: float = 0.5) -> np.ndarray:
    """
    Barnes-Hut法で全粒子の加速度を計算。

    Parameters
    ----------
    theta : float
        開き角パラメータ (0=直接法と同等, 1.0=粗い近似)
    """
    tree = _build_tree(p)
    acc  = np.zeros((p.N, 3))
    for i in range(p.N):
        acc[i] = tree.calc_acc(p.pos[i], theta, src_idx=i)
    return acc
