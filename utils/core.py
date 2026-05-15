"""
共通データ構造・ユーティリティ
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


G = 6.674e-11   # 重力定数 [SI]
SOFTENING = 1e-3  # ソフトニングパラメータ（ゼロ除算防止）


@dataclass
class Particles:
    """N体粒子の状態をまとめて保持するクラス"""
    pos: np.ndarray    # shape (N, 3) [m]
    vel: np.ndarray    # shape (N, 3) [m/s]
    mass: np.ndarray   # shape (N,)  [kg]
    acc: np.ndarray = field(init=False)

    def __post_init__(self):
        self.acc = np.zeros_like(self.pos)

    @property
    def N(self):
        return len(self.mass)

    def copy(self) -> "Particles":
        return Particles(
            pos=self.pos.copy(),
            vel=self.vel.copy(),
            mass=self.mass.copy(),
        )


def make_random_particles(N: int, seed: int = 42) -> Particles:
    """再現性のあるランダム粒子群を生成（単位: AU, M☉ スケール）"""
    rng = np.random.default_rng(seed)
    pos  = rng.standard_normal((N, 3))
    vel  = rng.standard_normal((N, 3)) * 0.1
    mass = rng.uniform(0.5, 2.0, N)
    return Particles(pos=pos, vel=vel, mass=mass)


def make_clustered_particles(N: int, n_clusters: int = 5, seed: int = 42) -> Particles:
    """複数クラスターを持つ天体分布を生成（AU, M☉ スケール）

    各クラスターは重い中心天体（恒星相当）と軽い周辺天体（惑星/小天体相当）で構成される。
    クラスター中心は互いに ~5 AU 離れて配置される。
    """
    rng = np.random.default_rng(seed)

    # クラスター中心を広い空間に配置
    cluster_centers = rng.standard_normal((n_clusters, 3)) * 5.0

    # 各クラスターへ粒子を均等割り当て（余りは先頭クラスターに追加）
    base = N // n_clusters
    counts = np.full(n_clusters, base, dtype=int)
    counts[: N - base * n_clusters] += 1

    pos_list, vel_list, mass_list = [], [], []

    for k in range(n_clusters):
        n_k = int(counts[k])
        center = cluster_centers[k]

        # クラスター内の位置：中心からガウス分布（広がりはクラスターごとにランダム）
        spread = rng.uniform(0.3, 0.8)
        local_pos = rng.standard_normal((n_k, 3)) * spread

        # 質量：index=0 が重い中心天体、残りは軽い周辺天体
        masses = rng.uniform(0.1, 0.5, n_k)
        masses[0] = rng.uniform(1.0, 3.0)

        # 速度：クラスター全体のドリフト + 小さな熱的ランダム成分
        drift = rng.standard_normal(3) * 0.05
        local_vel = rng.standard_normal((n_k, 3)) * 0.02 + drift

        pos_list.append(center + local_pos)
        vel_list.append(local_vel)
        mass_list.append(masses)

    return Particles(
        pos=np.vstack(pos_list),
        vel=np.vstack(vel_list),
        mass=np.concatenate(mass_list),
    )


def kinetic_energy(p: Particles) -> float:
    return 0.5 * np.sum(p.mass[:, None] * p.vel**2)


def potential_energy(p: Particles) -> float:
    """O(N²) で厳密に計算（検証用）"""
    E = 0.0
    for i in range(p.N):
        for j in range(i + 1, p.N):
            r = np.linalg.norm(p.pos[i] - p.pos[j])
            E -= G * p.mass[i] * p.mass[j] / (r + SOFTENING)
    return E


def leapfrog_step(p: Particles, acc_fn, dt: float):
    """Leapfrog（Störmer-Verlet）積分法で1ステップ進める"""
    p.vel += 0.5 * dt * p.acc
    p.pos += dt * p.vel
    p.acc = acc_fn(p)
    p.vel += 0.5 * dt * p.acc
