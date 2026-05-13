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
