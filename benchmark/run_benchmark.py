"""
ベンチマーク — 直接法 / Barnes-Hut / グラフ圧縮法の比較
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Japanese font setup — use IPAexGothic bundled with japanize_matplotlib if available
_jp_font_candidates = [
    Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "japanize_matplotlib" / "fonts" / "ipaexg.ttf",
    Path(__file__).parent.parent / "ipaexg.ttf",
]
for _fp in _jp_font_candidates:
    if _fp.exists():
        fm.fontManager.addfont(str(_fp))
        plt.rcParams["font.family"] = fm.FontProperties(fname=str(_fp)).get_name()
        break

from nbody_graph.utils.core import make_random_particles, make_clustered_particles, G, SOFTENING
from nbody_graph.methods.direct          import compute_acceleration_direct
from nbody_graph.methods.barnes_hut      import compute_acceleration_barneshut
from nbody_graph.methods.graph_compression import compute_acceleration_graph


# ─────────────────────────────────────────────────────
# 1. 加速度誤差の計算
# ─────────────────────────────────────────────────────

def relative_error(acc_ref: np.ndarray, acc_approx: np.ndarray) -> float:
    """各粒子の相対誤差の RMS"""
    norms_ref = np.linalg.norm(acc_ref, axis=1)
    diff      = np.linalg.norm(acc_ref - acc_approx, axis=1)
    # ゼロ除算を避けるため微小値でクリップ
    rel = diff / np.maximum(norms_ref, 1e-30)
    return float(np.sqrt(np.mean(rel**2)))


# ─────────────────────────────────────────────────────
# 2. 単一サイズでの精度・速度比較
# ─────────────────────────────────────────────────────

def run_single_benchmark(N: int = 200, n_trials: int = 3):
    p = make_clustered_particles(N)

    results = {}

    # --- 直接法（基準）---
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        acc_direct = compute_acceleration_direct(p)
        times.append(time.perf_counter() - t0)
    results["direct"] = {
        "time": np.median(times),
        "error": 0.0,
        "label": "直接法 O(N²)",
    }

    # --- Barnes-Hut ---
    for theta in [0.3, 0.5, 0.7]:
        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            acc_bh = compute_acceleration_barneshut(p, theta=theta)
            times.append(time.perf_counter() - t0)
        results[f"bh_θ{theta}"] = {
            "time": np.median(times),
            "error": relative_error(acc_direct, acc_bh),
            "label": f"Barnes-Hut θ={theta}",
        }

    # --- スペクトルスパース化（複数 keep_ratio）---
    for kr in [0.3, 0.5, 0.7, 1.0]:
        times = []
        acc_gc = None
        for _ in range(n_trials):
            t0 = time.perf_counter()
            acc_gc, _, _ = compute_acceleration_graph(p, keep_ratio=kr)
            times.append(time.perf_counter() - t0)
        results[f"gc_spec_k{kr}"] = {
            "time": np.median(times),
            "error": relative_error(acc_direct, acc_gc),
            "label": f"スペクトル {int(kr*100)}%",
        }

    return results


# ─────────────────────────────────────────────────────
# 3. Nスケーリング比較
# ─────────────────────────────────────────────────────

def run_scaling_benchmark(N_list=None):
    if N_list is None:
        N_list = [100, 200, 500, 1000, 2000, 3000, 5000, 10000]

    times_direct  = []
    times_bh      = []
    times_gc_spec = []

    for N in N_list:
        p = make_clustered_particles(N)
        print(f"  N={N}...", end="", flush=True)

        t0 = time.perf_counter(); compute_acceleration_direct(p);          times_direct.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); compute_acceleration_barneshut(p);       times_bh.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); compute_acceleration_graph(p, 0.5);      times_gc_spec.append(time.perf_counter() - t0)

        print(" done")

    return N_list, times_direct, times_bh, times_gc_spec


# ─────────────────────────────────────────────────────
# 4. プロット
# ─────────────────────────────────────────────────────

def plot_results(single_results, scaling_data, out_path="benchmark_results.png", N_single=200):
    fig = plt.figure(figsize=(14, 15))
    fig.patch.set_facecolor("#f8f8f6")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)

    colors = {
        "direct":  "#444441",
        "bh":      "#534AB7",
        "gc_spec": "#6B1F0A",
    }

    # ── (A) 精度 vs 速度（散布図）──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#f8f8f6")
    ax1.set_title("精度 vs 計算時間（N=200）", fontsize=13, fontweight="500", pad=10)

    for key, res in single_results.items():
        if key == "direct":
            continue
        if key.startswith("bh"):
            c, mk = colors["bh"], "o"
        else:
            c, mk = colors["gc_spec"], "P"
        ax1.scatter(res["error"], res["time"] * 1000,
                    color=c, marker=mk, s=90, zorder=3)
        ax1.annotate(res["label"], (res["error"], res["time"] * 1000),
                     textcoords="offset points", xytext=(6, 2),
                     fontsize=8, color=c)

    ax1.axhline(single_results["direct"]["time"] * 1000,
                color=colors["direct"], linestyle="--", linewidth=1.2,
                label=f'直接法 ({single_results["direct"]["time"]*1000:.1f} ms)')
    ax1.set_xlabel("相対誤差 (RMS)", fontsize=11)
    ax1.set_ylabel("計算時間 [ms]", fontsize=11)
    ax1.set_xscale("log")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── (B) Nスケーリング ──
    N_list, td, tbh, tgc_spec = scaling_data
    ns = np.array(N_list)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#f8f8f6")
    ax2.set_title("計算時間 vs N", fontsize=12, fontweight="500")
    ax2.plot(ns, [t*1000 for t in td],       "o-", color=colors["direct"],  label="直接法 O(N2)",      linewidth=1.5)
    ax2.plot(ns, [t*1000 for t in tbh],      "s-", color=colors["bh"],      label="Barnes-Hut",        linewidth=1.5)
    ax2.plot(ns, [t*1000 for t in tgc_spec], "P-", color=colors["gc_spec"], label="スペクトル k=32",   linewidth=1.5)
    ax2.set_xlabel("N (粒子数)", fontsize=10)
    ax2.set_ylabel("計算時間 [ms]", fontsize=10)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── (C) スペクトル: 保持率 vs 誤差 ──
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#f8f8f6")
    ax3.set_title("スペクトルスパース化: 保持率 vs 誤差", fontsize=12, fontweight="500")
    krs_base = [0.3, 0.5, 0.7, 1.0]
    errs = [single_results[f"gc_spec_k{kr}"]["error"] for kr in krs_base if f"gc_spec_k{kr}" in single_results]
    krs_plot = [kr for kr in krs_base if f"gc_spec_k{kr}" in single_results]
    ax3.plot(krs_plot, errs, "P-", color=colors["gc_spec"], linewidth=1.5)
    ax3.set_xlabel("保持率 (keep_ratio)", fontsize=10)
    ax3.set_ylabel("相対誤差 (RMS)", fontsize=10)
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3)

    # ── (D) 粒子分布（XY / XZ 射影）──
    p_vis = make_clustered_particles(N_single)
    vmin, vmax = p_vis.mass.min(), p_vis.mass.max()
    cmap = "plasma"
    s = (p_vis.mass - vmin) / (vmax - vmin) * 40 + 10   # 10〜50px

    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor("#f8f8f6")
    ax4.set_title(f"粒子分布 XY — クラスター分布 (N={N_single})", fontsize=12, fontweight="500")
    sc4 = ax4.scatter(p_vis.pos[:, 0], p_vis.pos[:, 1],
                      c=p_vis.mass, cmap=cmap, vmin=vmin, vmax=vmax,
                      s=s, alpha=0.75, linewidths=0)
    fig.colorbar(sc4, ax=ax4, label="質量 [Msun]", shrink=0.85)
    ax4.set_xlabel("X [AU]", fontsize=10)
    ax4.set_ylabel("Y [AU]", fontsize=10)
    ax4.set_aspect("equal", adjustable="datalim")
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor("#f8f8f6")
    ax5.set_title(f"粒子分布 XZ — クラスター分布 (N={N_single})", fontsize=12, fontweight="500")
    sc5 = ax5.scatter(p_vis.pos[:, 0], p_vis.pos[:, 2],
                      c=p_vis.mass, cmap=cmap, vmin=vmin, vmax=vmax,
                      s=s, alpha=0.75, linewidths=0)
    fig.colorbar(sc5, ax=ax5, label="質量 [Msun]", shrink=0.85)
    ax5.set_xlabel("X [AU]", fontsize=10)
    ax5.set_ylabel("Z [AU]", fontsize=10)
    ax5.set_aspect("equal", adjustable="datalim")
    ax5.grid(True, alpha=0.3)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n図を保存: {out_path}")
    return fig


# ─────────────────────────────────────────────────────
# 5. メイン実行
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("N体問題 手法比較ベンチマーク")
    print("=" * 50)

    print("\n[1/2] 単一サイズ比較 (N=200)...")
    single = run_single_benchmark(N=200, n_trials=3)

    print("\n結果サマリ:")
    print(f"{'手法':<30} {'時間[ms]':>10} {'相対誤差':>12}")
    print("-" * 54)
    for key, res in single.items():
        err_str = f"{res['error']:.4e}" if res["error"] > 0 else "基準"
        print(f"{res['label']:<30} {res['time']*1000:>10.2f} {err_str:>12}")

    print("\n[2/2] スケーリング比較...")
    scaling = run_scaling_benchmark()

    out = str(Path(__file__).parent.parent / "benchmark_results.png")
    plot_results(single, scaling, out_path=out)
    print("\n完了!")
