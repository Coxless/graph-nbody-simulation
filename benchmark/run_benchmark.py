"""
ベンチマーク — 直接法 / Barnes-Hut / FMM / グラフ圧縮法の比較
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

from nbody_graph.utils.core import make_random_particles, G, SOFTENING
from nbody_graph.methods.direct          import compute_acceleration_direct
from nbody_graph.methods.barnes_hut      import compute_acceleration_barneshut
from nbody_graph.methods.fmm             import compute_acceleration_fmm
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
    p = make_random_particles(N)

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

    # --- FMM ---
    for nc in [2, 3, 4]:
        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            acc_fmm = compute_acceleration_fmm(p, n_cells_per_dim=nc)
            times.append(time.perf_counter() - t0)
        results[f"fmm_c{nc}"] = {
            "time": np.median(times),
            "error": relative_error(acc_direct, acc_fmm),
            "label": f"FMM (cells={nc}³)",
        }

    # --- グラフ圧縮法（3手法 × 複数 keep_ratio）---
    for method, short, label_prefix in [
        ("threshold",  "gc",      "閾値圧縮"),
        ("importance", "gc_imp",  "重み比例サンプリング"),
        ("spectral",   "gc_spec", "スペクトルスパース化"),
    ]:
        for kr in [0.3, 0.5, 0.7, 1.0]:
            times = []
            acc_gc = None
            for _ in range(n_trials):
                t0 = time.perf_counter()
                acc_gc, _, _ = compute_acceleration_graph(p, keep_ratio=kr, method=method)
                times.append(time.perf_counter() - t0)
            results[f"{short}_k{kr}"] = {
                "time": np.median(times),
                "error": relative_error(acc_direct, acc_gc),
                "label": f"{label_prefix} {int(kr*100)}%",
                "method": method,
            }

    return results


# ─────────────────────────────────────────────────────
# 3. Nスケーリング比較
# ─────────────────────────────────────────────────────

def run_scaling_benchmark(N_list=None):
    if N_list is None:
        N_list = [50, 100, 200, 300, 500, 800]

    times_direct = []
    times_bh     = []
    times_fmm    = []
    times_gc     = []

    for N in N_list:
        p = make_random_particles(N)
        print(f"  N={N}...", end="", flush=True)

        t0 = time.perf_counter(); compute_acceleration_direct(p);      times_direct.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); compute_acceleration_barneshut(p);   times_bh.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); compute_acceleration_fmm(p);         times_fmm.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); compute_acceleration_graph(p, 0.5);  times_gc.append(time.perf_counter() - t0)
        print(" done")

    return N_list, times_direct, times_bh, times_fmm, times_gc


# ─────────────────────────────────────────────────────
# 4. プロット
# ─────────────────────────────────────────────────────

def plot_results(single_results, scaling_data, out_path="benchmark_results.png"):
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#f8f8f6")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    colors = {
        "direct":  "#444441",
        "bh":      "#534AB7",
        "fmm":     "#0F6E56",
        "gc":      "#993C1D",    # 閾値圧縮
        "gc_imp":  "#D4621A",    # 重み比例サンプリング
        "gc_spec": "#6B1F0A",    # スペクトルスパース化
    }
    gc_methods = {
        "gc":      ("threshold",  "閾値圧縮",          "^",  colors["gc"]),
        "gc_imp":  ("importance", "重み比例",          "D",  colors["gc_imp"]),
        "gc_spec": ("spectral",   "スペクトル",        "P",  colors["gc_spec"]),
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
        elif key.startswith("fmm"):
            c, mk = colors["fmm"], "s"
        elif key.startswith("gc_spec"):
            c, mk = colors["gc_spec"], "P"
        elif key.startswith("gc_imp"):
            c, mk = colors["gc_imp"], "D"
        else:
            c, mk = colors["gc"], "^"
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
    N_list, td, tbh, tfmm, tgc = scaling_data
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#f8f8f6")
    ax2.set_title("計算時間 vs N", fontsize=12, fontweight="500")
    ax2.plot(N_list, [t*1000 for t in td],  "o-", color=colors["direct"], label="直接法 O(N²)",       linewidth=1.5)
    ax2.plot(N_list, [t*1000 for t in tbh], "s-", color=colors["bh"],     label="Barnes-Hut",        linewidth=1.5)
    ax2.plot(N_list, [t*1000 for t in tfmm],"^-", color=colors["fmm"],    label="FMM",               linewidth=1.5)
    ax2.plot(N_list, [t*1000 for t in tgc], "D-", color=colors["gc"],     label="閾値圧縮 (50%)",    linewidth=1.5)
    ax2.set_xlabel("N (粒子数)", fontsize=10)
    ax2.set_ylabel("計算時間 [ms]", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── (C) グラフ圧縮 3手法: 保持率 vs 誤差 ──
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#f8f8f6")
    ax3.set_title("グラフ圧縮 3手法: 保持率 vs 誤差", fontsize=12, fontweight="500")
    krs_base = [0.3, 0.5, 0.7, 1.0]
    for prefix, (_, label, mk, c) in gc_methods.items():
        keys = [f"{prefix}_k{kr}" for kr in krs_base]
        errs = [single_results[k]["error"] for k in keys if k in single_results]
        krs_plot = [kr for kr in krs_base if f"{prefix}_k{kr}" in single_results]
        ax3.plot(krs_plot, errs, f"{mk}-", color=c, label=label, linewidth=1.5)
    ax3.set_xlabel("保持率 (keep_ratio)", fontsize=10)
    ax3.set_ylabel("相対誤差 (RMS)", fontsize=10)
    ax3.set_yscale("log")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

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
    scaling = run_scaling_benchmark([50, 100, 200, 300, 500])

    out = str(Path(__file__).parent.parent / "benchmark_results.png")
    plot_results(single, scaling, out_path=out)
    print("\n完了!")
