# nbody_graph — グラフ圧縮によるN体問題ソルバー

直接法・Barnes-Hut法・FMM・**グラフ圧縮法（独自手法）** の比較フレームワーク。

## 構成

```
nbody_graph/
├── utils/
│   └── core.py              # Particles クラス、Leapfrog積分、エネルギー計算
├── methods/
│   ├── direct.py            # 直接法 O(N²) — 基準実装
│   ├── barnes_hut.py        # Barnes-Hut法 O(N log N)
│   ├── fmm.py               # 簡略FMM（多重極展開）
│   └── graph_compression.py # グラフ圧縮法（独自手法・ここを拡張）
└── benchmark/
    └── run_benchmark.py     # 全手法の精度・速度比較スクリプト
```

## 実行

```bash
pip install -r requirements.txt
python -m nbody_graph.benchmark.run_benchmark
```

## グラフ圧縮法の拡張ポイント

`methods/graph_compression.py` の `compress_graph_threshold()` を置き換えることで
任意の圧縮アルゴリズムを差し込める：

- スペクトルスパース化（Spielman-Srivastava）
- コミュニティ検出（Louvain / Leiden）
- コアセット近似
- ランダムウォークベースのクラスタリング
