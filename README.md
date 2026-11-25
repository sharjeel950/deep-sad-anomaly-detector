# 🛰️ Deep SAD — Semi-supervised Anomaly Detector

An interactive **Streamlit** workbench for **Deep SAD** (*Deep Semi-supervised Anomaly Detection*, Ruff et al., ICLR 2020), implemented from scratch in **PyTorch**. It learns what "normal" looks like from mostly-unlabeled data and sharpens the decision boundary with only a handful of labeled anomalies.

> **Project topic:** Semi-supervised anomaly detector (Streamlit) · **Algorithm:** Deep SAD (Deep Semi-supervised SVDD)

---

## Features

| Area | What you get |
|---|---|
| **Algorithm** | Faithful Deep SAD: autoencoder pretraining → centre initialisation → semi-supervised hypersphere training with the inverse-distance term for labeled anomalies |
| **Data** | 3 synthetic generators (2-D and high-D, each with 3 anomaly *types*), 2 real datasets (WDBC, Digits – offline), and **CSV upload** with any label column |
| **Semi-supervised protocol** | Control labeled anomalies, labeled normals, hidden contamination of the unlabeled pool and **how many anomaly types the labels cover** ("unknown unknowns") |
| **Baselines** | Deep SVDD (unsupervised), Isolation Forest, One-Class SVM, Random Forest (supervised) – same training information |
| **Evaluation** | AUROC, average precision, best-F1, ROC / PR curves, interactive threshold with confusion matrix, score histograms |
| **Interpretability** | Latent-space projection with the hypersphere centre, anomaly-score surface for 2-D data |
| **Label-efficiency study** | AUROC vs. number of labeled anomalies (0 labels = Deep SVDD) |
| **Deployment** | Score pasted/edited rows or a CSV, download scored results, export a model checkpoint |
| **Engineering** | Modular `src/` package, 15 automated tests (unit + headless UI), reproducible benchmark script |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                         # CPU PyTorch is enough
streamlit run app.py
```

Then: **Train** tab → *Train Deep SAD* → **Evaluate** tab. The default settings finish in a few seconds on a laptop CPU.

## The algorithm in one minute

Given `n` unlabeled samples and `m` labeled samples with `ỹ = +1` (normal) or `−1` (anomaly):

```
min_W   1/(n+m) Σ_i  ‖φ(x_i;W) − c‖²                    ← unlabeled: pulled to the centre c
      + η/(n+m) Σ_j ( ‖φ(x̃_j;W) − c‖² )^ỹ_j            ← labeled normal: pulled in, labeled anomaly: pushed out (inverse distance)
      + λ/2 Σ_ℓ ‖W^ℓ‖²_F
```

Anomaly score: `s(x) = ‖φ(x;W) − c‖²`. With **no labels** this is exactly Deep SVDD.

Pipeline implemented in `src/trainer.py`:

1. **Pretrain** a bias-free autoencoder (labeled anomalies are excluded so they are not reconstructed).
2. **Initialise c** as the mean embedding; coordinates with `|c_k| < 0.1` are moved to ±0.1.
3. **Train** the encoder with the Deep SAD loss (Adam, step LR decay at 80 % of the epochs).
4. **Score** with the squared distance to `c`; choose a threshold from a quantile of the training scores or interactively.

### Design decisions

* **Bias-free layers + batch-norm without affine parameters** – prevents *hypersphere collapse* (mapping everything to `c`).
* **Centre kept away from 0** – otherwise the network can reach it trivially.
* **Labeled anomalies excluded from autoencoder pretraining** and from the centre computation.
* **Small additions beyond the paper:** gradient-norm clipping (5.0) for stability with the inverse-distance term; features standardised with a scaler fitted on the training set only.
* **No test leakage:** the test split is fully labeled and is never used for scaling, pretraining, centre initialisation or thresholding.

## Evaluation protocol

`make_semi_supervised_split` in `src/data.py` builds what the learner sees:

* **Test set** – stratified hold-out, natural anomaly ratio, fully labeled.
* **Train set** = unlabeled pool (normals + a few *hidden* anomalies) ∪ optional labeled normals ∪ *k* labeled anomalies drawn only from the first *t* anomaly types.

Baselines get identical information: unsupervised ones see everything except the labeled anomalies; the Random Forest treats labeled anomalies as class 1 and the rest as class 0.

## Benchmark (test AUROC, mean ± std over 3 seeds, 10 labeled anomalies)

| Dataset | Deep SAD | Deep SVDD | Isolation Forest | One-Class SVM | Random Forest |
|---|---|---|---|---|---|
| Two Moons (2-D) | 0.995 | 0.942 | **0.998** | 0.960 | 0.954 |
| Gaussian Clusters (2-D) | 0.992 | 0.956 | **1.000** | 0.972 | 0.932 |
| High-Dim Tabular (20-D) | **0.985** | 0.955 | 0.927 | 0.949 | 0.942 |
| Breast Cancer WDBC | **0.968** | 0.853 | 0.922 | 0.912 | 0.960 |
| Digits (normal = 8) | **0.991** | 0.949 | 0.983 | 0.983 | 0.984 |

What the numbers say (full tables incl. average precision and the "labels cover only one anomaly type" setting are in [`docs/BENCHMARK.md`](docs/BENCHMARK.md)):

* Adding 10 labeled anomalies lifts Deep SAD well above its unsupervised special case (**Deep SVDD**) on every dataset, e.g. 0.853 → 0.968 AUROC on WDBC.
* On **low-dimensional toy data Isolation Forest is excellent** and Deep SAD does not beat it – a deep network has little to offer in 2-D. Deep SAD's advantage appears on **higher-dimensional and real tabular data**.
* When labels cover only one of three anomaly types, the **supervised Random Forest degrades sharply** (e.g. 0.954 → 0.747 on Two Moons) while Deep SAD stays robust – the practical argument for semi-supervised learning.

Reproduce: `python scripts/benchmark.py --seeds 3 --labels 10`

## Project structure

```
deep-sad-anomaly-detector/
├── app.py                  # Streamlit UI (Overview · Data · Train · Evaluate · Label efficiency · Inference)
├── src/
│   ├── model.py            # bias-free encoder / decoder / autoencoder
│   ├── trainer.py          # DeepSAD: pretrain, init_center, fit, score, save/load, spawn
│   ├── data.py             # generators, real datasets, CSV loader, semi-supervised split
│   ├── baselines.py        # Deep SVDD, Isolation Forest, One-Class SVM, Random Forest
│   ├── metrics.py          # AUROC / AP / F1, thresholds, confusion matrix
│   └── plots.py            # all Plotly figures
├── tests/                  # pytest: 12 unit tests + 3 headless UI tests
├── scripts/benchmark.py    # reproducible comparison → docs/BENCHMARK.md
├── docs/                   # BENCHMARK.md, VIVA_QA.md
└── .streamlit/config.toml  # theme
```

## Using your own data

Choose **Upload your own CSV** in the sidebar, pick the label column and tick which value(s) mean *anomaly*. Numeric columns are used directly, low-cardinality text columns are one-hot encoded, missing values are median-imputed. For scoring new data later, the file needs the same feature columns (a template CSV can be downloaded in the **Inference** tab).

## Testing

```bash
pytest -q          # 12 unit tests + 3 Streamlit AppTest smoke tests
```

## Limitations & ideas for extension

* Fully-connected networks only – add CNN encoders for images or LSTMs for time series.
* The threshold comes from an assumed contamination rate; a small labeled validation set would allow calibrated thresholds.
* η, latent size and epochs are set by hand; add Optuna search with label-aware validation.
* Compare against more methods (e.g. DevNet, Autoencoder-based scores, Deep SVDD with soft boundary).

## References

* L. Ruff, R. A. Vandermeulen, N. Görnitz, A. Binder, E. Müller, K.-R. Müller, M. Kloft. **Deep Semi-Supervised Anomaly Detection.** ICLR 2020 (arXiv:1906.02694).
* L. Ruff et al. **Deep One-Class Classification.** ICML 2018.
* F. T. Liu, K. M. Ting, Z.-H. Zhou. **Isolation Forest.** ICDM 2008.
