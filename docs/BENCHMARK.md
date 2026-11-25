# Benchmark results

Test-set metrics, mean ± std over 3 seeds. 10 labeled anomalies, 2% hidden contamination in the unlabeled pool, no labeled normals. Encoder 128-64-16, 30 pretrain + 50 train epochs. Best mean per row in bold.

Regenerate with `python scripts/benchmark.py`.

## A. Labels cover ALL anomaly types

**Test AUROC**

| Dataset | Deep SAD (semi-supervised) | Deep SVDD (unsupervised) | Isolation Forest | One-Class SVM | Random Forest (supervised) |
|---|---|---|---|---|---|
| Two Moons (2-D) | 0.995 ± 0.005 | 0.942 ± 0.020 | **0.998 ± 0.001** | 0.960 ± 0.008 | 0.954 ± 0.015 |
| Gaussian Clusters (2-D) | 0.992 ± 0.009 | 0.956 ± 0.016 | **1.000 ± 0.000** | 0.972 ± 0.003 | 0.932 ± 0.078 |
| High-Dim Tabular (20-D) | **0.985 ± 0.010** | 0.955 ± 0.023 | 0.927 ± 0.026 | 0.949 ± 0.030 | 0.942 ± 0.028 |
| Breast Cancer WDBC | **0.968 ± 0.024** | 0.853 ± 0.043 | 0.922 ± 0.038 | 0.912 ± 0.035 | 0.960 ± 0.028 |
| Digits (normal = 8) | **0.991 ± 0.008** | 0.949 ± 0.035 | 0.983 ± 0.011 | 0.983 ± 0.011 | 0.984 ± 0.013 |

**Test average precision**

| Dataset | Deep SAD (semi-supervised) | Deep SVDD (unsupervised) | Isolation Forest | One-Class SVM | Random Forest (supervised) |
|---|---|---|---|---|---|
| Two Moons (2-D) | 0.965 ± 0.032 | 0.763 ± 0.077 | **0.981 ± 0.008** | 0.914 ± 0.009 | 0.882 ± 0.029 |
| Gaussian Clusters (2-D) | 0.929 ± 0.077 | 0.683 ± 0.063 | **0.997 ± 0.001** | 0.818 ± 0.018 | 0.859 ± 0.141 |
| High-Dim Tabular (20-D) | **0.924 ± 0.039** | 0.785 ± 0.035 | 0.716 ± 0.053 | 0.846 ± 0.058 | 0.694 ± 0.129 |
| Breast Cancer WDBC | 0.881 ± 0.088 | 0.532 ± 0.118 | 0.752 ± 0.064 | 0.635 ± 0.067 | **0.933 ± 0.049** |
| Digits (normal = 8) | **0.948 ± 0.042** | 0.799 ± 0.126 | 0.907 ± 0.061 | 0.881 ± 0.081 | 0.910 ± 0.080 |

## B. Labels cover only ONE of three anomaly types (synthetic sets) – the realistic 'unknown unknowns' case

**Test AUROC**

| Dataset | Deep SAD (semi-supervised) | Deep SVDD (unsupervised) | Isolation Forest | One-Class SVM | Random Forest (supervised) |
|---|---|---|---|---|---|
| Two Moons (2-D) | 0.963 ± 0.013 | 0.937 ± 0.022 | **0.997 ± 0.002** | 0.969 ± 0.005 | 0.747 ± 0.035 |
| Gaussian Clusters (2-D) | 0.984 ± 0.005 | 0.969 ± 0.006 | **1.000 ± 0.000** | 0.991 ± 0.004 | 0.738 ± 0.031 |
| High-Dim Tabular (20-D) | **0.982 ± 0.001** | 0.954 ± 0.016 | 0.934 ± 0.023 | 0.950 ± 0.029 | 0.870 ± 0.036 |
| Breast Cancer WDBC | **0.968 ± 0.024** | 0.853 ± 0.043 | 0.922 ± 0.038 | 0.912 ± 0.035 | 0.960 ± 0.028 |
| Digits (normal = 8) | 0.980 ± 0.024 | 0.967 ± 0.025 | 0.982 ± 0.012 | **0.986 ± 0.013** | 0.975 ± 0.012 |

**Test average precision**

| Dataset | Deep SAD (semi-supervised) | Deep SVDD (unsupervised) | Isolation Forest | One-Class SVM | Random Forest (supervised) |
|---|---|---|---|---|---|
| Two Moons (2-D) | 0.774 ± 0.064 | 0.709 ± 0.100 | **0.978 ± 0.014** | 0.915 ± 0.019 | 0.541 ± 0.065 |
| Gaussian Clusters (2-D) | 0.867 ± 0.033 | 0.711 ± 0.038 | **0.997 ± 0.001** | 0.921 ± 0.024 | 0.523 ± 0.060 |
| High-Dim Tabular (20-D) | **0.901 ± 0.015** | 0.794 ± 0.041 | 0.725 ± 0.056 | 0.847 ± 0.058 | 0.596 ± 0.103 |
| Breast Cancer WDBC | 0.881 ± 0.088 | 0.532 ± 0.118 | 0.752 ± 0.064 | 0.635 ± 0.067 | **0.933 ± 0.049** |
| Digits (normal = 8) | **0.931 ± 0.071** | 0.795 ± 0.145 | 0.914 ± 0.052 | 0.898 ± 0.092 | 0.842 ± 0.095 |

_Runtime: 112s on CPU._
