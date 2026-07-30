# GANFS: GAN-Based Feature Selection

[![PyPI version](https://img.shields.io/pypi/v/ganfs.svg)](https://pypi.org/project/ganfs/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library for **feature selection using Generative Adversarial Networks**. GANFS trains a GAN on your data and uses perturbation-based sensitivity analysis on the discriminator to rank and select the most important features.

## Installation

**Install via PyPI (Recommended):**
```bash
pip install ganfs
```

**Install from GitHub (Latest Development Version):**
```bash
pip install git+https://github.com/patelharsh15/GANFS-GAN-based-feature-selection.git
```

**From source:**
```bash
git clone https://github.com/patelharsh15/GANFS-GAN-based-feature-selection.git
cd GANFS-GAN-based-feature-selection
pip install -e .
```

## Quick Start

```python
from ganfs import GANFS
import pandas as pd

# Load your dataset
df = pd.read_csv("my_data.csv")
X = df.drop("label", axis=1)
y = df["label"]

# Initialize and train GANFS
selector = GANFS(epochs=200, batch_size=4096)
selector.fit(X, y)

# View feature ranking
ranking = selector.get_feature_ranking()
print(ranking)

# Select top 20 features
X_selected = selector.transform(X, k=20)

# Save/load trained models
selector.save("my_ganfs_model")
loaded = GANFS.load("my_ganfs_model")
```

## API Reference

### `GANFS` Class

#### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `epochs` | int | 500 | Number of GAN training epochs |
| `batch_size` | int | 4096 | Batch size for GAN training |
| `learning_rate` | float | 0.001 | Adam optimizer learning rate |
| `label_smoothing` | tuple | (0.9, 0.1) | Label smoothing for (real, fake) |
| `perturbation_mode` | str | 'dynamic' | 'dynamic' or 'static' perturbation scaling |
| `perturbation_factors` | list | [0.5, 1.0, 2.0, 5.0, 10.0] | Perturbation multipliers |
| `checkpoint_dir` | str/None | None | Directory for training checkpoints |
| `verbose` | bool | True | Print progress information |
| `random_state` | int/None | None | Random seed for reproducibility |

#### Methods

| Method | Description |
|---|---|
| `fit(X, y)` | Train GAN and compute feature sensitivities |
| `transform(X, k)` | Select top-K features from X |
| `fit_transform(X, y, k)` | Fit and transform in one step |
| `get_feature_ranking()` | Get DataFrame of features ranked by sensitivity |
| `get_feature_pairs_from_data(X, top_n)` | Analyze synergistic feature pair interactions |
| `save(path)` | Save trained model to disk |
| `GANFS.load(path)` | Load a saved model from disk |

### Usage with scikit-learn

```python
from ganfs import GANFS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Feature selection
selector = GANFS(epochs=200)
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train, k=20)
X_test_selected = selector.transform(X_test, k=20)

# Downstream classification
clf = RandomForestClassifier()
clf.fit(X_train_selected, y_train)
accuracy = accuracy_score(y_test, clf.predict(X_test_selected))
print(f"Accuracy with top-20 GANFS features: {accuracy:.4f}")
```

## How It Works

1. **GAN Training** — A Generator-Discriminator pair is trained on the feature data. The Generator learns to produce realistic synthetic samples, while the Discriminator learns to distinguish real from fake.

2. **Sensitivity Analysis** — After training, each feature is perturbed (using dynamic perturbation magnitudes scaled to each feature's natural granularity) and the discriminator's response is measured. Features that cause the largest output changes are the most discriminative.

3. **Feature Ranking** — Features are ranked by their average sensitivity scores across multiple perturbation levels and directions.

4. **Feature Selection** — The top-K features can be selected for downstream tasks (classification, regression, etc.).

## Project Structure

```
├── ganfs/                           # Python package source code
│   ├── __init__.py                  # Public API
│   ├── ganfs.py                     # Main GANFS class
│   ├── models.py                    # Generator & Discriminator networks
│   ├── sensitivity.py               # Sensitivity analysis functions
│   └── utils.py                     # GPU setup & preprocessing utilities
├── pyproject.toml                   # Package build configuration
└── research_archive/                # Original research files and experiments
    ├── GAN Algo Final.ipynb         # Original research notebook
    ├── benchmarking.ipynb           # Benchmarking vs traditional methods
    ├── training_checkpoints/        # Saved model checkpoints
    ├── feature_pair_interactions.csv
    └── feature_sensitivity_results.csv
```

## Dataset Setup (for reproducing research results)

The original research uses the **CIC-DDoS2019** dataset. The dataset files are too large (~12 GB) to host on GitHub.

### Download Instructions

1. Visit the [CIC-DDoS2019 dataset page](https://www.unb.ca/cic/datasets/ddos-2019.html)
2. Request access and download the following CSV files:
   - `DrDoS_DNS.csv`, `DrDoS_LDAP.csv`, `DrDoS_MSSQL.csv`, `DrDoS_NTP.csv`
   - `DrDoS_NetBIOS.csv`, `DrDoS_SNMP.csv`, `DrDoS_SSDP.csv`, `DrDoS_UDP.csv`
3. Place all files in a `CIC-DDoS2019/` folder at the repository root
4. Update the `base_path` in the notebook to `"./CIC-DDoS2019/"`

## Requirements

- Python 3.8+
- TensorFlow 2.x (GPU support recommended)
- NumPy, Pandas, scikit-learn

## Acknowledgments

This project uses the **CIC-DDoS2019** dataset provided by the Canadian Institute for Cybersecurity, University of New Brunswick. 

If you use the dataset, please cite the original authors:

```
Iman Sharafaldin, Arash Habibi Lashkari, Saqib Hakak, and Ali A. Ghorbani,
"Developing Realistic Distributed Denial of Service (DDoS) Attack Dataset and Taxonomy",
IEEE 53rd International Carnahan Conference on Security Technology, Chennai, India, 2019.
```

## License

MIT License — see [LICENSE](LICENSE) for details.
