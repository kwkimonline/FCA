# FCA (Fair Clustering via Alignment)

This repository contains the official PyTorch implementation of **FCA**, the algorithm proposed in the paper:  
["Fair Clustering via Alignment"](https://icml.cc/virtual/2025/poster/44309) by *Kunwoong Kim, Jihu Lee, Sangchul Park, and Yongdai Kim.*  
Published in **[ICML 2025](https://icml.cc/Conferences/2025)**.

---

## 📑 Table of Contents

- [Installation](#installation)  
- [Environments](#environments)  
- [Usage](#usage)  
  - [Example Commands](#example-commands)  
  - [Arguments](#arguments)  
- [Dataset](#dataset)  
- [License](#license)
- [Citation](#citation)

---

## Installation

```bash
git clone https://github.com/kwkimonline/FCA.git
cd FCA
pip install -r requirements.txt
```

## Environments

- Python >= 3.9
- numpy >= 2.2.6
- pandas >= 2.2.3
- POT >= 0.9.5
- requests >= 2.32.3
- scikit-learn >= 1.6.1
- scipy >= 1.15.3
- torch >= 2.0.0


## **Usage**

Run the FCA-C algorithm via the command-line interface:

### Example Commands:

```bash
# 1. Perfectly fair clustering on {Adult, Bank} dataset
python -m src.main --data_name {Adult, Bank} --K {number of clusters} --l2_normalize
# 2. Relaxed fair clustering (control of fairness level) on the {Adult, Bank} dataset
python -m src.main --data_name {Adult, Bank} --epsilon {[0.0, 1.0]} --K {number of clusters} --l2_normalize
# 3. Perfectly fair clustering on {Adult, Bank} dataset without L2 normalization of data
python -m src.main --data_name {Adult, Bank} --epsilon {[0.0, 1.0]} --K {number of clusters}
```

### Arguments

| Flag                 | Type    | Default | Description                                                         |
|----------------------|---------|---------|---------------------------------------------------------------------|
| `--data_name`        | string  | `Adult` | Name of dataset (must match a subfolder in `data/`)                |
| `--epsilon`          | float   | `0.1`   | Fairness tolerance (upper-bound on imbalance)                      |
| `--iters`            | int     | `50`    | Number of outer iterations (epochs)                                |
| `--iters_inner`      | int     | `1`     | Number of inner iterations per batch                               |
| `--batch_size`       | int     | `1024`  | Mini-batch size                                                    |
| `--full_batch`       | flag    | (off)   | If set, use the entire dataset as one batch                        |
| `--save_iters`       | flag    | (off)   | Save cost & balance after each iteration                           |
| `--gradient_descent` | flag    | (off)   | Enable gradient-based center updates                               |
| `--use_cuda`         | flag    | (off)   | Move center updates to GPU (requires `--gradient_descent`)         |
| `--identical_sample` | int     | `-1`    | Sample up to N points per group (for debugging)                    |
| `--balancing`        | flag    | (off)   | Load data with pre-balanced classes                                |
| `--W_sort`           | flag    | (off)   | Reverse the mask sorting order in `optimize_W`                     |
| `--numItermax`       | int     | `1000000` | Max iterations for the OT solver                                  |
| `--lr`               | float   | `0.05`  | Learning rate for gradient-based center updates                    |
| `--seed`             | int     | `2024`  | Random seed for reproducibility                                     |

## Dataset

```text
data/
├── Adult/
│   └── adult.data
├── Bank/
│   └── bank-additional-full.csv
```

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Citation

To be updated soon.
