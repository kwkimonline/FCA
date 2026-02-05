# FCA (Fair Clustering via Alignment)

This repository contains the official implementation of **FCA**, the algorithm proposed in the paper:  
["Fair Clustering via Alignment"](https://icml.cc/virtual/2025/poster/44309) by *Kunwoong Kim, Jihu Lee, Sangchul Park, and Yongdai Kim.*  
Published in **[ICML 2025](https://icml.cc/Conferences/2025)**.

---

## 📑 Table of Contents

- [Installation](#installation)  
- [Environments](#environments)  
- [Usage](#usage)  
  - [Example Commands](#example-commands)  
  - [Arguments](#arguments)  
- [Datasets](#datasets)  
- [License](#license)
- [Citation](#citation)


---

## Updates

- May 30, 2025: The first version released
- Feb 4, 2026: An updated version released

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

Run the FCA algorithm via the command-line interface.

### Example Commands:

```bash
# 1. Perfectly fair clustering on Adult dataset
python -m src.main --data_name Adult --K {number of clusters} --l2_normalize
# 2. Relaxed fair clustering (control of fairness level) on Adult dataset
python -m src.main --data_name Adult --epsilon {[0.0, 1.0]} --K {number of clusters} --l2_normalize
```

### Arguments

| Flag                   | Type    | Default   | Description                                                   |
|------------------------|---------|-----------|---------------------------------------------------------------|
| `--seed`               | int     | `2025`    | Random seed                                                   |
| `--iters`              | int     | `100`     | Number of iterations                                          |
| `--numItermax`         | int     | `1000000` | Maximum iterations for the OT solver                          |
| `--numThreads`         | int     | `5`       | Number of threads for the OT solver                           |
| `--epsilon`            | float   | `0.0`     | Fairness-controlling hyper-parameter (lower -> fairer)        |
| `--lr`                 | float   | `0.05`    | Learning rate for gradient‐based center updates               |
| `--batch_size`         | int     | `1024`    | Mini‐batch size for partitioning technique                    |
| `--data_name`          | string  | `Adult`   | Name of dataset (must match a subfolder in `data/`)           |
| `--l2_normalize`       | flag    | (off)     | Apply L2 normalization to input features                      |
| `--gradient_descent`   | flag    | (off)     | Enable gradient‐based center updates                          |
| `--use_cuda`           | flag    | (off)     | # of GPU when using --gradient_descent                        |
| `--max_iter`           | int     | `300`     | Maximum iterations for center update                          |
| `--iters_inner`        | int     | `1`       | Number of inner iterations per batch (in FCA-C)               |
| `--K`                  | int     | `-1`      | Number of clusters                                            |

## Datasets

```text
data/
├── Adult/
│   └── adult.data
```

## License
This project is licensed under the MIT License.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Citation

```text
@inproceedings{
  kim2025fair,
  title={Fair Clustering via Alignment},
  author={Kunwoong Kim and Jihu Lee and Sangchul Park and Yongdai Kim},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=jImlK83NmV}
}
```
