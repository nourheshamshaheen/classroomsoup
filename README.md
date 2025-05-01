# 🥣 ClassroomSoup

This is part of the final project for the course INF6953PE: Deep Learning Dynamics (Winter 2025) with Professor Sarath Chandar. This is the work of Nour Shaheen.

This repository presents a **curriculum‑learning & model‑merging** code for image‑classification.

Out‑of‑the‑box you can:

* Train **data curricula** (easy‑to‑hard sampling) or **task curricula** (coarse‑to‑fine labels) on **CIFAR‑100** or **ImageNet**
* Generate or reuse difficulty scores (C‑scores) and validation splits
* Log every run to **Comet** with one flag
* Evaluate checkpoints in a single command
* Merge checkpoints with **greedy soup, iterative uniform soup, or Fisher‑weighted averaging**

---

## Table of Contents
1. [Setup Environment](#setup-environment)
2. [Project Overview](#project-overview)
3. [Folder Structure](#folder-structure)
4. [Quick Start](#quick-start)
5. [Training Options](#training-options)
6. [Validation Split & C‑Scores](#validation-split--c-scores)
7. [Model Merging](#model-merging)
8. [Testing a Checkpoint](#testing-a-checkpoint)
9. [Environment Variables](#environment-variables)
10. [Contributing](#contributing) & [License](#license)

---

## Setup Environment


### 1. Clone the Repository
```bash
git clone https://github.com/nourheshamshaheen/classroomsoup.git
cd classroomsoup
```

### 2. Requirements and Environment
1. **Create a conda environment**
   ```bash
   conda create -n class_soup python=3.12
   # ‑‑or‑‑
   conda create --prefix $HOME/class_soup python=3.12
   ```

2. **Activate the environment**
   ```bash
   conda activate class_soup
   # ‑‑or‑‑
   conda activate $HOME/class_soup
   ```

3. **Install a CUDA‑compatible PyTorch build**

    Install the correct prebuilt PyTorch version before installing other libraries. It had to be compatible with your CUDA driver. For example, if you have CUDA 11.7, you should install PyTorch 2.0.1 compiled with cu117:

   ```bash
   pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117      --index-url https://download.pytorch.org/whl/cu117
   ```

4. **Install project dependencies** (ensure you have Rust)
   ```bash
   pip install --no-deps -r requirements.txt
   ```

5. **(Optional) pre‑commit hooks**
   ```bash
   pip install pre-commit ruff black
   pre-commit install
   pre-commit run --all-files
   ```

### 3. Set Environment

This project requires environment variables for proper configuration.

1. Create a `.env` file: Copy the provided `.env.example` file in the repository to create a new .env file in the project root.

```bash
cp .env.example .env
```

2. Edit the `.env` file: Open the `.env` file and add your specific environment variables as key-value pairs.

3. Load the environment variables:

```bash
source scripts/load_env.sh
```

---

### Project Overview

| Curriculum Type | Script(s) | What It Does |
|-----------------|-----------|--------------|
| **Data**        | `scripts/data_curriculum/*` | Presents *easiest* samples first using C‑scores and a pacing scheduler |
| **Task**        | `scripts/task_curriculum/*` | Trains on *coarse* labels (stage 1), then *fine* labels (stage 2) |
| **Regular**     | `scripts/regular_training.sh` | Baseline without curriculum |
| **Merging**     | `src/merge.py` | Greedy soup, iterative uniform soup, Fisher averaging |
| **Testing**     | `scripts/testing.sh` | Benchmarks any `.ckpt`/`.pth` on the test set |
---

## Folder Structure
```text
classroomsoup/
├── scripts/           # Ready‑to‑run bash experiments
│   ├── data_curriculum/
│   ├── task_curriculum/
│   └── testing.sh
├── split/             # Pre‑computed val/index split(s)
├── src/
│   ├── cscores/       # Difficulty scores (CIFAR‑100 & ImageNet)
│   ├── curriculum_data.py
│   ├── gen_val_indices.py
│   ├── main.py
│   ├── reg_train.py
│   ├── merge.py
│   └── ...
└── README.md          # ← you are here
```

---

## Quick Start

### 1. Data Curriculum on CIFAR‑100 (linear pacing)
```bash
bash scripts/data_curriculum/train_data_curriculum_linear_cifar100.sh
```

### 2. Task Curriculum (coarse → fine)
```bash
bash scripts/task_curriculum/task_coarse_experiment_CIFAR100.sh   # stage 1
bash scripts/task_curriculum/task_fine_experiment_CIFAR100.sh     # stage 2
```

### 3. Regular Baseline
```bash
bash scripts/regular_training.sh
```

Logs go to **Comet** when `--use_comet` is passed and your `COMET_API_KEY` is set.

---

## Training Options
If you need custom hyper‑parameters, edit the corresponding script or create a new one.
Key flags inside the scripts:

| Flag | Example | Meaning |
|------|---------|---------|
| `--curriculum_type` | `data` \| `task` | Choose curriculum flavour |
| `--pacing`          | `linear` \| `exponential` | Data‑curriculum pacing |
| `--starting_percent`| `0.3` | Initial data fraction |
| `--step`            | `15`  | Epochs between pacing updates |
| `--variant`         | `staged` \| `continuous` | Task‑curriculum strategy |
| `--val_indices_path`| `split/cifar100_val_split.npz` | Reuse or replace validation split |

---

## Validation Split & C‑Scores

* A ready‑made stratified split for CIFAR‑100 lives at **`split/cifar100_val_split.npz`**.
* Want a fresh split?
  ```bash
  python src/gen_val_indices.py  # creates split/cifar100_val_split.npz
  ```
* Difficulty scores (C‑scores) are pre‑computed in **`src/cscores/`**.

---

## Model Merging

Merge several checkpoints into a single network:

```bash
python src/merge.py \
  --technique greedy \                   # greedy | iterative | fisher
  --model-name resnet50 \
  --num-classes 100 \
  --checkpoints path/to/*.ckpt \
  --curriculum-type data \
  --dataset CIFAR100
```

For Fisher averaging add `--compute-fisher` **or** `--fisher-paths ...`.

---

## Testing a Checkpoint

```bash
bash scripts/testing.sh \
  --checkpoint checkpoints/model.ckpt \
  --model_name resnet18 \
  --dataset CIFAR100 \
  --batch_size 128 \
  --device cuda
```

Outputs **macro‑F1** and **accuracy** on the test set.

---

## Environment Variables
| Variable | What it’s for |
|----------|---------------|
| `COMET_API_KEY` | Comet authentication |
| Any custom keys | Define them in `.env` and load with `source scripts/load_env.sh` |

---

## Contributing
Pull requests are welcome. Please run the pre‑commit hooks before pushing.

## License
This project is released under the MIT License.

---

*Happy souping!*


Note: C‑scores are taken from “Characterizing Structural Regularities of Labeled Data in Over‑parameterized Models” (Z. Jiang et al., ICML 2020).​
