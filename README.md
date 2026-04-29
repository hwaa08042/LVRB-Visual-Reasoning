# LVRB: Learning Visual Rule Boundaries

> A reproducible visual reasoning project that rigorously evaluates CNN abstract rule understanding and cross-rule generalization on RAVEN 3x3 RPM via rule-level ID/OOD splits.

## 1) Project Background and Research Motivation

Visual reasoning research has a persistent core question: are current models performing genuine rule reasoning, or mostly pattern memorization?  
The RAVEN dataset, built on 3x3 RPM (Raven's Progressive Matrices), is a standard benchmark for probing abstract relational reasoning.

This project is developed as a course-driven research effort and focuses on an end-to-end CNN baseline pipeline. Beyond model training, I emphasize data quality control, strict evaluation design, and reproducible reporting to investigate one key challenge in the field: cross-rule generalization under distribution shift.

## 2) Core Objectives and Research Questions

This project addresses the following research questions:

- Can a CNN trained on RAVEN 3x3 RPM learn transferable abstract rules rather than superficial visual patterns?
- How does performance change when training rules (ID) and testing rules (OOD) are strictly separated at the rule level?
- Can a full pipeline of data cleaning, anomaly handling, and training monitoring improve experiment stability and interpretability?

Corresponding objectives:

- Design and implement a strict **rule-level ID/OOD split** for controlled generalization testing.
- Build a complete and reproducible workflow: **data inspection -> cleaning/filtering -> training monitoring -> result visualization**.
- Use empirical evidence to characterize the gap between **pattern memorization** and **abstract reasoning**.

## 3) Project Highlights and Key Implementations

### 🧪 Core Experiment Design

- The central experiment is based on the RAVEN 3x3 RPM task with non-overlapping rule-level splits:
  - ID rules: `Center`, `Single`, `Outlier`
  - OOD rules: `Distribute`, `Merge`, `Progresion`
- This strict setup prevents same-rule leakage and provides a more reliable estimate of cross-rule generalization ability.

### 🛠️ Engineering Workflow and Reproducibility

- **Data pipeline**: automatic `.npz` scanning and invalid sample filtering (empty files, missing keys, corrupted labels, shape mismatches).
- **Training pipeline**: anomaly-aware training with loss/gradient checks, overfitting alerts, LR scheduling, checkpoint persistence, and checkpoint reload verification.
- **Evaluation pipeline**: automated generation of training summaries, generalization reports, robustness reports, and visual plots.

### 🎯 Research Significance

- Results show a clear gap between in-distribution fitting and out-of-distribution rule transfer, indicating that current CNN behavior is still closer to pattern memory than abstract reasoning.
- This finding aligns with a central issue in visual reasoning: high benchmark scores do not necessarily imply genuine rule understanding.

## 4) Project Structure

```text
LVRB-Visual-Reasoning/
├── data/
│   └── raven/                     # RAVEN 3x3 RPM data organized by rule folders
├── model/
│   └── cnn_baseline.py            # CNN baseline architecture and weight I/O
├── utils/
│   ├── data_loader.py             # Data loading, filtering, and dataset statistics
│   ├── contrastive_loss.py        # Contrastive learning loss module
│   ├── visualization.py           # Plotting and result visualization
│   └── code_check.py              # Utility checks for implementation consistency
├── experiments/
│   ├── generalization_test.py     # Rule-level ID/OOD generalization evaluation
│   ├── robustness_test.py         # Robustness evaluation under perturbations
│   ├── logs/                      # Training and evaluation logs
│   ├── results/                   # Generated reports and figures
│   └── weights/                   # Saved checkpoints
├── raven_dataloader.py            # Data inspection + split entry script
├── train_cnn.py                   # Main training script with monitoring
└── requirements.txt               # Dependency list
```

## 5) Quick Start (Environment Setup + Run Steps)

> The following commands are directly executable on macOS/Linux from the project root.

### Step 1. Install Poetry and create the environment

```bash
python3 -m pip install --upgrade pip
python3 -m pip install poetry
poetry env use python3
poetry shell
```

### Step 2. Install dependencies

```bash
poetry add torch torchvision numpy matplotlib pillow scikit-learn pandas
```

### Step 3. Run data loading and quality inspection (recommended first)

```bash
python raven_dataloader.py \
  --data-root data/raven \
  --batch-size 16 \
  --image-size 80
```

### Step 4. Train the CNN baseline

```bash
python train_cnn.py \
  --data_root data/raven \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-3 \
  --image_size 224
```

### Step 5. Run generalization and robustness experiments

```bash
python experiments/generalization_test.py
python experiments/robustness_test.py
```

### Step 6. Check outputs

- Logs: `experiments/logs/`
- Reports/Figures: `experiments/results/`
- Model checkpoints: `experiments/weights/`

## 6) Experimental Results and Conclusions

### 📊 Key Results (Current Repository Outputs)

- Generalization test (strict rule-level ID/OOD):
  - ID Accuracy: **23.53%**
  - OOD Accuracy: **12.24%**
  - Absolute Drop: **11.28%**
  - Relative Drop: **47.96%**
- Robustness test (excerpt):
  - Significant degradation under `random_shift` (Drop **8.16%**)

### ✅ Conclusions

- Under strict rule-level OOD settings, CNN performance drops substantially, showing limited cross-rule transfer.
- The results support a key claim: current models still rely more on pattern fitting than on abstract rule reasoning.
- A complete engineering loop (cleaning, anomaly handling, monitoring, and visualization) improves reproducibility and makes conclusions more trustworthy.

### 🖼️ Figure Placeholders

- `experiments/results/generalization_plot_contrastive_strict.png`
- `experiments/results/robustness_plot_contrastive_strict.png`
- `experiments/results/experiment_comparison_bar.png`

## 7) Personal Takeaways and Reflection

- This project strengthened my understanding of the full research loop: problem framing -> experiment design -> engineering implementation -> evidence-based interpretation.
- In visual reasoning tasks, overall accuracy alone can be misleading; rule-level splitting and OOD analysis are necessary to assess true reasoning capability.
- I now view data quality checks, anomaly monitoring, and structured reporting as essential components of scientific credibility, not optional engineering extras.

## 8) Future Improvement Directions

- Compare CNN baselines with more structured reasoning models (e.g., relation networks, symbolic constraints, or Transformer-based reasoning architectures).
- Expand OOD evaluation dimensions to compositional rules, hierarchical rule structures, adversarial perturbations, and cross-dataset transfer.
- Improve interpretability with error taxonomy analysis, feature/attention visualization, and rule-consistency metrics.
- Further automate the experimental pipeline with unified configs, batch experiment runners, result tracking, and statistical significance testing.

## 9) License (MIT)

This project is licensed under the MIT License.

You may use, modify, and distribute this project for research and educational purposes under the terms of the MIT License.
