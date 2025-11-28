<h1 align="center">ARGUS</h1>

<p align="center">
  <strong>AI-based Routine Genomic Understanding System</strong>
</p>

<p align="center">
  A Pan-Cancer Deep Learning Framework for Non-Invasive Genomic Profiling from Longitudinal Electronic Health Records
</p>


---

## Overview

**ARGUS** is a pan-cancer deep learning framework that transforms routine clinical data into a powerful precision oncology tool. By leveraging longitudinal electronic health records (EHRs) from a multi-center cohort of **80,720 patients** with **846,752 follow-up visits** across **16 major cancer types**, ARGUS enables non-invasive inference of tumor genomic alterations and therapeutic stratification.

### Key Features

- **Non-Invasive Genomic Profiling**: Infer actionable driver mutations (EGFR, KRAS, BRAF, ALK, etc.) from routine blood tests and vital signs
- **Multi-Omics Prediction**: Predict TMB, MSI status, and PD-L1 expression levels
- **Therapeutic Stratification**: Stratify patient outcomes for targeted therapies and immunotherapy
- **Model Interpretability**: SHAP-based feature attribution and phenotype-genotype mapping
- **Pan-Cancer Applicability**: Unified framework across 16 cancer types

### Performance Highlights

| Metric | Value |
|--------|-------|
| Mean AUROC (40+ driver genes) | 0.843 (0.718–0.951) |
| PD-L1 prediction accuracy | 85.4% |
| EGFR mutation AUROC (NSCLC) | 0.929 |
| Immunotherapy stratification HR | 2.13 (P < 0.01) |

## Architecture

<p align="center">
  <img src="assets/architecture.png" alt="ARGUS Architecture" width="800"/>
</p>

ARGUS employs a **dual-encoder architecture**:

1. **Static Encoder**: MLP-based encoding of demographic features (age, sex, cancer type)
2. **Temporal Transformer Encoder**: Self-attention mechanism for longitudinal EHR data
3. **Feature Fusion**: Concatenation of static and temporal representations
4. **Multi-Task Prediction Head**: Simultaneous prediction of 40+ genomic targets

## Installation

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0.0
- CUDA ≥ 11.8 (for GPU support)

### From PyPI

```bash
pip install argus-genomics
```

### From Source

```bash
git clone https://github.com/AIprojects-felix/ARGUS.git
cd ARGUS
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
pre-commit install
```

## Quick Start

### Basic Usage

```python
import torch
from argus.models import ARGUS
from argus.data import ARGUSDataset

# Initialize model
model = ARGUS(
    static_dim=18,        # Demographics + cancer type
    temporal_dim=180,     # 180+ clinical variables
    d_model=256,
    n_heads=8,
    n_layers=6,
    n_targets=43,         # 40+ genes + biomarkers
)

# Prepare data
static_features = torch.randn(32, 18)      # [batch, static_dim]
temporal_features = torch.randn(32, 100, 180)  # [batch, seq_len, temporal_dim]

# Inference
with torch.no_grad():
    output = model(static_features, temporal_features)
    predictions = output['predictions']  # [batch, n_targets]
```

### Training

```bash
# Using Hydra configuration
python scripts/train.py \
    model=argus_base \
    data.observation_window_days=180 \
    training.epochs=100 \
    training.batch_size=64
```

### Evaluation

```bash
python scripts/evaluate.py \
    checkpoint_path=experiments/outputs/best_model.ckpt \
    data.test_path=/path/to/test_data
```

### Interpretability Analysis

```bash
python scripts/interpret.py \
    checkpoint_path=experiments/outputs/best_model.ckpt \
    --shap \
    --umap \
    --penetrance
```

## Data Format

ARGUS expects longitudinal EHR data in the following format:

### Static Features (18 dimensions)
- Age (normalized)
- Sex (binary)
- Cancer type (16-dimensional one-hot)

### Temporal Features (180+ dimensions)
- **Hematology**: WBC, RBC, Hemoglobin, Platelets, Neutrophils, Lymphocytes, etc.
- **Biochemistry**: ALT, AST, ALP, LDH, Albumin, Creatinine, etc.
- **Coagulation**: PT, APTT, Fibrinogen, D-Dimer
- **Tumor Markers**: CEA, AFP, CA19-9, CA125, PSA, etc.
- **Vital Signs**: Heart rate, Blood pressure, Temperature

See [Data Format Documentation](docs/data_format.md) for detailed specifications.

## Model Configurations

| Config | d_model | Layers | Heads | Parameters |
|--------|---------|--------|-------|------------|
| `argus_small` | 128 | 4 | 4 | ~2M |
| `argus_base` | 256 | 6 | 8 | ~8M |
| `argus_large` | 512 | 12 | 16 | ~32M |

## Project Structure

```
ARGUS/
├── argus/                    # Main Python package
│   ├── models/               # Model architectures
│   │   ├── argus.py          # Main ARGUS model
│   │   ├── encoders/         # Static & Temporal encoders
│   │   ├── fusion/           # Feature fusion modules
│   │   ├── heads/            # Prediction heads
│   │   └── losses/           # Loss functions
│   ├── data/                 # Data processing
│   ├── training/             # Training utilities
│   ├── evaluation/           # Evaluation metrics
│   ├── interpretation/       # Interpretability tools
│   ├── inference/            # Inference utilities
│   ├── clinical/             # Clinical applications
│   └── utils/                # Utilities
├── configs/                  # Hydra configurations
├── scripts/                  # Training/evaluation scripts
├── notebooks/                # Jupyter notebooks
├── tests/                    # Unit and integration tests
├── docs/                     # Documentation
└── docker/                   # Docker configurations
```

## Reproducibility

To reproduce the results from our paper:

```bash
# Clone repository
git clone https://github.com/AIprojects-felix/ARGUS.git
cd ARGUS

# Install dependencies
pip install -e ".[all]"

# Download preprocessed data (if available)
# Data access requires institutional approval

# Train model
python scripts/train.py model=argus_base

# Evaluate
python scripts/evaluate.py

# Generate figures
python scripts/visualize.py
```

## Citation

If you use ARGUS in your research, please cite our paper:

```bibtex
@article{wu2025argus,
  title={A Pan-Cancer AI Framework for Non-Invasive Genomic Profiling},
  author={Wu, Liyuan and Yin, Hubin and Chen, Rui and Han, Sujun and others},
  journal={Nature Medicine},
  year={2025},
  publisher={Nature Publishing Group}
}
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was supported by grants from:
- National Natural Science Foundation of China
- National Key Research and Development Program of China
- Shanghai Shenkang Hospital Development Center

We thank all participants of the VTP (Virtual Tumor Project) consortium and the clinical staff at participating institutions.

## Contact

- **Lead Contact**: Fei Liu (liufei_2359@163.com)
- **Correspondence**: Shancheng Ren (renshancheng@gmail.com)

For questions about the code, please open an [Issue](https://github.com/AIprojects-felix/ARGUS/issues).

---

<p align="center">
  <sub>Built with ❤️ by the VTP Consortium</sub>
</p>
