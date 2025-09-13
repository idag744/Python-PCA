# PCA from Scratch: Dimensionality Reduction with Python

This project is a clean, from-scratch implementation of Principal Component Analysis (PCA) in Python.
It demonstrates linear algebra, eigen decomposition, and dimensionality reduction on both synthetic
datasets and the MNIST handwritten digit dataset.

## Quickstart

```bash
git clone <repo-url>
cd pca-from-scratch
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
pip install -r requirements.txt


## Run unit tests:
pytest -q


## Open the demo notebook:
jupyter notebook notebooks/mnist_pca_demo.ipynb

## Structure

pca-from-scratch/
│
├── pca/                   # Core package
│   ├── __init__.py
│   ├── pca.py             # normalize, eig, projection_matrix, PCA, PCA_high_dim
│   └── utils.py           # mse, draw_vector, visualization helpers
│
├── notebooks/
│   └── mnist_pca_demo.ipynb   # Interactive demo with plots & MNIST
│
├── data/
│   └── load_data.py           # MNIST loader (or link to sklearn.datasets.fetch_openml)
│
├── tests/
│   └── test_pca.py            # Pytest unit tests for PCA functions
│
├── requirements.txt
├── README.md
└── LICENSE
