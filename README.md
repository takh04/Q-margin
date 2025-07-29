# Q-margin

[**Understanding Generalization in Quantum Machine Learning with Margins**](https://arxiv.org/abs/2411.06919)  
*Accepted at ICML 2025*

---

Q-margin is the official research code accompanying our ICML 2025 paper.  
The project investigates how *margins*—a classical concept from statistical learning theory—translate to the quantum realm and how they correlate with generalization in quantum machine-learning (QML) models.

The repository contains two independent experimental pipelines:

1. **QPR** – *Quantum Phase Recognition* on 1-D spin chains (Cluster, TFIM, XXZ) using variational quantum circuits and Quantum Convolutional Neural Networks (QCNNs).
2. **NQE** – *Neural-Quantum Embedding* on image datasets (MNIST, Fashion-MNIST, KMNIST) that combines classical pre-processing with various quantum classifiers (IQP, TQE, NQE).

Both pipelines measure *margin distributions*, *generalization gaps*, and several *effective parameters* metrics providing large-scale empirical evidence for our theoretical results.

## Repository Structure

```
├── QPR/                   # Quantum Phase Recognition experiments
│   ├── data/              # Pre-generated spin-chain ground states & generators
│   ├── models/            # QCNN architectures and helpers
│   ├── results_*/         # Saved results (each subfolder = sweep)
│   └── run.py             # Entry-point for QPR sweeps
│
├── NQE/                   # Neural-Quantum Embedding experiments
│   ├── models/            # NQE, IQP, TQE models + utilities
│   ├── results/           # Saved results and trained checkpoints
│   ├── Figures/           # Plots created by notebooks / scripts
│   ├── data.py            # Image loading & PCA helpers
│   └── run.py             # Entry-point for NQE sweeps
│
├── LICENSE                # Apache-2.0
├── README.md              # ← you are here
└── .gitignore
```


## Citation

If you find this repository useful in your research, please cite our work:

```
@article{hur2024understanding,
  title={Understanding Generalization in Quantum Machine Learning with Margins},
  author={Hur, Tak and Park, Daniel K},
  journal={arXiv preprint arXiv:2411.06919},
  year={2024}
}
```

## License

Q-margin is released under the **Apache 2.0** license (see `LICENSE`).

## Contact

For questions or collaborations, please open an issue or email **takh0404@yonsei.ac.kr**.
