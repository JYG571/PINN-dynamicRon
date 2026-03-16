# PINN-dynamicRon: Physics-Informed Prediction of Dynamic On-Resistance in GaN HEMTs

Welcome to our repository for **PINN-dynamicRon**. This repository provides an open-source dataset and a Physics-Informed Neural Network (PINN) based framework to predict the dynamic on-resistance ($R_{on,dyn}$) of commercial GaN HEMT devices under various off-state voltages ($V_{doff}$) and stress time.

## Key Features

* **High-Quality Dataset:** Real measurement data of commercial GaN HEMTs obtained using a **Keysight B1505A** Precision Device Analyzer.
* **Dual-Variable Prediction:** Distinct models and datasets analyzing the degradation of $R_{on,dyn}$ with respect to both **Stress Time** and **Off-State Voltage ($V_{doff}$)**.
* **Static $R_{on}$ Adaptive Mechanism:** The prediction model features a unique adaptation capability. It dynamically adjusts its device-specific predictions based on the measured **static on-resistance ($R_{on,static}$)** of the target device, ensuring high accuracy across different individual components.
* **Physics-Informed LSTM:** Integrates empirical physical models (`physics_model.py`) into the LSTM training process via customized loss functions (`loss_function.py`), ensuring predictions strictly obey underlying semiconductor physics.

## Repository Structure

The repository is organized into two main branches based on the key variables affecting dynamic $R_{on}$:

```text
PINN-dynamicRon/
│
├── StressTime/                    # Models & Data for Ron degradation over Stress Time
│   ├── Dataset/
│   │   ├── static Ron.csv         # The static Ron corresponding to different devices
│   │   └── Rstresst.csv           # Time-dependent measurement dataset
│   └── code/
│       ├── dataset_utils.py       # Data loading and preprocessing
│       ├── loss_function.py       # Physics-informed loss formulation
│       ├── model.py               # Neural network architecture definitions
│       ├── physics_model.py       # Physical model
│       ├── train_lstm_model.py    # Main training script
│       ├── verify_prediction.py   # Inference and evaluation script
│       └── README.md              # Sub-module specific documentation
│
├── Vdoff/                         # Models & Data for Ron degradation over Vdoff
│   ├── Dataset/
│   │   ├── static Ron.csv         # The static Ron corresponding to different devices
│   │   └── RVdoff.csv             # Voltage-dependent measurement dataset
│   └── code/
│       ├── config.py              # Global configuration file
│       ├── dataset_processing.py  # Data preprocessing
│       ├── loss_function.py       # Physics-informed loss formulation
│       ├── model.py               # Neural network architecture definitions
│       ├── physics_model.py       # Physical model
│       ├── train_model.py         # Main training script
│       ├── predict.py             # Predict script
│       └── README.md              # Sub-module specific documentation
│
└── README.md                      # This file
```

## Dataset Description

The datasets (Rstresst.csv and RVdoff.csv) contain experimental data extracted from commercial GaN High Electron Mobility Transistors (HEMTs).

Equipment: Keysight B1505A

Variables Recorded: Static on-resistance, dynamic on-resistance, stress time, off-state drain voltage.

## License and Data Availability

To promote open science and reproducible research, we have open-sourced both the code and the dataset used in this study:

* **Source Code:** The neural network models (including the proposed DA-PINN) and training scripts are licensed under the [MIT License](LICENSE).
* **Dataset:** The experimental data, including the dynamic $R_{on}$ measurements under various stress conditions ($V_{D,off}$ and stress time) obtained via the Keysight B1505A, are shared under the [Creative Commons Attribution 4.0 International (CC BY 4.0) License](https://creativecommons.org/licenses/by/4.0/).


## Getting Started
### Prerequisites
Ensure you have the following libraries installed:

* Python 3.8+
* TensorFlow
* NumPy
* Pandas 
* Matplotlib