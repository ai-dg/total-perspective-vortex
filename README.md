# Total Perspective Vortex 🧠

![Score](https://img.shields.io/badge/Score-96%25-brightgreen)   
**A Brain-Computer Interface (BCI) system for motor imagery classification using Common Spatial Patterns (CSP) and Logistic Regression**

> Classify motor imagery tasks from EEG signals using CSP feature extraction and machine learning. Train models to distinguish between left/right hand movements and both hands/both feet movements.

---

## ▌Project Overview

This project implements a complete **Brain-Computer Interface (BCI)** system for classifying motor imagery tasks from EEG data.\

The system uses **Common Spatial Patterns (CSP)** to extract discriminative features from multi-channel EEG signals, followed by **Logistic Regression** for classification.\

It processes data from the **EEG Motor Movement/Imagery Dataset** (PhysioNet) and classifies two types of motor imagery tasks:

- **Left Fist vs Right Fist** (runs 3, 4, 7, 8, 11, 12)
- **Both Fists vs Both Feet** (runs 5, 6, 9, 10, 13, 14)

📘 Educational BCI project: **you'll implement CSP from scratch and train classification models**.

| Electrodes (channels) | Average accuracy of 6 experiments |
|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/0a277a69-61a0-4477-b507-8d2eef960d21"  width="500"> | <img src="https://github.com/user-attachments/assets/f2294f05-e9b6-4f62-9d3c-d4e58a44cade" width="500"> |

</div>

---

## ▌Features

✔️ **CSP Algorithm**: Implements Common Spatial Patterns from scratch for feature extraction\

✔️ **EEG Data Processing**: Loads and filters EEG signals (8-30 Hz bandpass)\

✔️ **Motor Imagery Classification**: Distinguishes between different motor imagery tasks\

✔️ **Model Training**: Trains Logistic Regression classifiers with cross-validation\

✔️ **Model Save/Load**: Export and import trained models using pickle\

✔️ **Stream Mode**: Replay pre-computed epochs one by one with timing information (offline, not live acquisition)\

✔️ **Batch Experiments**: Run comprehensive experiments across all 109 subjects\

✔️ **Command Line Interface**: Full CLI with train, predict, and stream modes\

✔️ **Modular Architecture**: Separate modules for data processing, CSP, and classification

---

## ▌Dataset Information

### ■ Run Types

The dataset contains 14 runs per subject with different motor imagery tasks:

| Run | Task Type | Movement Type | T1 | T2 | Used for Classification |
|-----|-----------|---------------|----|----|------------------------|
| **R03** | Task 1 (real) | Left hand / Right hand | Left hand | Right hand | ✔️ |
| **R04** | Task 2 (imagined) | Left hand / Right hand | Left hand | Right hand | ✔️ |
| **R05** | Task 3 (real) | Both hands / Both feet | Both hands | Both feet | ✔️ |
| **R06** | Task 4 (imagined) | Both hands / Both feet | Both hands | Both feet | ✔️ |
| **R07** | Task 1 (real) | Left hand / Right hand | Left hand | Right hand | ✔️ |
| **R08** | Task 2 (imagined) | Left hand / Right hand | Left hand | Right hand | ✔️ |
| **R09** | Task 3 (real) | Both hands / Both feet | Both hands | Both feet | ✔️ |
| **R10** | Task 4 (imagined) | Both hands / Both feet | Both hands | Both feet | ✔️ |
| **R11** | Task 1 (real) | Left hand / Right hand | Left hand | Right hand | ✔️ |
| **R12** | Task 2 (imagined) | Left hand / Right hand | Left hand | Right hand | ✔️ |
| **R13** | Task 3 (real) | Both hands / Both feet | Both hands | Both feet | ✔️ |
| **R14** | Task 4 (imagined) | Both hands / Both feet | Both hands | Both feet | ✔️ |

### ■ Experiment Configurations

The system supports 6 different experiment configurations:

| Experiment ID | Runs Used | Task Type |
|---------------|-----------|-----------|
| 0 | 3, 7, 11 | Left/Right hand (real) |
| 1 | 4, 8, 12 | Left/Right hand (imagined) |
| 2 | 3, 4, 7, 8, 11, 12 | Left/Right hand (mixed) |
| 3 | 5, 9, 13 | Hands/Feet (real) |
| 4 | 6, 10, 14 | Hands/Feet (imagined) |
| 5 | 5, 6, 9, 10, 13, 14 | Hands/Feet (mixed) |

### ■ EEG Frequency Bands

| Band | Frequency | Significance |
|------|-----------|--------------|
| **Delta** | 0.5 – 4 Hz | Deep sleep (not useful) |
| **Theta** | 4 – 8 Hz | Relaxation, weak motor imagery link |
| **Alpha (µ/Mu)** | **8 – 12 Hz** | ⭐ **Sensorimotor rhythm (SMR)** - decreases (ERD) during motor imagery |
| **Beta** | **12 – 30 Hz** | ⭐ **Motor activity** - increases (ERS) during motor imagery |
| **Gamma** | 30 – 80 Hz | High-frequency cognition (often muscle noise) |

**Filtering**: The system applies a **8-30 Hz bandpass filter** to focus on alpha and beta bands, which are most relevant for motor imagery classification.

---

## ▌How it works

### ■ Method Used

The classification pipeline uses **Common Spatial Patterns (CSP)** for feature extraction followed by **Logistic Regression** for classification.

### ■ CSP Algorithm

1. **Covariance Computation**: Compute normalized covariance matrices for each class (Equation 3)
   - Separate epochs by class (label 2 vs label 3)
   - Normalize each epoch's covariance matrix by its trace
   - Average covariance matrices within each class

2. **Generalized Eigenvalue Problem**: Solve Σ⁺w = λΣ⁻w (Equation 5)
   - Find eigenvalues and eigenvectors
   - Sort eigenvalues in ascending order

3. **W Matrix Construction**: Select k=3 smallest and k=3 largest eigenvectors
   - These represent the most discriminative spatial filters
   - Stack them to form the CSP transformation matrix W (shape: 6 × n_channels)

4. **Feature Extraction**: Project epochs onto CSP space
   - z_i = W @ X_i (projection)
   - f_i = log(var(z_i, axis=1)) (log-variance features)

### ■ Classification

- **Model**: Logistic Regression with max_iter=10000
- **Cross-Validation**: k-fold with k = max(2, min(10, smallest class count))
- **Features**: 6-dimensional CSP feature vectors (from 2k filters)

### ■ Epoch Creation

- **Time window**: tmin=-0.5s to tmax=4.0s around events
- **Events**: T1 (label 2) and T2 (label 3) only
- **Filtering**: 8-30 Hz bandpass filter applied before epoch creation

---

## ▌Getting Started

### ■ Requirements

- Python 3.9+
- `numpy` (numerical operations)
- `scipy` (eigenvalue decomposition)
- `mne` (EEG data processing)
- `scikit-learn` (machine learning)
- `matplotlib` and `PyQt5` (required: `processor.py` sets the Qt5Agg backend at import)

### ■ Installation

1. Clone the repository

```bash
git clone https://github.com/ai-dg/total-perspective-vortex.git
cd total-perspective-vortex
```

2. Install dependencies

```bash
pip install numpy scipy mne scikit-learn matplotlib PyQt5
# or: ./dependencies.sh
```

3. Download the dataset

```bash
./download_data.sh ./data           # runs 03–14 (.edf and .edf.event) for S001–S109
./download_one_subject.sh S001 ./data   # a single subject
```

The data lands in `./data/` with the following structure:

```
data/
├── S001/
│   ├── S001R03.edf
│   ├── S001R03.edf.event
│   ├── S001R04.edf
│   └── ...
├── S002/
│   └── ...
└── ...
```

Download from: https://physionet.org/content/eegmmidb/1.0.0/

---

## ▌Usage Instructions

### ■ Basic Syntax

```bash
python mybci.py [SUBJECT_ID] [RUN] [MODE]
```

### ■ Available Modes

| Mode | Description |
|------|-------------|
| `train` | Train a model on the specified subject and run |
| `predict` | Predict using a pre-trained model |
| `stream` | Process epochs one by one with timing information |

### ■ Usage Examples

#### 1. Train a model

```bash
# Train on subject 1, run 4 (left/right hand imagined)
python mybci.py 1 4 train

# Train on subject 2, run 14 (hands/feet imagined)
python mybci.py 2 14 train
```

#### 2. Predict with a trained model

```bash
# Predict on subject 1, run 13 (full output)
python mybci.py 1 13 predict

# The model will automatically load the appropriate model file
# based on the run type (left_fist_right_fist or both_fists_both_feet).
# There is one model file per run type, so predicting on a subject uses
# the model of whichever subject was trained last for that run type.
```

#### 3. Stream mode (epoch by epoch)

```bash
# Process epochs one by one with timing
python mybci.py 1 4 stream
```

#### 4. Run comprehensive experiments

```bash
# Run all 6 experiments across all 109 subjects
python mybci.py

# This will:
# - Train on the first run of each experiment for every subject (overwrites models/*.pkl)
# - Test on the other runs of that experiment
# - Print per-subject accuracy, the mean per experiment and the overall mean
```

---

## ▌Example Output

### Training

```bash
$ python mybci.py 4 14 train

launching training...
Run types: 
left_fist_right_fist : runs 3,4,7,8,11,12
both_fists_both_feet : runs 5,6,9,10,13,14
[0.6666 0.4444 0.4444 0.4444 0.4444 0.6666 0.8888 0.1111 0.7777 0.4444]
cross_val_score: 0.5333
Model saved in ./models/both_fists_both_feet.pkl
```

### Prediction

```bash
$ python mybci.py 4 14 predict

epoch nb: [prediction] [truth] equal?
epoch 00:         [02]    [02] True
epoch 01:         [03]    [03] True
epoch 02:         [02]    [02] True
...
Accuracy: 0.5333
```

### Stream Mode

```bash
$ python mybci.py 1 4 stream

chunk 00: pred=2, truth=2, time=0.001234s
chunk 01: pred=3, truth=3, time=0.001156s
chunk 02: pred=2, truth=2, time=0.001198s
...
```

---

## ▌Project Structure

```
total-perspective-vortex/
├── mybci.py              # Main script (entry point)
├── logreg.py             # Logistic Regression training and prediction
├── mycsp.py              # CSP algorithm implementation
├── processor.py          # EEG data loading and preprocessing
├── csp-example.py        # MNE CSP reference example
├── download_data.sh      # PhysioNet download, all subjects
├── download_one_subject.sh  # PhysioNet download, one subject
├── BlaTomLemKawMue08.pdf # CSP paper (equations 3 and 5)
├── models/               # Pre-trained models
│   ├── left_fist_right_fist.pkl
│   └── both_fists_both_feet.pkl
├── data/                 # EEG dataset (not included)
│   ├── S001/
│   ├── S002/
│   └── ...
├── vortex.mplstyle       # Matplotlib style configuration
└── README.md            # This file
```

---

## ▌Technical Details

### Architecture

The project follows a modular architecture:

- **`processor.py`**: Loads EDF files, applies bandpass filtering (8-30 Hz), creates epochs
- **`mycsp.py`**: Implements CSP algorithm (covariance matrices, eigenvalue problem, W matrix, feature extraction)
- **`logreg.py`**: Trains Logistic Regression models, handles model save/load, prediction pipeline
- **`mybci.py`**: Command-line interface, orchestrates training and prediction workflows

### Key Electrodes

For motor imagery classification, the most important electrodes are:

| Electrode | Zone | Function |
|-----------|------|----------|
| **C3** | Left motor cortex | Right hand movement/imagery |
| **C4** | Right motor cortex | Left hand movement/imagery |
| **Cz** | Central midline | Trunk/legs control |
| **FC3, FC4** | Pre-motor | Movement preparation |
| **CP3, CP4** | Post-motor | Sensory feedback |

### Code Quality

- Follows Python PEP 8 standards
- Comprehensive docstrings with Logic/Return format
- Modular design with clear separation of concerns
- Error handling for file operations
- Flake8 clean on the four main modules (79 character line limit)

---

## ▌Performance Results

### ■ Cross-Validation Scores

The one run logged in this README (subject 4, run 14) scores **0.53**, close to chance for two classes. Scores vary with:
- Subject variability
- Run type (real vs imagined movements)
- Number of epochs available

### ■ Model Performance

Classification accuracy varies widely by subject and task type — the single run logged above sits near chance — and depends on:
- **Task difficulty**: Real movements generally easier than imagined
- **Subject**: Individual differences in EEG signal quality
- **Data quality**: Number of valid epochs after filtering

---

## ▌Theoretical Background

### Common Spatial Patterns (CSP)

CSP is a spatial filtering technique that finds linear combinations of EEG channels that maximize the variance for one class while minimizing it for another.

**Mathematical Formulation**:

1. **Covariance matrices** (Equation 3):
   - C⁺ = mean of normalized covariance matrices for class +
   - C⁻ = mean of normalized covariance matrices for class -

2. **Generalized eigenvalue problem** (Equation 5):
   - $C⁺w = λC⁻w$
   - Eigenvectors w represent spatial filters
   - Eigenvalues λ indicate discriminative power

3. **Feature extraction**:
   - Project signal: $z = W @ X$
   - Compute log-variance: $f = log(var(z))$

### Motor Imagery and EEG

- **Event-Related Desynchronization (ERD)**: Alpha (8-12 Hz) power decreases during motor imagery
- **Event-Related Synchronization (ERS)**: Beta (12-30 Hz) power increases after motor imagery
- **Spatial patterns**: C3/C4 show contralateral activation (left hand imagery → right hemisphere)

---

## ▌References

- **Dataset**: [EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)
- **CSP Algorithm**: Blankertz, Tomioka, Lemm, Kawanabe and Müller, "Optimizing Spatial Filters for Robust EEG Single-Trial Analysis", IEEE Signal Processing Magazine, 2008 (`BlaTomLemKawMue08.pdf`)
- **MNE-Python**: [Documentation](https://mne.tools/stable/index.html)

---

## 📜 License

This project was completed as part of an **academic curriculum**.\

It is intended for **educational purposes** and demonstrates implementation of CSP and motor imagery classification from scratch.

If you wish to use or study this code, please ensure it complies with **your institution's policies**.
