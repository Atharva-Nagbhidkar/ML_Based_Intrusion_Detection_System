# Intrusion Detection System for IoT Networks

A Machine Learning-based solution to detect intrusions in IoT environments using a two-stage model pipeline for improved accuracy and classification of cyberattacks.

## Overview

This project builds a smart Intrusion Detection System (IDS) for IoT networks using a hybrid approach:
1. **Binary Classification** – Detect if a network flow is malicious.
2. **Multi-Class Classification** – If malicious, classify the specific type of attack.

## Project Highlights

- **Dataset**: UNSW_NB15 (UNSW Canberra)
- **Stage 1**: Random Forest Classifier
- **Stage 2**: Sequential Neural Network
- **Objective**: Improve security monitoring for IoT by accurately identifying threats.

## Attack Categories

- DDoS
- DoS
- Reconnaissance
- Data Theft
- Backdoor
- Exploits
- Fuzzers
- Generic
- Analysis

---

## Model Architecture

### 1. Random Forest Classifier (Stage 1)

- Input: 29 selected features from UNSW_NB15 dataset
- Output: Binary label (`Normal` or `Attack`)
- Purpose: Efficient pre-filtering of benign traffic

### 2. Sequential Neural Network (Stage 2)

- Input: Same features for rows predicted as `Attack`
- Layers:
  - Dense (64 units, ReLU)
  - Dropout (0.3)
  - Dense (32 units, ReLU)
  - Output: Softmax layer (9 units for each attack type)
- Output: Attack category

---

## Technologies Used

- Python
- Scikit-learn
- TensorFlow / Keras
- Pandas, NumPy
- Matplotlib, Seaborn

---


## Setup Instructions

1. Install dependencies::
   ```sh
   pip install tensorflow scikit-learn numpy pandas matplotlib

2. Clone the repository:
   ```sh
    git clone https://github.com/Atharva-Nagbhidkar/ML_Based_Intrusion_Detection_System.git
3. Navigate to the project directory:
    ```sh
    cd ml-based-intrusion-detection-system


---

