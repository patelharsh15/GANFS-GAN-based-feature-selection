# GANFS: GAN-Based Feature Selection for DDoS Detection

A novel approach to feature selection using Generative Adversarial Networks (GANs) for network intrusion detection, specifically targeting DDoS attacks. This project leverages the CIC-DDoS2019 dataset and demonstrates that GAN-based feature selection can achieve competitive or superior performance compared to traditional feature selection methods.

## Project Structure

```
├── GAN Algo Final.ipynb        # Main GANFS algorithm notebook (training + feature selection)
├── benchmarking.ipynb           # Benchmarking against traditional feature selection methods
├── GAN Algo.ipynb               # Earlier version of the GAN algorithm
├── GAN Algo bkp.ipynb           # Backup of the GAN algorithm
├── training_checkpoints/        # Saved model checkpoints from training
├── 1/                           # Additional model experiments
│   ├── discriminator_model2.keras
│   ├── generator_model2.keras
│   └── training_checkpoints2/
├── feature_pair_interactions.csv # Feature interaction analysis results
├── feature_sensitivity_results.csv # Feature sensitivity analysis results
├── submission/                  # Submission-related files
└── vm resources.txt             # VM resource notes
```

## Dataset Setup

This project uses the **CIC-DDoS2019** dataset. The dataset files are too large (~12 GB) to host on GitHub, so you must download them manually.

### Download Instructions

1. **Visit the official CIC-DDoS2019 dataset page:**
   - https://www.unb.ca/cic/datasets/ddos-2019.html

2. **Request access** to the dataset by filling out the form on the page (if required).

3. **Download the following CSV files** from the dataset (Training Day — January 12, 2019):
   - `DrDoS_DNS.csv`
   - `DrDoS_LDAP.csv`
   - `DrDoS_MSSQL.csv`
   - `DrDoS_NTP.csv`
   - `DrDoS_NetBIOS.csv`
   - `DrDoS_SNMP.csv`
   - `DrDoS_SSDP.csv`
   - `DrDoS_UDP.csv`

4. **Place all downloaded CSV files** into a folder named `CIC-DDoS2019/` at the root of this repository:
   ```
   GAN_project/
   └── CIC-DDoS2019/
       ├── DrDoS_DNS.csv
       ├── DrDoS_LDAP.csv
       ├── DrDoS_MSSQL.csv
       ├── DrDoS_NTP.csv
       ├── DrDoS_NetBIOS.csv
       ├── DrDoS_SNMP.csv
       ├── DrDoS_SSDP.csv
       └── DrDoS_UDP.csv
   ```

5. **Update the `base_path` variable** in the notebook to point to your local dataset directory:
   ```python
   base_path = "./CIC-DDoS2019/"
   ```

## Requirements

- Python 3.8+
- TensorFlow 2.x (with GPU support recommended)
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- psutil

Install dependencies:
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib psutil
```

## Usage

### 1. Run the GANFS Algorithm

Open and run `GAN Algo Final.ipynb`:
- Loads and preprocesses the CIC-DDoS2019 dataset
- Trains the GAN (Generator + Discriminator)
- Performs GAN-based feature selection using discriminator weight analysis
- Evaluates selected features with downstream classifiers

### 2. Run Benchmarking

Open and run `benchmarking.ipynb`:
- Compares GANFS against traditional methods (Chi-Square, Mutual Information, ANOVA F-test)
- Evaluates with Random Forest, Logistic Regression, and SVM classifiers
- Reports Accuracy, Precision, Recall, F1-Score, and AUC-ROC

## Methodology

1. **Data Preprocessing** — Load 8 DDoS attack CSVs, sample 500K non-benign records per file, convert labels to binary (BENIGN=0, Attack=1)
2. **GAN Training** — Train a Generator-Discriminator pair on attack traffic features
3. **Feature Ranking** — Extract discriminator weights to rank features by importance
4. **Feature Selection** — Select top-K features based on GAN-derived importance scores
5. **Evaluation** — Train classifiers on selected features and compare with baselines

## License

This project is for academic and research purposes.

## Acknowledgments

- **CIC-DDoS2019 Dataset**: Canadian Institute for Cybersecurity, University of New Brunswick
- Iman Sharafaldin, Arash Habibi Lashkari, Saqib Hakak, and Ali A. Ghorbani, "Developing Realistic Distributed Denial of Service (DDoS) Attack Dataset and Taxonomy", IEEE 53rd International Carnahan Conference on Security Technology, Chennai, India, 2019.
