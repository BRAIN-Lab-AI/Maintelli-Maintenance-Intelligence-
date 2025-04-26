# Maintelli: Maintenance Intelligence

## Introduction
In indstrial settings, unplanned equipment downtime can lead to significant operational ana financial losses. Traditional maintenance strategies, such as reactive or scheduled maintenance, often fail to prevent unexpected failures. Predictive maintenance offers a solution by utilizing real-time sensor data and machine learning techniques to anticipate equipment malfunctions.
this project aims to apply such techniques to a TA-48 multistage centrifugal compressor, enhancing reliability and efficiency, by identifying failure patterns in sensor data, the project contributes to improved predictive maintenance models.

## Project Metadata
### Authors
- **Team:** Najla Alkhaldi, Heba Almashhad and Amal Almukhlal
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** ARAMCO and KFUPM

### Project Documents
- **Presentation:** [Project Presentation](/presentation.pptx)
- **Report:** [Project Report](/report.pdf)

### Reference Paper
- [Predictive Maintenance and Fault Monitoring Enabled by Machine Learning: Experimental Analysis of a TA-48 Multistage Centrifugal Plant Compressor]([[https://arxiv.org/abs/2112.10752](https://www.researchgate.net/publication/367553293_Predictive_Maintenance_and_Fault_Monitoring_Enabled_by_Machine_Learning_Experimental_Analysis_of_a_TA-48_Multistage_Centrifugal_Plant_Compressor)](https://www.mdpi.com/2076-3417/13/3/1790))


## Project Technicalities

### Terminologies
- **LSTM:** Long Short-Term Memory is a recurrent neural network model(RNN), designed to deal with sequential data.
- **Window Size:** the number of consecutive time steps taken to predict the next step.
- **Predictive Maintenance:** A stratgy that uses machine learning to predict equipment failure before it happens.
- **Anomaly Detection:** The process of identifying data point that differ signifcantly from normal behavior.
- **Threshold:** A value that is used to determine whether an MSE value is considred anomalous.
nputs, such as textual descriptions.

### Problem Statements
- **Problem 1:** Traditional machine learning models struggle to accurately capture the complex temporal dependencies present in multivariate sensor data for predictive maintenance.
- **Problem 2:** Existing approaches for anomaly detection often rely on labeled fault data, which is scarce in real-world industrial environments.

### Loopholes or Research Areas
- **Data:** Lack of reliable real-world data to train the model.
- **LSTM interpretability:** the model is considered a balck box. Therfore is it diffcult to explain why an anaomly was flagged.‹

### Problem vs. Ideation: Proposed 2 Ideas to Solve the Problems
1. **Sequence Based Architecture:** Utilize LSTM model for temporal multivariate in sensor data. Enabling the forecating of future machine behavior.
2. **Reconstruction-Based Anomaly Detection:** Apply Mean Squared Error (MSE) between predicted and actual sensor values as a robust metric to identify deviations from normal operating conditions.

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of the forecasting-based LSTM model using Python and TensorFlow/Keras. The solution includes:

- **Deep LSTM Architecture:** A multi layer LSTM model is designed to capture trends in sensor data.
- **MSE Based Anomaly Detection:** Mean Squared Error is computed between predicted and actual sensor outputs; anomalies are flagged using a dynamic threshold derived from validation errors.

### Key Components
- **`Maintili.ipyn`**: Contains the trained model and results.

## Model Workflow
The workflow of the LSTM model is forecast future sensor readings in industrial equipment based on historical sensor data, to enable early detection of the machine failure by using anomaly detection through reconstruction error analysis:

1. **Input:**
   - **Sensor Sequence Window:** The model takes a multuvariate teime series input of sensor data such as pessure, temperature, and vibration
   - **Preprocessing:** The input data is scaled using quantile normalization to map features into uniform distribution.

2. **Forecasting Process:**
   - **LSTM-Based Sequence Modeling:** TThe input sequence is passed through stacked LSTM layers, which capture short- and long-term dependencies across sensor channels. 
   - **Prediction Output:** The final layer predicts the next time step’s sensor readings (regression task), allowing the model to forecast what should happen next under normal conditions.

3. **Output:**
   - **Reconstruction MSE:** The predicted values are compared against the actual future sensor readings. The error is calculated using Mean Squared Error (MSE) for each test sample.
   - **Anomaly Detection:** A dynamic threshold, typically based on the mean and standard deviation of validation errors, is used to flag unusually high prediction errors as anomalies — signaling potential failures.


4. **Output:**
   - **Visualization and Evaluation:** Detected anomalies are visualized over time alongside true failure labels (e.g., machine_status = 'BROKEN'), and model performance is evaluated using F1 score, confusion matrix, and ROC-AUC metrics.

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/enhanced-stable-diffusion.git
    cd enhanced-stable-diffusion
    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Train the Model:**
    Configure the training parameters in the provided configuration file and run:
    ```bash
    python train.py --config configs/train_config.yaml
    ```

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of tensorflow, Keras, and other libraries for their amazing work.
- **Individuals:** Special thanks to Dr. Muzammil Behzad for his amazing efforts, invaluable guidance and support throughout this project.

