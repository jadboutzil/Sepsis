# Sepsis Prediction
Sepsis Prediction Using Clinical Data (PhysioNet Computing in Cardiology Challenge 2019)

This project develops a Transformer-based model for predicting sepsis using a variety of clinical data. The model processes 10 hours of input data to forecast the likelihood of sepsis in the subsequent hour. It achieved an AUC of 0.76 on the test set.

The data for this project originates from the 2019 PhysioNet Computing in Cardiology Challenge. More details about the data and the download link are available at: https://physionet.org/content/challenge-2019/1.0.0/

The dataset consists of PSV files, with each row corresponding to one hour of patient data.

To execute the project code, follow these notebooks:
1. `psv_to_df.ipynb`: Converts the PhysioNet PSV files into a Pandas DataFrame for easier analysis.
2. `feature_engineering.ipynb`: Creates features based on 10-hour data windows and their associated labels.
3. `feature_selection.ipynb`: Analyzes feature correlations and eliminates highly correlated features.
4. `TransformerSepsis.ipynb`: Outlines the model, conducts training, and assesses its performance on the validation and test datasets.

The rest of this readme will detail the various stages of the analysis pipeline.

## 1. Redefine Output Labels
According to the PhysioNet Challenge details, the labels for the provided data are as follows:
<br>For sepsis patients, SepsisLabel is 1 if `t≥tsepsis−6` and 0 if `t<tsepsis−6`
<br>For non-sepsis patients, SepsisLabel is 0

In other words, the SepsisLabel is set to 1 six hours before the onset of sepsis. However, for the purposes of this project, sepsis only needs to be predicted one hour in advance. So the labels are redefined such that:
<br>For sepsis patients, SepsisLabel is 1 if `t≥tsepsis` and 0 if `t<tsepsis`
<br>For non-sepsis patients, SepsisLabel is 0

To actually realize this change, the first six values of SepsisLabel equals 1 are set to 0 for each patient’s data.

## 2. Window the Data
For each patient, the data is windowed into ten hour windows with an output label corresponding to the sepsis state in the eleventh hour. The window is then slid forward by one hour, until there is no more data for that subject. Note that there is no overlap of two different patients in any given window.

## 3. Backfill Missing Data for Non-Sparse Variables; Calculate Median for Sparse Variables
Many of the variables in the dataset are sparse, as is expected with clinical data. However, HR, MAP, O2Sat, SBP, Resp are relatively continuous (less than 15% missing). For these variables, any missing data is replaced with backfilling the most recent non-NaN value. 

For the remainder of the variables, summarize the window of ten hours with the median of the values in that window. If all the values in that window are NaN, then just report the median as NaN.


## 4. Feature Standardization
Each of the variables is standardized by subtracting the mean and dividing by the standard deviation. Note that the mean and standard deviation are calculated from the training set, and the same scaling factors are applied to both the training and testing sets. The test set consists of 6000 randomly sampled patients from the original 40000 patients.

## 5. Feature Correlation Analysis
Any features with high correlation are redundant and unnecessarily increase model complexity. The correlations are visualized with a heat map.


## 6. Define the Model
In each window, there are two types of data: time series data, which has a sequence length of ten, and single measurements. A recurrent neural network is the natural choice for modeling time series data, whereas a simple shallow network would be suitable for single measurement data. Consequently, two distinct models are trained and subsequently combined into one output, which is then processed by a softmax layer.

It's important to note that the second model includes a mask layer, which is an effective method for dealing with NaN values in the data. This layer necessitates replacing all NaN values with a constant value, which it then disregards during the model's training and evaluation phases. The constant used is pi, as it is a unique number, but in reality, any constant could serve this purpose.