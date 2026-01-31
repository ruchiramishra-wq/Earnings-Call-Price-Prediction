# Earnings-Call-Price-Prediction
NLP model that consumes language used in earnings calls and predict next 3/5/10 day price moves

Consume Earnings call transcripts data from Kaggle: 

import kagglehub

# Download latest version
Try the below: 
path = kagglehub.dataset_download("tpotterer/motley-fool-data.pkl")

print("Path to dataset files:", path)

Else, download to local and read pkl file. 

# Input Data

# Features 

# Target Definition 

# Transcript 

# Results 

While doing the bootstrapping to get the confidence intervals of the AUC score, we assume that the points are IID which is not necessarily true in financial datasets. 
