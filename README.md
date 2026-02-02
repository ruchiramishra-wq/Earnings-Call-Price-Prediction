# Earnings-Call-Price-Prediction
This project studies whether earnings call transcripts can be used to predict short-horizon (1 day/5 day) stock price reactions around earnings announcements. 

We use two different ways to extract signals from the earnings call transcripts- TF-IDF and FinBERT embeddings and compare them. We also compare the effect of transcripts alone vs transcripts+finance features. 

We compare different models:
- Baseline model- Random
- Logistic regressions (with just finance features)
- Logistic regression with TF-IDF 
- Logistic regression with TF-IDF + finance features

- Simple NN using FinBERT Embeddings (Mean-Pooling and Attention Pooling)
- Simple NN using FinBERT Emeddings with Attention Pooling+ Finance features
- SVM using FinBERT Embeddings with SVM
- SVM using FinBERT Embeddings+finance features 

We used time-based splits to split our data into train-val-test datasets. For evaluation, we use AUC and get confidence intervals using bootstrap. 






# Download latest version
Try the below: 
path = kagglehub.dataset_download("tpotterer/motley-fool-data.pkl")

print("Path to dataset files:", path)

Else, download to local and read pkl file. 

# Input Data
Motley Fool Data, S&P 500 and earnings call transcripts
# Features 

# Target Definition 

# Transcript 

# Results 

While doing the bootstrapping to get the confidence intervals of the AUC score, we assume that the points are IID which is not necessarily true in financial datasets. 
