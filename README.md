# Earnings-Call-Price-Prediction

## Overview
This project predicts short-term stock price movements (1-day and 5-day horizons) following earnings announcements by analyzing earnings call transcripts combined with financial metrics.

## Motivation
Earnings calls contain valuable qualitative information (management tone, forward guidance) that may not be captured in quantitative metrics alone. This project tests whether NLP can extract predictive signals for trading strategies.

## Data
- **Transcripts**: Motley Fool earnings call transcripts
- **Financial Data**: S&P 500 price data, Yahoo finance
- **Time Period**: 2019-04-17 to 2022-12-01
- **Sample Size**: 3544 

## Methodology

### Target Variable
Binary classification: Market-adjusted return (subtract the market return value) increases vs decreases N days post-earnings:
-abn_return=stock_return-market_return
-where n-day return is defined as $return_{n}=(Price_{t+n}-Price_t)/Price_t$
                   
### Feature Engineering
**Text Features:**
- TF-IDF: TF-IDF represents documents as sparse vectors that weight words by their frequency in a document and their rarity across the corpus.
- FinBERT Embeddings: We use the FinBERT model (BERT model trained on financial data) to get vectors for each transcript by doing mean-pooling and attention pooling over chunks. 
  
**Financial Features:**
- Market adjusted historical 20-day volatility (where volatility=standard deviation)
- Historical 1-day market-adjusted return
- Historical 5-day market-adjusted return
- Historical 20-day market-adjusted return

### Models Compared
1. Random Baseline
2. Logistic Regression (financial features only)
3. Logistic Regression + TF-IDF
4. Neural Network + FinBERT (mean pooling)
5. Neural Network + FinBERT (attention pooling)
6. Neural Network + FinBERT + financial features

### Validation
- Time-based train/validation/test splits (prevents lookahead bias)
- Evaluation metric: AUC-ROC with 95% confidence intervals (bootstrap)

## Results
| Model | Test AUC | Test AUC CI | Test SE | Return Period |
|-------|----------|-------------|---------|---------------|
| random (Baseline) | 0.4751 | (0.4101, 0.5372) | 0.0328 | 1-Day |
| finance_only (Baseline) | 0.4782 | (0.4132, 0.5437) | 0.0330 | 1-Day |
| tfidf (Baseline) | 0.5239 | (0.4603, 0.5907) | 0.0334 | 1-Day |
| finance_tfidf (Baseline) | 0.4630 | (0.4022, 0.5302) | 0.0332 | 1-Day |
| AttnMLPPoolClassifier (Transcript Only) | 0.5098 | (0.4413, 0.5808) | 0.0357 | 1-Day |
| AttnPoolTwoTower (Transcript + Finance) | 0.4263 | (0.3612, 0.4928) | 0.0337 | 1-Day |
| MeanPoolClassifier (Transcript Only) | 0.4805 | (0.4145, 0.5508) | 0.0350 | 1-Day |
| random (Baseline) | 0.4470 | (0.3843, 0.5134) | 0.0328 | 5-Day |
| finance_only (Baseline) | 0.4614 | (0.3907, 0.5256) | 0.0334 | 5-Day |
| tfidf (Baseline) | 0.6207 | (0.5599, 0.6796) | 0.0306 | 5-Day |
| finance_tfidf (Baseline) | 0.5406 | (0.4780, 0.6034) | 0.0325 | 5-Day |
| AttnMLPPoolClassifier (Transcript Only) | 0.4756 | (0.4066, 0.5424) | 0.0354 | 5-Day |
| AttnPoolTwoTower (Transcript + Finance) | 0.4821 | (0.4259, 0.5419) | 0.0298 | 5-Day |
| MeanPoolClassifier (Transcript Only) | 0.5247 | (0.4600, 0.5945) | 0.0341 | 5-Day |


### Key Findings
- NLP features (FinBERT embeddings) provide some predictive signal, especially for longer horizons (5-day).
- Combining financial features with text embeddings did not improve performance, possibly due to financial features being weak predictors alone.
- Attention pooling slightly outperformed mean pooling, indicating certain transcript sections may be more informative.
- Overall AUCs are modest, suggesting limited predictability of short-term price moves from earnings calls.
- Best baseline model TF-IDF + Logistic Regression outperformed neural models, indicating simpler models may suffice.

## Limitations
- Bootstrap confidence intervals assume IID samples (violated in financial time series - consider block bootstrap)
- Limited sample size and market conditions may affect generalizability

## Future Work
- Incorporate sentiment scores
- Test on different market periods (bull vs bear)
- Extend to longer time horizons
- Add technical indicators
- Try regression instead of classification

## Setup & Usage
Installation instructions:
How to run the code:
1. Clone the repository
2. Install dependencies from requirements.txt
3. Download the Motley Fool transcripts and S&P 500 data from Kaggle and provide the paths in a `.env` file as shown in `.env.example`.
4. Order of execution (alternatively use the Jupyter notebook `run_me.ipynb`): 
    - Run data_cleaning_util.py to preprocess transcripts and financial data.
    - Run the finbert_embed_utils.py to generate FinBERT embeddings. (Alternatively use the python notebook modal_create_bert_embeddings_cache.ipynb in modal to create the embeddings and cache them on disk.)
    - Run the finbert_model_utils.py to train and evaluate models.
5. Run the Jupyter notebook `run_me.ipynb` to execute the experiments and view results.

## References
Huang, Allen H., Hui Wang, and Yi Yang. "FinBERT: A Large Language Model for Extracting Information from Financial Text." Contemporary Accounting Research (2022).
