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
-where n-day return is defined as $\text{return}_{n}=(\text{Price}_{t+n}-\text{Price}_t)/Price_t$
                   
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
7. SVM + FinBERT
8. SVM + FinBERT + financial features

### Validation
- Time-based train/validation/test splits (prevents lookahead bias)
- Evaluation metric: AUC-ROC with 95% confidence intervals (bootstrap)

## Results


### Key Findings
- 
## Limitations
- Bootstrap confidence intervals assume IID samples (violated in financial time series - consider block bootstrap)
- 
## Future Work
- Incorporate sentiment scores
- Test on different market periods (bull vs bear)
- Extend to longer time horizons
- Add technical indicators
- Try regression instead of classification

## Setup & Usage
Installation instructions:
How to run the code:

## References
Huang, Allen H., Hui Wang, and Yi Yang. "FinBERT: A Large Language Model for Extracting Information from Financial Text." Contemporary Accounting Research (2022).
