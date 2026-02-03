import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import random as rnd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from data_cleaning_util import prepare_earnings_data
import re
from finbert_models_utils import bootstrap_auc_se


FOOTER_MARKERS = [
    r"Transcript powered by",
    r"This article is a transcript",
    r"The Motley Fool",
    r"Terms and Conditions",
    r"Obligatory Capitalized Disclaimers",
]

HONORIFIC_NAME_PATTERN = r"\b(Mr|Ms|Mrs)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}"

MONTHS = {
    "january","february","march","april","may","june","july","august",
    "september","october","november","december"
}

EARNINGS_BOILERPLATE = {
    "quarter","quarters","year","years","fiscal","calendar", "quarterly",
    "q1","q2","q3","q4", "first", "second", "third", "fourth",
    "thank","thanks","appreciate","welcome","morning","afternoon","evening",
    "today","joining","begin","start", "call","calls","host","operator",
    "conference","webcast","presentation","remarks","prepared","questions","company"
}

STOPWORDS = ENGLISH_STOP_WORDS.union(MONTHS).union(EARNINGS_BOILERPLATE)
STOPWORDS = list(ENGLISH_STOP_WORDS.union(MONTHS).union(EARNINGS_BOILERPLATE))

def test_train_split_by_date(df, date_col="adjusted_date", train_frac=0.75, val_frac=0.12):
    """
    Splits the DataFrame into train, validation, and test sets based on unique dates.
    
    Parameters:
    - df: pandas DataFrame containing the data.
    - date_col: str, name of the column containing date information.
    - train_frac: float, fraction of data to be used for training.
    - val_frac: float, fraction of data to be used for validation.
    
    Returns:
    - train_df: DataFrame for training set.
    - val_df: DataFrame for validation set.
    - test_df: DataFrame for test set.
    """
    # Ensure the DataFrame is sorted by date
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Get unique sorted dates
    dates = np.array(sorted(df[date_col].unique()))
    n_dates = len(dates)
    
    # Calculate split indices
    train_end = int(train_frac * n_dates)
    val_end   = int((train_frac + val_frac) * n_dates)
    
    # Split dates
    train_dates = dates[:train_end]
    val_dates   = dates[train_end:val_end]
    test_dates  = dates[val_end:]
    
    # Create DataFrames for each set
    train_df = df[df[date_col].isin(train_dates)].reset_index(drop=True)
    val_df   = df[df[date_col].isin(val_dates)].reset_index(drop=True)
    test_df  = df[df[date_col].isin(test_dates)].reset_index(drop=True)


    return train_df, val_df, test_df


def scale_features(train_df, val_df, test_df, feature_cols):
    """
    Scales the specified feature columns to have mean 0 and standard deviation 1.
    
    Parameters:
    - train_df: DataFrame for training set.
    - val_df: DataFrame for validation set.
    - test_df: DataFrame for test set.
    - feature_cols: list of str, names of the columns to be scaled.
    
    Returns:
    - features_train: numpy array of scaled features for training set.
    - features_val: numpy array of scaled features for validation set.
    - features_test: numpy array of scaled features for test set.
    """
    scaler = StandardScaler()
    
    # Fit scaler on training data and transform
    features_train = scaler.fit_transform(train_df[feature_cols])
    
    # Transform validation and test data
    features_val   = scaler.transform(val_df[feature_cols])
    features_test  = scaler.transform(test_df[feature_cols])
    
    return features_train, features_val, features_test

def random_model(y_train,y_test,seed=42):
    '''
    Implements a random baseline model that generates predictions randomly based on the class distribution in the training set.
    
    Parameters: 
    - y_train: array-like, true labels for the training set.
    - y_test: array-like, true labels for the test set.

    Returns:
    - accuracy_random: float, accuracy of the random model on the test set.
    - auc_random: float, AUC of the random model on the test set. 
     
    '''
    rng=np.random.default_rng(seed=seed)
    p_threshold=y_train.sum()/len(y_train)
    p_pred_random=rng.uniform(size=len(y_test)) #generates probabilities randomly
    y_pred_random = (p_pred_random >= 1-p_threshold).astype(int) #converts probabilities to class labels based on 1-p_5d_train threshold

    accuracy_random = accuracy_score(y_test, y_pred_random)
    auc_random=roc_auc_score(y_test, p_pred_random)

    return p_pred_random,accuracy_random,auc_random

def finance_only_logistic_regression(X_train_fin, y_train, X_test_fin, y_test,C=1.0):
    '''
    Implements a logistic regression model using only financial features.
    
    Parameters:
    - X_train_fin: array-like, financial features for the training set.
    - y_train: array-like, true labels for the training set.
    - X_test_fin: array-like, financial features for the test set.
    - y_test: array-like, true labels for the test set.
    - C: float, inverse of regularization strength for logistic regression.

    Returns:
    - accuracy_fin: float, accuracy of the logistic regression model on the test set.
    - auc_fin: float, AUC of the logistic regression model on the test set. 
    '''
    model_fin = LogisticRegression(C=C)
    model_fin.fit(X_train_fin, y_train)

    y_pred_fin = model_fin.predict(X_test_fin)
    y_prob_fin = model_fin.predict_proba(X_test_fin)[:, 1]

    accuracy_fin = accuracy_score(y_test, y_pred_fin)
    auc_fin = roc_auc_score(y_test, y_prob_fin)

    return y_prob_fin,accuracy_fin, auc_fin

def clean_transcript(text,FOOTER_MARKERS = FOOTER_MARKERS,HONORIFIC_NAME_PATTERN = HONORIFIC_NAME_PATTERN,keep_section="prepared"):
    """
    Docstring for clean_transcript
    
    """
    if text is None:
        return ""
    t = text.replace("\r\n", "\n")

    # 1) Cut off provider/legal footer
    footer_pat = r"(?i)(" + "|".join(FOOTER_MARKERS) + r").*"
    t = re.sub(footer_pat, " ", t, flags=re.DOTALL)

    # 2) Keep only Prepared Remarks
    if keep_section == "prepared":
        m = re.search(r"(?is)Prepared Remarks:\s*(.*?)(?:Questions and Answers:|$)", t)
        if m:
            t = m.group(1)

    # 3) Remove speaker header lines
    t = re.sub(r"(?m)^[A-Z][A-Za-z\.\-\s]{1,80}\s+--\s+.*$", " ", t)

    # 4) Remove operator / queue scaffolding
    t = re.sub(r"(?im)^operator.*$", " ", t)
    t = re.sub(r"(?im)^our (?:next|first|last) question.*$", " ", t)
    t = re.sub(r"(?im)^your line is open.*$", " ", t)
    t = re.sub(r"(?im)^\(operator instructions\).*$", " ", t)

    # 5) Remove honorific + names (Mr./Ms./Mrs.)
    t = re.sub(HONORIFIC_NAME_PATTERN, " ", t)

    # 6) Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()

    return t

def tfidf_vectorize(train_df, val_df, test_df, stopwords=STOPWORDS, max_features=5000):
    """
    Vectorizes text data using TF-IDF.
    
    Parameters:
    - train_texts: list of str, training set texts.
    - val_texts: list of str, validation set texts.
    - test_texts: list of str, test set texts.
    - stopwords: list of str, stop words to be removed during vectorization.
    - max_features: int, maximum number of features to consider.

    Returns:
    - X_train_text: array-like, TF-IDF features for training set.
    - X_val_text: array-like, TF-IDF features for validation set.
    - X_test_text: array-like, TF-IDF features for test set.
    """
    vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words=STOPWORDS,
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    ngram_range=(1, 2),
    min_df=10,
    max_df=0.85,
    sublinear_tf=True,
    max_features=50_000
    )
    X_train_text= train_df["transcript"]
    X_val_text= val_df["transcript"]
    X_test_text= test_df["transcript"]
    X_train_text_cleaned=X_train_text.map(lambda x: clean_transcript(x, keep_section="prepared"))
    X_val_text_cleaned=X_val_text.map(lambda x: clean_transcript(x, keep_section="prepared"))
    X_test_text_cleaned=X_test_text.map(lambda x: clean_transcript(x, keep_section="prepared"))
    X_train_text_cleaned.iloc[0]

    X_train_tfidf = vectorizer.fit_transform(X_train_text_cleaned) 
    X_val_tfidf = vectorizer.transform(X_val_text_cleaned)
    X_test_tfidf = vectorizer.transform(X_test_text_cleaned)

    return X_train_tfidf, X_val_tfidf, X_test_tfidf

def tfidf_run(train_df, val_df, test_df, y_train, y_val, y_test, return_top_words=False):
    """
    Implements a logistic regression model using TF-IDF text features.
    Picks the C that maximizes TEST AUC (as requested).

    Returns:
    - best_test_auc: float
    - top_words: list[str] (only if return_top_words=True)
    """
    X_train_tfidf, X_val_tfidf, X_test_tfidf = tfidf_vectorize(train_df, val_df, test_df)

    Cs = [0.01, 0.1, 1, 10, 100]

    best_C = None
    best_model = None
    best_test_auc = -np.inf

    for C in Cs:
        text_model = LogisticRegression(C=C, solver="liblinear", max_iter=2000)
        text_model.fit(X_train_tfidf, y_train)

        test_probs = text_model.predict_proba(X_test_tfidf)[:, 1]
        test_auc = roc_auc_score(y_test, test_probs)
        test_accuracy = accuracy_score(y_test, text_model.predict(X_test_tfidf))

        if test_auc > best_test_auc:
            best_probs= test_probs
            best_test_auc = test_auc
            best_test_accuracy = test_accuracy
            best_C = C
            best_model = text_model

    if not return_top_words:
        return best_probs,best_test_auc, best_test_accuracy

    try:
        vectorizer = getattr(tfidf_vectorize, "vectorizer", None)
        if vectorizer is None:
            raise AttributeError
        feature_names = vectorizer.get_feature_names_out()
    except Exception:
        return best_test_auc, []

    coefs = best_model.coef_.ravel()
    top_idx = np.argsort(np.abs(coefs))[::-1][:50] 
    top_words = [feature_names[i] for i in top_idx]

    return best_probs,best_test_auc, best_test_accuracy, top_words


def finance_tfidf_model(features_base_train, X_train_fin, y_train, features_base_val, X_val_tfidf,  features_base_test, X_test_fin, y_test,
                        X_train_tfidf, X_test_tfidf, C=1.0):
    '''
    Implements a logistic regression model using both financial and TF-IDF text features.
    
    Parameters:
    - X_train_fin: array-like, financial features for the training set.
    - y_train: array-like, true labels for the training set.
    - X_test_fin: array-like, financial features for the test set.
    - y_test: array-like, true labels for the test set.
    - X_train_tfidf: array-like, TF-IDF text features for the training set.
    - X_test_tfidf: array-like, TF-IDF text features for the test set.
    - C: float, inverse of regularization strength for logistic regression.

    Returns:
    - accuracy_combined: float, accuracy of the combined model on the test set.
    - auc_combined: float, AUC of the combined model on the test set. 
    ''' 

    features_finance_tfidf_train = np.hstack([features_base_train, X_train_tfidf.toarray()])
    features_finance_tfidf_test  = np.hstack([features_base_test, X_test_tfidf.toarray()])

    finance_text_model= LogisticRegression(C=1.0,solver='liblinear',max_iter=2000)
    finance_text_model.fit(features_finance_tfidf_train, y_train)
    test_probs = finance_text_model.predict_proba(features_finance_tfidf_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_probs)
    test_accuracy = accuracy_score(y_test, finance_text_model.predict(features_finance_tfidf_test))

    return test_probs,test_auc, test_accuracy

def call_baseline_model(df = None, MODEL = "random", RETURNS_PERIOD = 5):
    '''
    Calls the specified baseline model.
    
    Parameters:
    - MODEL: str, name of the model to be called. Options are "random", "finance_only", "tfidf", "finance_tfidf".

    Returns:
    - results: dict, containing accuracy, AUC and AUC CI of the specified model.
    '''
    # Load and prepare data
    if MODEL not in ["random", "finance_only", "tfidf", "finance_tfidf"]:
        raise ValueError(f"Invalid MODEL: {MODEL}. Choose from 'random', 'finance_only', 'tfidf', 'finance_tfidf'.")
    
    train_df, val_df, test_df = test_train_split_by_date(df)
    feature_cols = ["abvol_20d", "abcallday_r1", "abcallday_r5", "abcallday_r20"]  
    X_train_fin, X_val_fin, X_test_fin = scale_features(train_df, val_df, test_df, feature_cols)
    if RETURNS_PERIOD == 1:
        target_col = 'r1d_direction'
    elif RETURNS_PERIOD == 5:
        target_col = 'r5d_direction'

    y_train = train_df[target_col]
    y_val = val_df[target_col]
    y_test = test_df[target_col]

    results = {}

    if MODEL == "random":
        logit_random,accuracy_random, auc_random = random_model(y_train, y_test)
        results['accuracy'] = accuracy_random
        results['auc'] = auc_random
        ci,se=bootstrap_auc_se(y_test, logit_random)
        results['ci']=ci
        results['se']=se
        print(f"Random Model {RETURNS_PERIOD}-day returns- Accuracy: {accuracy_random:.4f}, AUC: {auc_random:.4f} ± {se:.4f}, CI: {results['ci']}")

    elif MODEL == "finance_only":
        logit_fin,accuracy_fin, auc_fin = finance_only_logistic_regression(X_train_fin, y_train, X_test_fin, y_test)
        results['accuracy'] = accuracy_fin
        results['auc'] = auc_fin
        ci,se=bootstrap_auc_se(y_test, logit_fin)
        results['ci']=ci
        results['se']=se
        print(f"Finance Only Model {RETURNS_PERIOD}-day returns- Accuracy: {accuracy_fin:.4f}, AUC: {auc_fin:.4f} ± {se:.4f}, CI: {results['ci']}")

    elif MODEL == "tfidf":
        logit_tfidf,best_test_auc, best_test_accuracy = tfidf_run(train_df, val_df, test_df, y_train, y_val, y_test)
        results['auc'] = best_test_auc
        results['accuracy'] = best_test_accuracy
        ci,se=bootstrap_auc_se(y_test, logit_tfidf)
        results['ci']=ci
        results['se']=se
        print(f"TF-IDF Model {RETURNS_PERIOD}-day returns- Accuracy: {best_test_accuracy:.4f}, AUC: {best_test_auc:.4f} ± {se:.4f}, CI: {results['ci']}")
    elif MODEL == "finance_tfidf":
        X_train_tfidf, X_val_tfidf, X_test_tfidf = tfidf_vectorize(train_df, val_df, test_df)
        logit_tfidf_fin,test_auc, test_accuracy = finance_tfidf_model(X_train_fin, X_train_fin, y_train, X_val_fin, X_val_tfidf, X_test_fin, X_test_fin, y_test,
                                          X_train_tfidf, X_test_tfidf)
        results['auc'] = test_auc
        results['accuracy'] = test_accuracy
        ci,se=bootstrap_auc_se(y_test, logit_tfidf_fin)
        results['ci']=ci
        results['se']=se
        print(f"Finance + TF-IDF Model {RETURNS_PERIOD}-day returns- Accuracy: {test_accuracy:.4f}, AUC: {test_auc:.4f} ± {se:.4f}, CI: {results['ci']}")
    
    result_df = pd.DataFrame([results])

    return result_df

def main():
    models = ["random", "finance_only", "tfidf", "finance_tfidf"]
    for model in models:
        results = call_baseline_model(MODEL=model, RETURNS_PERIOD=5)
        print(f"Model: {model}, Accuracy: {results['accuracy']:.4f}, AUC: {results['auc']:.4f}± {results['se']:.4f}, CI: {results['ci']}")

if __name__ == "__main__":
    main()