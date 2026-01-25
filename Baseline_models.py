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
    import numpy as np
    
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
    from sklearn.preprocessing import StandardScaler
    
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
    import random as rnd
    import numpy as np
    rng=np.random.default_rng(seed=seed)
    p_threshold=y_train.sum()/len(y_train)
    p_pred_random=rng.uniform(size=len(y_test)) #generates probabilities randomly
    y_pred_random = (p_pred_random >= 1-p_threshold).astype(int) #converts probabilities to class labels based on 1-p_5d_train threshold

    from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix
    accuracy_random = accuracy_score(y_test, y_pred_random)
    auc_random=roc_auc_score(y_test, p_pred_random)

    return accuracy_random,auc_random

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
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score,roc_auc_score

    model_fin = LogisticRegression(penalty='l2',C=C)
    model_fin.fit(X_train_fin, y_train)

    y_pred_fin = model_fin.predict(X_test_fin)
    y_prob_fin = model_fin.predict_proba(X_test_fin)[:, 1]

    accuracy_fin = accuracy_score(y_test, y_pred_fin)
    auc_fin = roc_auc_score(y_test, y_prob_fin)

    return accuracy_fin, auc_fin

def clean_transcript(text:str,FOOTER_MARKERS,HONORIFIC_NAME_PATTERN,keep_section="prepared")->str:
    """
    Docstring for clean_transcript
    
    """"


    import re
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

