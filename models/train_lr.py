import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

from config import settings
from utils.logger import get_logger

logger = get_logger("models.train_lr")

def train_logistic_regression(df, model_path=os.path.join(settings.MODEL_DIR, "lr_model.pkl"), scaler_path=os.path.join(settings.MODEL_DIR, "scaler_lr.pkl")):
    """
    Trains a Logistic Regression model strictly using an out-of-core time-based split.
    """
    logger.info("Initializing Logistic Regression Training...")
    
    # Feature columns (exclude targets)
    feature_cols = [c for c in df.columns if c not in ['Target_LR', 'Target_Price']]
    
    # Time-based split (80% train, 20% test)
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df['Target_LR']
    X_test = test_df[feature_cols]
    y_test = test_df['Target_LR']
    
    logger.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Training
    # Balanced classes assist heavily in imbalanced financial data sets
    model = LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced', C=0.1)
    model.fit(X_train_scaled, y_train)
    
    # Evaluation
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    
    logger.info("--- Logistic Regression Test Metrics ---")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    
    # Save artifacts
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Model saved to {model_path} and scaler to {scaler_path}")
    
    # Return predictions mapped to the test set indices for backtesting integration
    results = pd.DataFrame(index=test_df.index)
    results['LR_Prob'] = probs
    results['LR_Pred'] = preds
    results['Actual_Target'] = y_test
    
    return results, model, scaler

def predict_lr(df, model, scaler):
    """Generates predictions for incoming data."""
    feature_cols = [c for c in df.columns if c not in ['Target_LR', 'Target_Price']]
    X = scaler.transform(df[feature_cols])
    probs = model.predict_proba(X)[:, 1]
    return probs
