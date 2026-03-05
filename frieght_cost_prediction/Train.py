import joblib
from pathlib import Path
from data_preprocessing import load_vendor_invoice_data, prepare_features, split_data
from modeling_evaluation import (
    train_linear regression,
    train_decision_tree,
    train_random_forest,
    evaluate model
)
def main():
    db_path = "data/inventory.db"
    model_dir=Path("models")
    model_dir.mkdir (exist_ok=True)
    # Load Load data
    df = load_vendor_invoice_data(db_path)
    # Prepare data
    X, y = prepare_features (df)
    X_train, X_test, y_train, y_test = split_data(x, y)
    # Train models
    lr_model = train_linear regression (X_train, y_train)
    dt_model = train_decision_tree (✗_train, y_train)
    rf_model = train_random_forest (X_train, y_train)
    # Evaluate models