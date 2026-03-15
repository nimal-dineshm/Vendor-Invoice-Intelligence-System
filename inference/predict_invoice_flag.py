import joblib
import pandas as pd
import os

# Paths to your saved artifacts
MODEL_PATH = "models/predict_flag_invoice.pkl"
SCALER_PATH = "models/scaler.pkl"

def load_artifacts():
    """Load both the trained model and the scaler."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices using scaled features.
    """
    # 1. Load artifacts
    model, scaler = load_artifacts()
    
    # 2. Convert input to DataFrame
    input_df = pd.DataFrame(input_data)
    
    # 3. Define features in the EXACT order used during training
    features = [
        "invoice_quantity", 
        "invoice_dollars", 
        "Freight", 
        "total_item_quantity", 
        "total_item_dollars"
    ]
    
    # 4. SCALE the data
    # .values is used to avoid the "feature names" warning
    X_scaled = scaler.transform(input_df[features])
    
    # 5. Predict
    # We use scaled data for the model, then attach results to the original df
    input_df['Predicted_Flag'] = model.predict(X_scaled)
    
    return input_df

if __name__ == "__main__":
    # Example inference run (local testing)
    # Row 0: Large mismatch (5000 vs 7500) -> Should flag as 1
    # Row 3: Small mismatch (100 vs 150) -> Should flag as 1
    sample_data = {
        "invoice_quantity": [100, 50, 10, 5],
        "invoice_dollars": [5000, 2000, 500, 100],
        "Freight": [200, 100, 50, 20],
        "total_item_quantity": [150, 80, 20, 10],
        "total_item_dollars": [7500, 3000, 800, 150]
    }

    prediction = predict_invoice_flag(sample_data)
    print(prediction)