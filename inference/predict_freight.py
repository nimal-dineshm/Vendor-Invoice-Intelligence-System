import os
import joblib
import pandas as pd

def load_model():
    base_path = os.path.dirname(__file__)
    
    model_path = os.path.join(base_path, "..", "models", "predict_freight_model.pkl")
    
    with open(model_path, "rb") as f:
        return joblib.load(f)

def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendor invoices.

    Parameters
    ----------
    input_data : dict

    Returns
    -------
    pd.DataFrame with predicted freight cost
    """
    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df['Predicted_Freight'] = model.predict(input_df).round()
    return input_df

if __name__ == "__main__":

    sample_data = {
        "Dollars": [18500, 9000, 3000, 250]
    }

    prediction = predict_freight_cost(sample_data)
    print(prediction)