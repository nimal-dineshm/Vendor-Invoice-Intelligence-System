import os
import joblib
import pandas as pd

def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(current_dir, "..", "models", "predict_freight_model.pkl")
    
    return joblib.load(model_path)

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