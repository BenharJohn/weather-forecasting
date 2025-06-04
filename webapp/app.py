from flask import Flask, jsonify
import pandas as pd
from pathlib import Path

# Import training function without triggering heavy computation on import
from Final_Code import train_and_predict

app = Flask(__name__)

def load_forecast():
    forecast_file = Path(__file__).resolve().parent / "forecast.csv"
    if forecast_file.exists():
        return pd.read_csv(forecast_file)
    # If forecast not found, generate and save it
    forecast_df = train_and_predict(save_path=forecast_file)
    return forecast_df

@app.route("/forecast")
def forecast_endpoint():
    forecast_df = load_forecast()
    return forecast_df.to_json(orient="records")

if __name__ == "__main__":
    app.run(debug=True)
