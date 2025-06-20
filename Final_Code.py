"""Weather forecasting models and utilities."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import geopandas as gpd
from pathlib import Path


def train_and_predict(data_path: str = "GlobalWeatherRepository.csv", save_path: str | None = None) -> pd.DataFrame:
    """Train forecasting models and return the ensemble forecast.

    Parameters
    ----------
    data_path:
        Path to the input weather CSV file.
    save_path:
        If given, the forecast will be written to this file.

    Returns
    -------
    pandas.DataFrame
        Ensemble forecast indexed by date.
    """
    # -------------------- Load and Preprocess Data --------------------
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: File '{data_path}' not found. Please ensure it is in the correct location.")
        return pd.DataFrame()

    # Convert 'last_updated' to datetime and extract year and month
    data["last_updated"] = pd.to_datetime(data["last_updated"])
    data["year"] = data["last_updated"].dt.year
    data["month"] = data["last_updated"].dt.month

    # Aggregate daily averages for all cities
    daily_avg = data.groupby(data["last_updated"].dt.date).agg({
        "temperature_celsius": "mean",
        "precip_mm": "mean",
    })

    # Handle missing dates
    date_range = pd.date_range(start=daily_avg.index.min(), end=daily_avg.index.max(), freq="D")
    daily_avg = daily_avg.reindex(date_range)

    # Fill missing values
    daily_avg["temperature_celsius"] = daily_avg["temperature_celsius"].interpolate()
    daily_avg["precip_mm"] = daily_avg["precip_mm"].fillna(0)

    # Detect anomalies (values below the 1st percentile)
    anomalies = daily_avg[daily_avg["temperature_celsius"] < daily_avg["temperature_celsius"].quantile(0.01)]
    print("Anomalies detected:")
    print(anomalies)

    # Handle anomalies by imputing them using linear interpolation
    daily_avg.loc[anomalies.index, "temperature_celsius"] = daily_avg["temperature_celsius"].interpolate()

    # Normalize data for LSTM
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(daily_avg[["temperature_celsius"]])

    # Split data into train/test sets (80% train, 20% test)
    split_index = int(len(daily_avg) * 0.8)
    train = daily_avg.iloc[:split_index]
    test = daily_avg.iloc[split_index:]

    # -------------------- SARIMA MODEL --------------------
    order = (2, 1, 2)
    seasonal_order = (1, 1, 1, 30)  # Monthly seasonality

    sarima_model = SARIMAX(
        train["temperature_celsius"],
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results_sarima = sarima_model.fit(disp=False)

    forecast_sarima = results_sarima.get_forecast(steps=len(test))
    forecast_mean_sarima = forecast_sarima.predicted_mean

    mae_sarima = mean_absolute_error(test["temperature_celsius"], forecast_mean_sarima)
    rmse_sarima = np.sqrt(mean_squared_error(test["temperature_celsius"], forecast_mean_sarima))
    print(f"SARIMA MAE: {mae_sarima:.2f}, RMSE: {rmse_sarima:.2f}")

    # -------------------- LSTM MODEL --------------------
    lookback = 30

    def create_sequences(data: np.ndarray, lookback: int = 30) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i : i + lookback])
            y.append(data[i + lookback])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data, lookback)
    X_train_lstm, X_test_lstm = X[: split_index - lookback], X[split_index - lookback :]
    y_train_lstm, y_test_lstm = y[: split_index - lookback], y[split_index - lookback :]

    lstm_model = Sequential(
        [LSTM(50, activation="relu", input_shape=(lookback, 1), return_sequences=True), LSTM(50, activation="relu"), Dense(1)]
    )
    lstm_model.compile(optimizer="adam", loss="mse")
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=32, validation_data=(X_test_lstm, y_test_lstm))

    y_pred_scaled_lstm = lstm_model.predict(X_test_lstm)
    y_pred_lstm = scaler.inverse_transform(y_pred_scaled_lstm)

    mae_lstm = mean_absolute_error(y_test_lstm.flatten(), y_pred_scaled_lstm.flatten())
    rmse_lstm = np.sqrt(mean_squared_error(y_test_lstm.flatten(), y_pred_scaled_lstm.flatten()))
    print(f"LSTM MAE: {mae_lstm:.2f}, RMSE: {rmse_lstm:.2f}")

    # -------------------- ENSEMBLE MODEL --------------------
    weights = [0.2, 0.8]  # Assign higher weight to LSTM
    ensemble_forecast = weights[0] * forecast_mean_sarima.values + weights[1] * y_pred_lstm.flatten()

    mae_ensemble = mean_absolute_error(test["temperature_celsius"], ensemble_forecast)
    rmse_ensemble = np.sqrt(mean_squared_error(test["temperature_celsius"], ensemble_forecast))
    print(f"Optimized Ensemble MAE: {mae_ensemble:.2f}, RMSE: {rmse_ensemble:.2f}")

    forecast_df = pd.DataFrame({"date": test.index[: len(ensemble_forecast)], "ensemble_temperature_celsius": ensemble_forecast})

    # Save forecast if requested
    if save_path:
        forecast_file = Path(save_path)
        forecast_df.to_csv(forecast_file, index=False)
        print(f"Forecast saved to {forecast_file}")

    # -------------------- Additional Analyses --------------------
    climate_trends = data.groupby("year").agg({"temperature_celsius": "mean", "precip_mm": "mean"}).reset_index()
    plt.figure(figsize=(12, 6))
    plt.plot(climate_trends["year"], climate_trends["temperature_celsius"], label="Average Temperature (°C)", color="red")
    plt.plot(climate_trends["year"], climate_trends["precip_mm"], label="Average Precipitation (mm)", color="blue")
    plt.title("Long-Term Climate Trends")
    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

    air_quality_columns = [
        "temperature_celsius",
        "precip_mm",
        "humidity",
        "wind_mph",
        "air_quality_Carbon_Monoxide",
        "air_quality_Ozone",
        "air_quality_Nitrogen_dioxide",
        "air_quality_Sulphur_dioxide",
        "air_quality_PM2.5",
        "air_quality_PM10",
    ]
    data_corr = data[air_quality_columns].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(data_corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Matrix of Weather Parameters and Air Quality")
    plt.show()

    features = ["temperature_celsius", "humidity", "wind_mph", "precip_mm"]
    target = "air_quality_PM2.5"
    X_features = data[features]
    y_target = data[target]
    rf_model = RandomForestRegressor()
    rf_model.fit(X_features, y_target)
    importance_scores_rf = rf_model.feature_importances_
    plt.figure(figsize=(8, 6))
    plt.bar(features, importance_scores_rf, color="skyblue")
    plt.title("Feature Importance for Air Quality (PM2.5)")
    plt.xlabel("Features")
    plt.ylabel("Importance Score")
    plt.grid(True)
    plt.show()

    return forecast_df


if __name__ == "__main__":
    train_and_predict()
