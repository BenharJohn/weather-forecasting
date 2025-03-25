# Weather Forecasting with PM Accelerator

## Project Overview

This project, developed under the auspices of PM Accelerator, focuses on creating a robust weather forecasting system by leveraging historical weather data and advanced predictive models. The primary objective is to analyze past weather patterns and implement various forecasting techniques to enhance prediction accuracy, thereby contributing to more informed decision-making in weather-dependent sectors.

## Repository Contents

- **Data Collection and Cleaning Scripts:** Python scripts used for gathering and preprocessing historical weather data.
- **Exploratory Data Analysis (EDA) Notebooks:** Jupyter notebooks containing visualizations and statistical analyses to uncover insights from the data.
- **Forecasting Models:** Implementations of SARIMA, LSTM, and an optimized ensemble model for weather prediction.
- **Evaluation Metrics:** Scripts to assess model performance using metrics like RMSE and MAE.
- **Visualization Outputs:** Graphs and plots illustrating model predictions and actual weather trends.

## Methodology

### 1. Data Collection and Cleaning

- **Dataset Overview:**  The dataset comprises historical weather parameters such as temperature, humidity, precipitation, and wind speed, collected over multiple years from reputable meteorological sources.

- **Data Cleaning Process:**
  - **Handling Missing Values:** Applied interpolation and mean imputation techniques to address gaps in temperature and humidity data.
  - **Removing Duplicates:** Identified and eliminated redundant records to maintain data integrity.
  - **Data Transformation:** Standardized units across all parameters for consistency (e.g., converting wind speed measurements to a uniform metric).
  - **Outlier Detection:** Employed statistical methods, including Z-score analysis, to detect and address anomalies in the data.

### 2. Exploratory Data Analysis (EDA)

- **Key Observations:**
  - Identified seasonal patterns in temperature fluctuations.
  - Observed correlations between humidity levels and precipitation.
  - Noted the influence of wind speed on temperature variations.

- **Visualizations:**
  - **Distribution Plots:** Illustrated the spread and central tendency of temperature and humidity data.
  - **Time Series Analysis:** Highlighted trends and seasonal variations over time.
  - **Correlation Matrix:** Mapped relationships between different weather parameters.
  - **Box Plots:** Utilized to detect outliers in temperature and precipitation datasets.

### 3. Forecasting Models

- **Baseline Model:**
  - **Simple Moving Average:** Established as a benchmark to gauge the performance of more complex models.

- **Advanced Models:**
  - **SARIMA (Seasonal AutoRegressive Integrated Moving Average):** Implemented to account for both trend and seasonality in time series data.
  - **LSTM (Long Short-Term Memory Networks):** Deployed to capture long-term dependencies and intricate patterns in sequential data.
  - **Optimized Ensemble Model:** Combined predictions from SARIMA and LSTM models to enhance overall forecasting accuracy.

- **Feature Engineering:**
  - **Lag Features:** Created to incorporate past weather conditions as predictive inputs.
  - **Rolling Statistics:** Generated moving averages to smooth out short-term fluctuations and highlight longer-term trends.
  - **One-Hot Encoding:** Applied to categorical variables, such as seasons, to facilitate model processing.

### 4. Model Evaluation

- **Performance Metrics:**
  - **Root Mean Squared Error (RMSE):** Assessed the average magnitude of prediction errors.
  - **Mean Absolute Error (MAE):** Measured the average absolute errors between predicted and actual values.

  | Model               | RMSE  | MAE  |
  |---------------------|-------|------|
  | SARIMA              | 3.55  | 2.94 |
  | LSTM                | 0.10  | 0.08 |
  | Optimized Ensemble  | 1.57  | 1.22 |

- **Anomaly Detection:**
  - Identified unusual weather patterns, particularly in January, indicating potential climate anomalies or data inconsistencies.
  - Detected extreme temperature and precipitation levels on specific dates, warranting further investigation.

### 5. Insights and Findings

- **Key Insights:**
  - The LSTM model demonstrated superior accuracy, achieving the lowest error rates among the models tested.
  - The ensemble approach effectively combined the strengths of individual models, resulting in improved forecasting performance.
  - Anomalies detected suggest potential shifts in climate patterns or highlight areas where data quality may need to be addressed.

- **Limitations & Future Improvements:**
  - Expanding the dataset to include additional parameters, such as atmospheric pressure and solar radiation, could enhance model performance.
  - Further hyperparameter tuning of deep learning models may yield better accuracy.
  - Integrating real-time weather data could facilitate continuous learning and more dynamic forecasting capabilities.

## Results

- **Model Performance:**
  - The LSTM model outperformed SARIMA and the baseline moving average model in terms of RMSE and MAE.
  - The optimized ensemble model provided a balanced approach, leveraging the strengths of both SARIMA and LSTM.

- **Visual Outputs:**
  - **Training Loss Curves:** Demonstrated the learning progression of the LSTM model over epochs.
  - **Predicted vs. Actual Trends:** Graphs illustrating the alignment between model forecasts and actual weather data.
  - **Anomaly Detection Plots:** Highlighted instances of significant deviations from expected weather patterns.
  - **Forecasting Visualizations:** Provided visual representations of future weather predictions with confidence intervals.

## Getting Started

### Prerequisites

- **Programming Language:** Python 3.x
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Statsmodels, TensorFlow, Keras
- **Platforms:** Jupyter Notebook or Google Colab
