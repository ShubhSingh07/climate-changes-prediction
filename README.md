# Climate Change Prediction Project

## Overview
This project aims to analyze and predict climate change indicators such as temperature anomalies, CO2 levels, and sea level changes using machine learning models. The project leverages data preprocessing, feature engineering, and predictive modeling techniques to forecast future climate trends.

## Project Structure

## Key Features
1. **Data Preprocessing**:
   - Handling missing values using forward fill and backward fill strategies.
   - Scaling features using `StandardScaler` for normalization.

2. **Feature Engineering**:
   - Rolling averages for temperature anomalies.
   - Lag features for CO2 levels.

3. **Predictive Modeling**:
   - Linear Regression and Random Forest models for temperature anomaly prediction.
   - Evaluation metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.

4. **Future Predictions**:
   - Forecasting temperature anomalies for the years 2025–2035.
   - Visualization of projected trends.

## Use Cases
1. **Climate Trend Analysis**:
   - Understand historical climate trends using temperature anomalies, CO2 levels, and sea level changes.

2. **Climate Forecasting**:
   - Predict future climate indicators to assist policymakers and researchers in planning mitigation strategies.

3. **Feature Engineering Insights**:
   - Explore the impact of rolling averages and lag features on predictive accuracy.

4. **Model Evaluation**:
   - Compare the performance of Linear Regression and Random Forest models.

## Outputs
1. **Preprocessed Data**:
   - Cleaned and scaled dataset ready for modeling.

2. **Model Evaluation**:
   - Linear Regression and Random Forest models evaluated using MAE, MSE, and R-squared metrics.
   - Residual plots and actual vs. predicted scatter plots for model diagnostics.

3. **Future Predictions**:
   - Projected temperature anomalies for 2025–2035 visualized as a line plot.

## How to Run
1. Clone the repository and ensure all dependencies are installed.
2. Open `adjusting.ipynb` in Jupyter Notebook or VS Code.
3. Run the notebook cells sequentially to preprocess data, train models, and generate predictions.

## Dependencies
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Visualization
- Residual plots for model diagnostics.
- Scatter plots for actual vs. predicted values.
- Line plots for future temperature anomaly projections.

## Future Scope
- Incorporate additional climate indicators such as precipitation and wind patterns.
- Explore advanced machine learning models like Gradient Boosting and Neural Networks.
- Integrate real-time data for continuous forecasting.

  # Climate Change Prediction Project

## Key Outputs and Visualizations


**Code:**
```python
# Handling missing values using forward fill strategy
climate_df[['Temperature_Anomaly', 'CO2', 'Sea_Level_mm']] = climate_df[['Temperature_Anomaly', 'CO2', 'Sea_Level_mm']].fillna(method='ffill')
print(climate_df)
# Evaluate the Linear Regression model
print("\nLinear Regression Evaluation")
evaluate_model(lr_model, X_test, y_test)

# Evaluate the Random Forest model
print("\nRandom Forest Evaluation")
evaluate_model(rf_model, X_test, y_test)
# Predict future temperature anomalies
plt.figure(figsize=(10, 5))
plt.plot(future_years['Year'], future_preds, marker='o', label='Projected Temperature Anomaly')
plt.title("Projected Temperature Anomaly (2025–2035)")
plt.xlabel("Year")
plt.ylabel("Temperature Anomaly")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# Rolling averages and lag features
climate_df['Temp_Rolling_Mean'] = climate_df['Temperature_Anomaly'].rolling(window=3).mean()
climate_df['CO2_Lag_1'] = climate_df['CO2'].shift(1)
climate_df.fillna(method='bfill', inplace=True)
print(climate_df)
# Scaling features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(climate_df[['CO2', 'Sea_Level_mm']])
climate_df[['CO2_scaled', 'Sea_Level_scaled']] = scaled_features
print(climate_df[['CO2_scaled', 'Sea_Level_scaled']])
Project Directory:
- images/
  - residual_plot.png
  - actual_vs_predicted.png
  - projected_temperature_anomaly.png
** OUTPUT:
![Screenshot 2025-06-02 140148](https://github.com/user-attachments/assets/a6cfa187-51a9-4a56-9f28-b84911dc5924)
![image](https://github.com/user-attachments/assets/3366a968-80b1-4981-a206-6bd356062161)
![image](https://github.com/user-attachments/assets/a2ea0b56-0401-42fc-9d44-3fa00a9eb785)
![image](https://github.com/user-attachments/assets/c7a6cc22-6b98-4a0d-9b49-4f9777d897ae)

![image](https://github.com/user-attachments/assets/bc7d34bb-9737-4eb4-94c4-18c349dadb73)




