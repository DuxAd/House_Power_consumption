#  Household Power Consumption Forecasting
## Overview   

This project aim to predict the next hour’s household power consumption (Global_active_power at t+1) using the data from the UCI Household Electric Power Consumption dataset which can be found here : https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
  
The dataset contains ~2.07 million observations of power consumption metrics.
Four models — Linear Regression, Random Forest, Long Short-Term Memory (LSTM) and GRU neural network — are implemented and compared to predict future power consumption.
The project demonstrates data preprocessing, feature engineering,
exploratory data analysis and model evaluation for a real-world time-series forecasting task.
  
Size: 2,075,259 minute-level observations (4 years - Dec 2006 to Nov 2010)  
Features: 7 features  
-  Global_active_power: Total active power consumption (kW, target variable)  
-  Sub_metering_1, Sub_metering_2, Sub_metering_3: Energy consumption (Wh) for kitchen, laundry, and climate control appliances, respectively  
-  Voltage, Global_intensity and Global_reactive_power

  Engineered features: 6 Engineered features
-  last (1-hour lag)
-  Second_Last (2-hour lag)
-  rolling_mean (24-hour mean)
-  hour
-  day_of_week
-  month
  
## Models
Four models were trained to predict Global_active_power at t+1:  
-  Linear Regression
-  Random Forest Regressor
-  LSTM Neural Network (Architecture: Two LSTM layers (128 and 64 units, ReLU activation), BatchNormalization, Dropout(0.2), and a dense output layer)
-  GRU Neural Network (Architecture: Two LSTM layers (256 and 128 units, ReLU activation), BatchNormalization, Dropout(0.2), and a dense output layer)

Evaluation Metrics    
-  Mean Squared Error (MSE): Measures prediction error in kW².
-  Mean Absolute Error (MAE): Average absolute error in kW.
-  Root Mean Squared Error (RMSE): Square root of MSE, in kW.
-  R² Score: Proportion of variance explained by the model.
-  Cross-Validation: Mean accuracy and standard deviation for Linear Regression and Random Forest.

## Results  
The models were evaluated on a test set (20% of data):
-	Linear Regression: MSE= 0.279, MAE= 0.383, RMSE= 0.528, R²= 0.47224, Cross-Validation Mean Accuracy= 0.499, Cross-Validation Std Dev= 0.0354
-	Random Forest: MSE= 0.251, MAE= 0.362, RMSE= 0.501, R²= 0.52476, Cross-Validation Mean Accuracy= 0.464, Cross-Validation Std Dev= 0.184
-	LSTM: MSE= 0.314, MAE= 0.399, RMSE= 0.560, R²= 0.40663, 
- GRU:  MSE= 0.306, MAE= 0.410, RMSE= 0.554, R²= 0.42058, 
