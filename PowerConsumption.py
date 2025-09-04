import pandas as pd
import os 
import numpy as np
import MyFunction

##### File importation
file = "household_power_consumption.txt"
path = os.path.join(os.getcwd(),file)

df = pd.read_csv(path, sep =';',na_values=['?']) #, parse_dates=[["Date", "Time"]])

df['Date_Time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df = df.drop(['Date', 'Time'], axis=1)
df.set_index('Date_Time', inplace=True)


## Checking Missing Values
print("\n####### Checking missing values #######")
for i in df.keys():
    if df[i].isna().sum() != 0:
        print("Missing ", df[i].isna().sum(), "Values in column " , i)

#df = df.resample("h").mean()

print("\n")

## Visualisation des données 
import matplotlib.pyplot as plt


df = MyFunction.Resample(df, "h")

df = df.interpolate(method='linear')
df['rolling_mean'] = df['Global_active_power'].rolling(window=24).mean()
df['last'] = df['Global_active_power'].shift(1)
df['Second_Last'] = df['Global_active_power'].shift(2)

df = df.dropna()

df['hour'] = np.sin(2*np.pi*df.index.hour/24)
df['day_of_week'] = np.sin(2*np.pi*df.index.dayofweek/7) 
df['month'] = np.sin(2*np.pi*df.index.month/12)
df = df.dropna(axis=1, how='all')

import seaborn as sns
correlation_matrix = df.corr(method='pearson')
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, 
            annot=True,        # Show the correlation values on the heatmap
            cmap='coolwarm',   # Choose a color palette
            fmt=".2f",         # Format the values to 2 decimal places
            linewidths=.5,
            mask = np.triu(np.ones_like(correlation_matrix)))      # Add lines between cells


print("\n######################### Data Description #########################")
print(df.describe())

X = df.drop(['hour', 'Voltage'], axis=1)
y = df['Global_active_power'].shift(-1)

## Split train/test
split_index = int(len(df) * 0.8)

X_train = X[:split_index]
y_train = y[:split_index]

X_test = X[split_index:-1]
y_test = y[split_index:-1]


X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler_x, scaler_y = MyFunction.Scaling(X_train, y_train, X_test, y_test, "Regression")

####### Modele Regression lineaire 
from sklearn.linear_model import LinearRegression

Regression = LinearRegression()
Regression.fit(X_train_scaled, y_train_scaled)
MyFunction.AffichageRes(Regression, X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_y)

####### Modele RandomForest
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(X_train_scaled, y_train_scaled)
MyFunction.AffichageRes(forest_reg, X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_y)

####### Modele LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping


n_timesteps = 24

X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler_x, scaler_y = MyFunction.Scaling(X_train, y_train, X_test, y_test, "LSTM")

#X_train_scaled = X_train_scaled.drop(['day_of_week'],axis=1)
#X_test_scaled = X_test_scaled.drop(['day_of_week'],axis=1)
             
X_np, y_np = MyFunction.LSTM_preprocess(X_train_scaled, y_train_scaled, n_timesteps)
X_np_test, y_np_test = MyFunction.LSTM_preprocess(X_test_scaled, y_test_scaled, n_timesteps)

lstm_units = 128 # 256

model_LSTM = Sequential(name = 'LSTM_Model', layers = [
        Input(shape=(n_timesteps, len(X_train_scaled.keys()) )),
        LSTM(lstm_units, activation='relu', return_sequences=True),
        LSTM(lstm_units//2, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
model_LSTM.compile(optimizer='adam', loss='mse')
model_LSTM.summary()
print("\n########## Training ##########")

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
history = model_LSTM.fit(X_np, y_np, epochs=50, validation_data = (X_np_test,y_np_test), callbacks=[early_stopping])

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

MyFunction.AffichageResLSTM(model_LSTM, X_np, X_np_test, y_np, y_np_test, scaler_y)
ghj

############## GRU 
gru_units = 128 # 256

model_LSTM = Sequential(name = 'GRU_Model', layers = [
        Input(shape=(n_timesteps, len(X_train_scaled.keys()) )),
        GRU(gru_units, activation='relu', return_sequences=True),
        GRU(gru_units//2, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
model_LSTM.compile(optimizer='adam', loss='mse')
model_LSTM.summary()
print("\n########## Training ##########")

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
history = model_LSTM.fit(X_np, y_np, epochs=50, validation_data = (X_np_test,y_np_test), callbacks=[early_stopping])

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

MyFunction.AffichageResLSTM(model_LSTM, X_np, X_np_test, y_np, y_np_test, scaler_y)
