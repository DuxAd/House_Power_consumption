from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def Resample(df, time_sample):
    df = df.resample(time_sample[0]).mean()

    n_column = 3
    n_rows = np.ceil(len(df.columns)/3)
    for i in range(len(df.keys())):
        plt.subplot(n_column, int(n_rows), i+1)
        plt.title(df.keys()[i])
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.plot(df.index, df[df.keys()[i]])
    plt.tight_layout()
    plt.show() 

    plt.figure()
    plt.title('Global_active_power')
    plt.plot(df.index, df['Global_active_power']*1000/60, label = 'Global_active_power')
    plt.plot(df.index, df['Sub_metering_1'], label = 'Sub_metering_1')
    plt.plot(df.index, df['Sub_metering_2'], label = 'Sub_metering_2')
    plt.plot(df.index, df['Sub_metering_3'], label = 'Sub_metering_3')
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Wh")
    
    return df 

def Scaling(X_train, y_train, X_test, y_test, scaler):
    
    if scaler != "LSTM":
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
    else:
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
    X_train_numpy = scaler_x.fit_transform(X_train)
    y_train_numpy = scaler_y.fit_transform(y_train.values.reshape(-1,1))

    X_test_numpy = scaler_x.transform(X_test)
    y_test_numpy = scaler_y.transform(y_test.values.reshape(-1,1))

    X_train_scaled = pd.DataFrame(X_train_numpy, index = X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_numpy, index = X_test.index, columns=X_test.columns)

    y_train_scaled = pd.Series(y_train_numpy.flatten(), index = y_train.index)
    y_test_scaled = pd.Series(y_test_numpy.flatten(), index = y_test.index)

    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler_x, scaler_y

 
    
def LSTM_preprocess(X, y, n_timesteps):
    X_np = X.to_numpy()
    y_np = y.to_numpy()
    
    X_sequence = []
    for i in range(len(X)-n_timesteps):
        X_sequence.append(X_np[i:i+n_timesteps])
        
    return np.array(X_sequence), y_np[n_timesteps:] # np.array(y_sequence)
    
    
def AffichageRes(model, X_train, X_test, y_train, y_test, scaler):
    
    
    y_pred = model.predict(X_test)
    y_pred_scaled = scaler.inverse_transform(y_pred.reshape(-1,1)).flatten()
    
    y_test_scaled = scaler.inverse_transform(y_test.values.reshape(-1,1)).flatten()
    
    # Évaluer les performances du modèle
    mse = mean_squared_error(y_test_scaled, y_pred_scaled)
    mae = mean_absolute_error(y_test_scaled, y_pred_scaled)
    r2 = r2_score(y_test_scaled, y_pred_scaled)
    
    tscv = TimeSeriesSplit(n_splits=6)
    
    scores = cross_val_score(model, X_train, y_train, cv=tscv)

    print('\n--------------------------------------------')
    print("\nRésultats du modele", model.__class__.__name__)
    print(f'MSE : {mse:.3f}, MAE : {mae:.3f}, sqrt(MSE) :{np.sqrt(mse):.3f}')
    print(f'R² : {r2:.5f}')
    print('Score de validation Croisée')
    print(f'Score : {scores}')
    print(f"Accuracy moyenne : {scores.mean():.3f}")
    print(f"Écart-type : {scores.std():.3}")
    print('\n--------------------------------------------\n')

    Affichage_y_pred(y_test_scaled, y_pred_scaled, model)

def AffichageResLSTM(model, X_train, X_test, y_train, y_test, scaler):
    
    y_pred = model.predict(X_test)
    y_pred_scaled = scaler.inverse_transform(y_pred.reshape(-1,1)).flatten()
    
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

    # Évaluer les performances du modèle
    mse = mean_squared_error(y_test_scaled, y_pred_scaled)
    mae = mean_absolute_error(y_test_scaled, y_pred_scaled)
    r2 = r2_score(y_test_scaled, y_pred_scaled)
    
    print('\n--------------------------------------------')
    print("\nRésultats du modele", model.name)
    print(f'MSE : {mse:.3f}, MAE : {mae:.3f}, sqrt(MSE) :{np.sqrt(mse):.3f}')
    print(f'R² : {r2:.5f}')
    print('\n--------------------------------------------\n')
    
    Affichage_y_pred(y_test_scaled, y_pred_scaled, model)
    
def Affichage_y_pred(y_test_scaled, y_pred_scaled, model):
    ## This function plots the predicted value y_pred against the real value y_pred
    
    plt.figure()
    #plt.plot(y_test_sorted.reshape(-1,1), label='Valeurs réelles', marker='o', linestyle='', markersize=1)
    plt.plot(y_test_scaled.reshape(-1,1), y_pred_scaled.reshape(-1,1), marker='o', linestyle='', markersize=1)
    
    plt.plot([max(y_test_scaled), min(y_test_scaled)], [max(y_test_scaled), min(y_test_scaled)])
    plt.title(f"{model.__class__.__name__} y_pred VS y_test")
    plt.xlabel('Actual Global_active_power (kW)')
    plt.ylabel('Predicted Global_active_power (kW)')
    plt.show()

    pas = round(len(y_pred_scaled)*0.5/100)
    plt.figure()
    plt.plot(y_pred_scaled.reshape(-1,1)[::pas], label='Valeurs prédites', marker='', linestyle='-', markersize=0.1)
    plt.plot(y_test_scaled.reshape(-1,1)[::pas], label='Valeurs réelles', marker='', linestyle='-', markersize=0.1)
    plt.title(f"y_pred VS y_test, {model.__class__.__name__}")
    plt.xlabel('Actual Global_active_power (kW)')
    plt.ylabel('Predicted Global_active_power (kW)')
    plt.show()
    