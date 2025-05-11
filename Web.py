import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from keras.models import load_model
from xgboost import XGBRegressor

model = load_model('D:\Code\Predictor\Predictors.keras')

st.header('Predictor')

stock = st.text_input('Enter Name In UPPERCASE(Eg: VNM): ')
days = st.text_input('Enter Days of Prediction(5 | 10 | 20): ')

if stock == "VNM":
    data = pd.read_csv("D:\Code\Predictor\VNM.csv", parse_dates=['Date'], dayfirst=False, thousands=',', decimal='.')
if stock == "HAG":
    data = pd.read_csv("D:\Code\Predictor\HAG.csv", parse_dates=['Date'], dayfirst=False, thousands=',', decimal='.')
if stock == "HPG":
    data = pd.read_csv("D:\Code\Predictor\HPG.csv", parse_dates=['Date'], dayfirst=False, thousands=',', decimal='.')
if stock == "FPT":
    data = pd.read_csv("D:\Code\Predictor\FPT.csv", parse_dates=['Date'], dayfirst=False, thousands=',', decimal='.')
if stock == "MBB":
    data = pd.read_csv("D:\Code\Predictor\MBB.csv", parse_dates=['Date'], dayfirst=False, thousands=',', decimal='.')


st.subheader('Stock Data')
st.write(data)

data.rename(columns={
    "Price": "Close",
    "Vol.": "Volume",
}, inplace = True)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0,1))

#convert M L K trong cột KL thành số thực
def convert_volume(vol_str):
    if isinstance(vol_str, str):
        if vol_str[-1] == 'M':
            return float(vol_str[:-1]) * 1_000_000
        elif vol_str[-1] == 'K':
            return float(vol_str[:-1]) * 1_000
    return float(vol_str)

data['Volume'] = data['Volume'].apply(convert_volume)
data = data.sort_values('Date')
data = data.ffill()
data

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale

def add_target_columns(data):
    data['Target_5'] = data['Close'].shift(-5)
    data['Target_10'] = data['Close'].shift(-10)
    data['Target_20'] = data['Close'].shift(-20)

def add_features(data):
    #%chênh giá mở và đóng, chênh giá đỉnh và sàn
    data['Daily_Return'] = (data['Close'] - data['Open']) / data['Open']
    data['High_Low_Spread'] = (data['High'] - data['Low']) / data['Open']

    #xem xu hướng của giá cổ phiếu dựa trên trung bình 15, 10 và 20 ngày gần nhất
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()

    #xem độ biến động thông qua độ lệch chuẩn của giá 5 ngày
    data['Volatility_5'] = data['Close'].rolling(window=5).std()
    data['Volume_MA5'] = data['Volume'].rolling(window=5).mean()
    data['Volume_Spike'] = (data['Volume'] > 1.5 * data['Volume'].rolling(window=20).mean()).astype(int)

add_features(data)
add_target_columns(data)

data['DayOfWeek'] = data['Date'].dt.dayofweek

def prepare_data_for_predicting_n_days_ahead(tbl, n: str):
    if (n != '5' and n != '10' and  n != '20'):
        return 0
        
    X_cols = ['Daily_Return', 'High_Low_Spread', 'MA5', 'MA10', 
        'MA20', 'Volatility_5', 'Volume_MA5', 'Volume_Spike', 'DayOfWeek']
    y_col = ['Target_' + n]
    needed_cols = X_cols + y_col

    temp1 = tbl[needed_cols].ffill().bfill()
    X = temp1[X_cols]
    y = temp1[y_col]

    split_point = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:] 
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    return X_train, X_test, y_train, y_test

def train(X_train, X_test, y_train, y_test, model):
    
    model.fit(X_train, y_train.values.ravel())
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")

    return y_pred

def plot_the_diff(y_test, y_pred, n: str, name: str):
    plt.plot(y_test.values, label='Actual Price', color='blue')
    plt.plot(y_pred, label='Predicted Price', color='red')
    plt.legend()
    plt.title(f'{name}: Actual vs Predicted Prices ({n} days)')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()

def temp2(data: pd.DataFrame, name: str, model):
    for n in [days]:
        X_train, X_test, y_train, y_test = prepare_data_for_predicting_n_days_ahead(data, n)
        y_pred = train(X_train, X_test, y_train, y_test, model)
        plot_the_diff(y_test, y_pred, n, name)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
model_rfr = RandomizedSearchCV(RandomForestRegressor(), param_grid, cv=3, scoring='neg_mean_squared_error', n_iter=10)
model_rfr_raw = RandomForestRegressor(n_estimators=100, random_state=42)

param_grid_x = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 1.0]
}
model_xgbr = RandomizedSearchCV(XGBRegressor(), param_grid_x, cv=3, scoring='neg_mean_squared_error', n_iter=10)
model_xgbr_raw = RandomForestRegressor(n_estimators=100, random_state=42)

st.subheader('Dự đoán bằng LSTM')
figLSTM = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Closed Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(figLSTM)


st.subheader('Dự đoán bằng RandomForestRegressor')
figRFR = plt.figure(figsize=(8,6))
temp2(data, 'RandomForestRegressor', model_rfr_raw)
st.pyplot(figRFR)


st.subheader('Dự đoán bằng XGBoost')
figXGB = plt.figure(figsize=(8,6))
temp2(data, 'XGBoost', model_xgbr_raw)
st.pyplot(figXGB)