# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 01:47:50 2026

@author: jgive
"""

import numpy as np
import pandas as pd
import time
from queue import Queue as que
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
#from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import yfinance as yf
import websocket as web
import json
import threading 
import plotly.io as pio
import os
import logging 
from quixstreams import Application

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
pio.renderers.default = "browser"

data = 'C:/Users/jgive/Documents/Python Projects/Mini Projects/Data/msft_intraday-30min_historical-data-03-17-2025.csv'
web.enableTrace(True)
ws = web.WebSocketApp("wss://ws.finnhub.io?token=d6b3o3pr01qnr27jhvr0d6b3o3pr01qnr27jhvrg")

yData = yf.download("DOGE-USD", period = "7d", interval = "1m")
#print(yData)
msft_daily = pd.DataFrame()
msft_wkly = pd.DataFrame()
msft_monthly = pd.DataFrame()
msft_rolling = pd.DataFrame()
current = pd.DataFrame()
signal_df = pd.DataFrame()
class DataHandler:
    
    def __init__(self, data):
        self.data = data
        
    def dataReader(self):
        
        df = self.data.copy()
        df.sort_index(ascending=True, inplace=True)
        #df['%Chg'] = pd.to_numeric(df['%Chg'].str.replace('%',''))/100
        df = df.rename(columns=str.lower)
        #df['close'] = df['last']
        df['raw_return'] = (df['close']-df['open'])
        
        return df
    
df = DataHandler(yData).dataReader()
df.index = df.index.tz_localize(None)
msft_mean = ((df['open'].copy()+df['close'].copy())/2).dropna()
close = df['close'].astype(float).squeeze()
rolling_return_1 =close.pct_change(1)
rolling_return_5 = close.pct_change(5)
rolling_return_10 = close.pct_change(10)
msft_rolling['return_1'] = rolling_return_1
msft_rolling['return_5'] = rolling_return_5
msft_rolling['return_10'] = rolling_return_10
msft_rolling['mean_5'] = close.rolling(window=5).mean()
msft_rolling['mean_10'] = close.rolling(window=10).mean()
msft_rolling['mean_20'] = close.rolling(window=20).mean()
mean20 = msft_rolling['mean_20'].astype(float).squeeze()
msft_rolling['sma_ratio'] = msft_rolling['mean_5'] / msft_rolling['mean_20']
msft_rolling['price_sma_dist'] = close.align(mean20, join='left')[0] / mean20 - 1
msft_rolling['lag_1'] = msft_rolling['return_1'].shift(1)
msft_rolling['lag_2'] = msft_rolling['return_1'].shift(2)
msft_rolling['lag_3'] = msft_rolling['return_1'].shift(3)
msft_rolling['high'] = df['high'].copy().rolling(window=20).max()
msft_rolling['low']  = df['low'].copy().rolling(window=20).min()
std = df['close'].rolling(20).std()
msft_rolling['std'] = std
msft_rolling['volatility_5'] = msft_rolling['return_1'].rolling(window = 5).std().dropna()
msft_rolling['volatility_10'] = msft_rolling['return_1'].rolling(window = 10).std().dropna()
msft_rolling['volatility_20'] = msft_rolling['return_1'].rolling(window = 20).std().dropna()
msft_mean = msft_mean.squeeze()
mean5 = msft_rolling['mean_5'].squeeze()
mean20 = msft_rolling['mean_20'].squeeze()
std = std.squeeze()
msft_rolling['zscore_5'] = (msft_mean - mean5) / std
msft_rolling['zscore_20'] = (msft_mean - mean20) / std
# quick version
delta = df['close'].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))
msft_rolling
Return = df['close'].pct_change()
msft_rolling = msft_rolling.add_prefix('roll_')
df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
df['mean_msft'] = msft_mean
df['rsi_msft'] = rsi
df['return_msft']= Return
df = df.join(msft_rolling)
df['next_return'] = df['return_msft'].shift(-1)
df['target'] = (df['next_return'] >= 0).astype(int)
df = df.dropna()
#print(df)
#print("After feature engineering and dropna:", df.shape)
#msft_daily.index = df.index.to_timestamp()
aligned_features = df.reindex(df.index, method='ffill')
aligned_features = aligned_features.ffill()
aligned_features = aligned_features.dropna(subset=['target'])
X = df[['open_doge-usd', 'high_doge-usd', 'low_doge-usd', 'close_doge-usd', 'mean_msft','rsi_msft', 'roll_return_1', 'roll_return_5','roll_return_10', 'roll_mean_5', 'roll_mean_20', 'roll_sma_ratio', 'roll_price_sma_dist', 'roll_lag_1', 'roll_high', 'roll_low','roll_volatility_5', 'roll_zscore_20']]
#print(aligned_features)
y = df['target']
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
#print("X_train shape:", X_train.shape)
#print("y_train shape:", y_train.shape)
#rf = RandomForestClassifier(n_estimators=300,max_depth=None,min_samples_split=2,min_samples_leaf=1,random_state=42)
mlp = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        solver='adam',
        max_iter=10000000,
        random_state=42, 
        hidden_layer_sizes=(128, 64, 32),
        alpha=0.0005
    )
)

model = mlp.fit(X_train, y_train)
#model = rf.fit(X_train, y_train)

class Strategy:
    def __init__(self, model, feature_df, portfolio):
        self.model = model
        self.feature_df = feature_df
        self.portfolio = portfolio

    def Intraday(self, market_event):
        ts = market_event.timestamp
        ts = pd.to_datetime(market_event.timestamp, unit='ms')
        

        row = market_event.row

        x =self.feature_df.loc[ts].values.reshape(1, -1)
        pred = self.model.predict(x)[0]
        if isinstance(pred, (pd.Series, pd.DataFrame, np.ndarray, list, tuple)):
            pred = np.array(pred).reshape(-1)[0]
        proba = self.model.predict_proba(x)[0]
        p_down, p_up = proba
        #print(proba)
        #print("PRED RAW TYPE:", type(pred))
       #print("PRED RAW VALUE:", pred)
        pred = int(pred)
        cash = self.portfolio.cash
        if isinstance(cash, pd.Series):
            cash = cash.squeeze()

        cash = float(cash)
        
        if pred == 1 and cash > 0 and row['roll_zscore_20'] <-1 :
            return SignalEvent(ts, "BUY")
        elif pred == 0 and self.portfolio.positions["DOGE-USD"] > 0 and row['roll_zscore_20'] >-1:
            return SignalEvent(ts, "SELL")
        else:
            return SignalEvent(ts, "HOLD")
    
class SignalEvent:
    
    def __init__(self, timestamp, signal_type):
        self.type = "SIGNAL"
        self.timestamp = timestamp
        self.signal_type = signal_type 
    
    def __repr__(self):
        return f"SignalEvent({self.timestamp}, {self.signal_type})"

class OrderEvent:
    def __init__(self, timestamp, symbol, direction, quantity, price,  raw_return ,order_type="MARKET"):
        self.type = "ORDER"
        self.timestamp = timestamp
        self.symbol = symbol
        self.direction = direction      # "BUY" or "SELL"
        self.quantity = quantity
        self.price = price
        self.raw_return = raw_return
        self.order_type = order_type

    def __repr__(self):
        return f"OrderEvent({self.timestamp}, {self.symbol}, {self.direction}, {self.quantity})"
    
class FillEvent:
    def __init__(self, timestamp, symbol, direction, quantity, price, raw_return, commission=0.0):
        self.type = "FILL"
        self.timestamp = timestamp
        self.symbol = symbol
        self.direction = direction
        self.quantity = quantity
        self.price = price
        self.commission = commission
        self.raw_return = raw_return

    def __repr__(self):
        return f"FillEvent({self.timestamp}, {self.symbol}, {self.direction}, {self.quantity}, {self.price})"    
    
class MarketEvent:
    def __init__(self, timestamp, row):
        self.type = "MARKET"
        self.timestamp = timestamp
        self.row = row

class ExecutionHandler:
    def execute_order(self, order_event):
        return FillEvent(
            order_event.timestamp,
            order_event.symbol,
            order_event.direction,
            order_event.quantity,
            order_event.price,
            order_event.raw_return
        )
"""       
def Daily_plt(df, i, window, signal_df=None):
    plt.ion()
    start = max(0, i - window + 1)
    plt_Daily = df[['open_doge-usd', 'high_doge-usd', 'low_doge-usd', 'close_doge-usd']].iloc[start:i+1]

    if plt_Daily.empty:
        return

    x = plt_Daily.index.to_numpy()
    y = plt_Daily['close_doge-usd'].values

    # Create figure once
    if not hasattr(Daily_plt, "fig"):
        Daily_plt.fig, Daily_plt.ax = plt.subplots(figsize=(10,5))
        Daily_plt.line, = Daily_plt.ax.plot([], [], lw=2, color='blue')
        Daily_plt.marker, = Daily_plt.ax.plot([], [], 'o', color='red')

        # signal overlay (black dots)
        Daily_plt.trend_scatter, = Daily_plt.ax.plot([], [], '', color='black')

    ax = Daily_plt.ax

    # Update price line + marker
    Daily_plt.line.set_data(x, y)
    Daily_plt.marker.set_data([x[-1]], [y[-1]])

    # --- NEW: update signal points ---
    if signal_df is not None and not signal_df.empty:
        signal_slice = signal_df.loc[plt_Daily.index.intersection(signal_df.index)]
        if not signal_slice.empty:
            tx = signal_slice.index.to_numpy()
            ty = signal_slice["price"].values
            Daily_plt.trend_scatter.set_data(tx, ty)
        else:
            Daily_plt.trend_scatter.set_data([], [])

    ax.relim()
    ax.autoscale_view()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    Daily_plt.fig.canvas.draw()
    Daily_plt.fig.canvas.flush_events()
"""
class Portfolio:
    def __init__(self, initial_cash=100):
        self.cash = initial_cash
        self.positions = {"DOGE-USD": 100}

    def update_from_fill(self, fill_event):
        qty = fill_event.quantity
        price = fill_event.price
        direction = fill_event.direction
        raw_ret = fill_event.raw_return

        if direction == "BUY":
            self.positions["DOGE-USD"] += qty
            self.cash -= qty * price

        elif direction == "SELL":
            self.positions["DOGE-USD"] -= qty
            self.cash += qty * price
        elif direction == "HOLD":
            self.cash += raw_ret
            
def Main_thread():   
    global current, signal_df
    line = que()
    signal_points = []
    portfolio = Portfolio()
    strategy = Strategy(model, X, portfolio)
    execution = ExecutionHandler()

    
    for i in range(len(df)):
        line.put(df.iloc[i])
       
    while not line.empty():
        row = line.get()
        current = pd.concat([current, row.to_frame().T])
        time.sleep(0.5)
        timestamp = current.index[-1]
        price = df.loc[current.index[-1],'close_doge-usd']
        raw_return = df.loc[current.index[-1],'return_msft']
        market_event = MarketEvent(timestamp, current.iloc[-1])
        signal = strategy.Intraday(market_event)
        
        if signal:
            print(signal)
            signal_points.append({
                "timestamp": timestamp,
                "price": price,
                "type": signal.signal_type
                })

        
            order = OrderEvent(
                timestamp,
                symbol="DOGE-USD",
                direction=signal.signal_type.replace("Consider ", ""),
                quantity=1,
                price=price,
                raw_return=raw_return)
            print(order)
            fill = execution.execute_order(order)
            print(fill)
            portfolio.update_from_fill(fill)

        # Plot
        i = df.index.get_loc(timestamp)
        if signal_points:
            signal_df = pd.DataFrame(signal_points).set_index("timestamp")
        else:
            signal_df = None

#        Daily_plt(df, i, 50, signal_df=signal_df)

    # Print final PnL
        print("Final Cash:", portfolio.cash)
        print("Final Position:", portfolio.positions["DOGE-USD"])
app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='live-chart'),
    dcc.Interval(id='interval', interval=500, n_intervals=0)  # update every 0.5s
])

@app.callback(
    Output('live-chart', 'figure'),
    Input('interval', 'n_intervals')
)
def update_chart(n):
    global current, signal_df

    if len(current) < 2:
        return go.Figure()
    df = current.tail(300)
    df['time'] = df.index
    df = df.join(signal_df, how='left')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['close_doge-usd'],
        mode='lines',
        name='Price'
    ))


    buys = df[df['type'] == 'BUY']
    sells = df[df['type'] == 'SELL']

    fig.add_trace(go.Scatter(
            x=buys['time'], y=buys['close_doge-usd'],
            mode='markers', marker=dict(color='green', size=10),
            name='BUY'
        ))

    fig.add_trace(go.Scatter(
            x=sells['time'], y=sells['close_doge-usd'],
            mode='markers', marker=dict(color='red', size=10),
            name='SELL'
        ))

    fig.update_layout(template='plotly_dark', height=600)
    return fig    
if __name__ == '__main__':   
    t_1 = threading.Thread(target=Main_thread, daemon=True)

    t_1.start()


    app.run(debug=True)
