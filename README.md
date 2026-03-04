# **Real‑Time ML‑Driven Backtesting Engine v1.0**

This project implements a fully functional **event‑driven backtesting and live‑stream simulation engine** in a single Python file. It combines machine‑learning‑based signal generation, feature engineering, order execution, portfolio accounting, and real‑time visualization into one cohesive system.

The engine processes historical or live market data one bar at a time, generates trading signals using a trained neural network, executes simulated trades, updates portfolio state, and streams results to a live Plotly Dash dashboard.

---

## **Features**

### **Event‑Driven Architecture**
- Market data is streamed bar‑by‑bar through a queue.
- Each bar becomes a `MarketEvent`.
- The engine processes events in strict chronological order.
- Signals → Orders → Fills → Portfolio updates occur automatically.

### **Machine Learning Strategy**
- Uses an `MLPClassifier` wrapped in a `StandardScaler` pipeline.
- Predicts next‑bar direction using engineered features.
- Generates BUY / SELL / HOLD signals based on:
  - model prediction  
  - model probability  
  - z‑score filters  
  - portfolio state  

### **Feature Engineering**
The engine computes a rich set of predictive features, including:

- Rolling returns (1, 5, 10)
- Rolling means (5, 10, 20)
- Rolling volatility windows
- Rolling highs/lows
- SMA ratios
- Price‑SMA distance
- Lagged returns
- RSI (14‑period)
- Z‑scores (5 and 20)
- Raw returns and next‑bar returns (for training)

These features are aligned with timestamps and fed into the ML model during live simulation.

### **Order & Execution System**
Implements a full event pipeline:

- `SignalEvent` → `OrderEvent` → `FillEvent`
- Market‑order execution model
- Raw return propagation for HOLD logic
- Fractional quantity support (crypto‑friendly)

### **Portfolio Accounting**
Tracks:

- Cash  
- Position size  
- PnL impact from fills  
- HOLD‑based cash adjustments  
- Crypto trading (DOGE‑USD)

### **Real‑Time Visualization**
A live Plotly Dash dashboard displays:

- Streaming price data  
- BUY/SELL markers  
- Auto‑updating chart  
- Last 300 bars of activity  

The engine runs in a background thread while Dash updates the UI.

### **Live Data Support**
- Historical data via Yahoo Finance (`yfinance`)
- WebSocket connection to Finnhub (optional)
- Timestamp normalization for streaming

---

## **Architecture Overview**

Although implemented in a single file, the engine follows a modular architecture:

- **DataHandler** — loads and preprocesses data  
- **Strategy** — ML‑driven signal generation  
- **MarketEvent / SignalEvent / OrderEvent / FillEvent** — event types  
- **ExecutionHandler** — simulates fills  
- **Portfolio** — tracks cash and positions  
- **Main Thread** — event loop and trading logic  
- **Dash App** — real‑time visualization  

This structure mirrors professional event‑driven backtesting engines.

---

## **How It Works**

1. **Load Data**  
   Historical DOGE‑USD data is downloaded and preprocessed.

2. **Feature Engineering**  
   Rolling indicators, RSI, z‑scores, and lagged features are computed.

3. **Train ML Model**  
   An MLP neural network is trained to predict next‑bar direction.

4. **Start Event Loop**  
   Each bar is pushed into a queue and processed sequentially.

5. **Generate Signals**  
   The strategy uses ML predictions + z‑score filters to emit signals.

6. **Execute Orders**  
   Orders are converted to fills using a simple execution model.

7. **Update Portfolio**  
   Cash and positions adjust based on fills.

8. **Stream to Dashboard**  
   A Dash app displays price and signals in real time.

---

## **Tech Stack**

- **Python**
- **Pandas / NumPy**
- **scikit‑learn (MLPClassifier)**
- **Plotly Dash**
- **Yahoo Finance (yfinance)**
- **WebSockets (Finnhub)**
- **Threading**
- **Queue‑based event system**

---

## **Versioning**

- **v1.0** — Fully functional single‑file engine  
- **v1.5 (planned)** — Modular file structure  
- **v2.0 (planned)** — Multi‑strategy, multi‑asset, batch backtesting  

---

## **Future Improvements**

- Modular architecture (data, strategy, execution, portfolio modules)
- Config‑driven engine setup
- Logging and performance metrics
- Multi‑asset portfolio support
- Strategy optimization and hyperparameter tuning

---

## **Project Status**

This is the first complete working version of the engine.  
It demonstrates:

- event‑driven design  
- ML‑based trading logic  
- real‑time visualization  
- portfolio accounting  
- live streaming  

All in a single, self‑contained Python file.

---


