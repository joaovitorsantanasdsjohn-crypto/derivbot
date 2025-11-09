import pandas as pd
import ta

def calc_indicators(df: pd.DataFrame):
    # df expected columns: open, high, low, close
    close = df['close'].astype(float)
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    upper = bb.bollinger_hband().iloc[-1]
    lower = bb.bollinger_lband().iloc[-1]
    last_close = close.iloc[-1]

    signal = None
    # signal only if RSI and Bollinger agree
    if rsi < 30 and last_close <= lower:
        signal = "CALL"
    elif rsi > 70 and last_close >= upper:
        signal = "PUT"
    return signal, float(rsi), float(upper), float(lower)
