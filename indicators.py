# indicators.py
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands

def calc_indicators(df):
    # RSI
    rsi = RSIIndicator(df["close"], window=14).rsi().iloc[-1]
    
    # EMAs
    ema20 = EMAIndicator(df["close"], window=20).ema_indicator().iloc[-1]
    ema50 = EMAIndicator(df["close"], window=50).ema_indicator().iloc[-1]
    
    # Bollinger Bands
    bb = BollingerBands(df["close"], window=20, window_dev=2)
    upper = bb.bollinger_hband().iloc[-1]
    lower = bb.bollinger_lband().iloc[-1]
    mid = bb.bollinger_mavg().iloc[-1]
    
    # ADX
    adx = ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).adx().iloc[-1]

    # Decision logic
    signal = None
    close = df["close"].iloc[-1]

    if (
        rsi > 55 and ema20 > ema50 and close > mid and adx > 20
    ):
        signal = "CALL"
    elif (
        rsi < 45 and ema20 < ema50 and close < mid and adx > 20
    ):
        signal = "PUT"

    return signal, rsi, upper, lower
