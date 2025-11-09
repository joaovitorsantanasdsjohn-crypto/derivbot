import os
import json
import time
import math
import asyncio
import threading
import websockets
import pandas as pd
from collections import defaultdict, deque
from flask import Flask
from indicators import calc_indicators
from model import MLModel

WS_URL = wss://ws.derivws.com/websockets/v3?app_id=111022
API_TOKEN = os.getenv("DERIV_TOKEN")  # DEFINA NO RENDER
SYMBOLS = ["frxEURUSD","frxUSDCHF","frxGBPCHF","frxUSDJPY","BTCUSD"]
AMOUNT = 1.0
DURATION = 5
DURATION_UNIT = "m"
CANDLE_SECONDS = 5 * 60  # 5 minutos
HISTORY_CANDLES = 20

# Risk / limits
DAILY_MAX_LOSSES = 3
DAILY_MAX_WINS = 5
CONSECUTIVE_LOSSES_TO_PAUSE = 2
PAUSE_SECONDS_AFTER_LOSSES = 2 * 60 * 60  # 2 horas
MODEL_PROBA_THRESHOLD = 0.7

# State
state_file = "state.json"
state = {"date": None, "wins": 0, "losses": 0, "consecutive_losses": 0, "paused_until": 0}

# Candle builders: per symbol keep current candle and history
candles = defaultdict(lambda: deque(maxlen=HISTORY_CANDLES))
current_candle = {}

# Flask
app = Flask(__name__)

@app.route("/")
def home():
    return "✅ BotDoJoão rodando"

# Load/save state
def load_state():
    global state
    try:
        with open(state_file, "r") as f:
            state = json.load(f)
    except Exception:
        pass
    # reset daily counters if date changed
    today = time.strftime("%Y-%m-%d")
    if state.get("date") != today:
        state.update({"date": today, "wins": 0, "losses": 0, "consecutive_losses": 0, "paused_until": 0})
        save_state()

def save_state():
    with open(state_file, "w") as f:
        json.dump(state, f)

# Helper: candle aggregation
def update_candle(symbol, tick_epoch, price):
    bucket = int(tick_epoch // CANDLE_SECONDS * CANDLE_SECONDS)
    c = current_candle.get(symbol)
    if c is None or c[0] != bucket:
        # close previous
        if c is not None:
            candles[symbol].append({"epoch": c[0], "open": c[1], "high": c[2], "low": c[3], "close": c[4]})
        # start new
        current_candle[symbol] = [bucket, price, price, price, price]
    else:
        # update
        current_candle[symbol][2] = max(current_candle[symbol][2], price)
        current_candle[symbol][3] = min(current_candle[symbol][3], price)
        current_candle[symbol][4] = price

# Decision and execution
async def request_proposal(ws, symbol, contract_type):
    req = {
        "proposal": 1,
        "amount": AMOUNT,
        "basis": "stake",
        "contract_type": contract_type,
        "currency": "USD",
        "symbol": symbol,
        "duration": DURATION,
        "duration_unit": DURATION_UNIT
    }
    await ws.send(json.dumps(req))
    while True:
        raw = await ws.recv()
        msg = json.loads(raw)
        if msg.get("proposal"):
            return msg["proposal"]

async def buy_contract(ws, proposal_id):
    req = {"buy": proposal_id}
    await ws.send(json.dumps(req))
    resp = json.loads(await ws.recv())
    return resp

async def run():
    load_state()
    if not API_TOKEN:
        print("ERRO: Defina DERIV_TOKEN como variável de ambiente.")
        return
    ml = MLModel()

    async with websockets.connect(WS_URL) as ws:
        # authorize
        await ws.send(json.dumps({"authorize": API_TOKEN}))
        auth = json.loads(await ws.recv())
        print("Authorize:", auth)

        # subscribe ticks for all symbols
        for s in SYMBOLS:
            await ws.send(json.dumps({"ticks": s, "subscribe": 1}))

        print("Subscribed to:", SYMBOLS)

        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            # tick
            if msg.get("tick"):
                symbol = msg["tick"]["symbol"]
                epoch = msg["tick"]["epoch"]
                price = float(msg["tick"]["quote"])
                update_candle(symbol, epoch, price)

                # if a candle just closed (len history increased)
                if len(candles[symbol]) >= HISTORY_CANDLES:
                    df = pd.DataFrame(list(candles[symbol]))
                    signal, rsi, upper, lower = calc_indicators(df)
                    # ML features: rsi, close, upper-lower, distance to band
                    features = [rsi, df["close"].iloc[-1], upper - lower, (df["close"].iloc[-1] - lower) / (upper - lower + 1e-9)]

                    # check risk/pauses
                    now = time.time()
                    if state.get("paused_until", 0) > now:
                        continue
                    if state.get("losses", 0) >= DAILY_MAX_LOSSES:
                        print("Limite diário de perdas alcançado. Pausando.")
                        state["paused_until"] = now + PAUSE_SECONDS_AFTER_LOSSES
                        save_state()
                        continue
                    if state.get("wins", 0) >= DAILY_MAX_WINS:
                        print("Limite diário de ganhos alcançado. Pausando." )
                        state["paused_until"] = now + PAUSE_SECONDS_AFTER_LOSSES
                        save_state()
                        continue

                    # Technical agreement: RSI + Bollinger must agree
                    tech_decision = signal  # CALL / PUT / None
                    if tech_decision is None:
                        continue

                    # ML predict
                    proba, ml_pred = ml.predict_proba_and_label(features)
                    # require ML agrees and probability above threshold
                    if ml_pred != tech_decision or proba < MODEL_PROBA_THRESHOLD:
                        continue

                    print(f"Abrindo operação {ml_pred} em {symbol} | prob={proba:.2f} | RSI={rsi:.2f}")
                    proposal = await request_proposal(ws, symbol, ml_pred)
                    prop_id = proposal.get("id")
                    if not prop_id:
                        print("Proposal sem id, pulando")
                        continue
                    buy_resp = await buy_contract(ws, prop_id)
                    print("Buy response:", buy_resp)

                    # Monitor contract result: Deriv envia "buy" result with `contract_id` and then a `proposal_open_contract` may be sent.
                    # Simplify: wait for 'proposal' with display_value? We'll wait for 'buy' response containing 'transaction' or 'buy'
                    # Here we capture the buy_resp and then wait for 'sell' result via ticks or history. For demo simplicity, we'll
                    # record immediate outcome if buy_resp contains 'error' mark loss, else treat as pending and wait for contract update.

                    if buy_resp.get("error"):
                        print("Erro ao comprar:", buy_resp.get("error"))
                        state["losses"] += 1
                        state["consecutive_losses"] += 1
                        save_state()
                        if state["consecutive_losses"] >= CONSECUTIVE_LOSSES_TO_PAUSE:
                            state["paused_until"] = time.time() + PAUSE_SECONDS_AFTER_LOSSES
                            save_state()
                    else:
                        # For demo, optimistic: increment wins (in practice, should wait the contract result).
                        # Better approach: store opened contract id and parse `proposal_open_contract` messages to determine result.
                        # To keep runtime-light, we'll track opened contracts and wait for `contract` messages.
                        pass

# Start bot in background thread and Flask web server

def start_bot_loop():
    asyncio.run(run())

if __name__ == "__main__":
    threading.Thread(target=start_bot_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
