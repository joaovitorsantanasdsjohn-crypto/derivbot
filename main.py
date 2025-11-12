import os
import json
import time
import asyncio
import threading
import websockets
import pandas as pd
from collections import defaultdict, deque
from flask import Flask
from indicators import calc_indicators
from model import MLModel

# -----------------------
# Configurações principais
# -----------------------
WS_URL = "wss://ws.derivws.com/websockets/v3?app_id=111022"
API_TOKEN = os.getenv("DERIV_TOKEN")  # DEFINA NO RENDER (chave exata DERIV_TOKEN)
SYMBOLS = ["frxEURUSD", "frxUSDCHF", "frxGBPCHF", "frxUSDJPY", "BTCUSD"]
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

# Flask (keep-alive)
app = Flask(__name__)

@app.route("/")
def home():
    return "✅ BotDoJoão rodando"

# -----------------------
# Load / Save state
# -----------------------
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

# -----------------------
# Helper: candle aggregation
# -----------------------
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

# -----------------------
# Request proposal / buy helpers
# -----------------------
async def request_proposal(ws, symbol, contract_type):
    req = {
        "proposal": 1,
        "amount": AMOUNT,
        "basis": "stake",
        "contract_type": contract_type,
        "currency": "USD",
        "symbol": symbol,
        "duration": DURATION,
        "duration_unit": DURATION_UNIT,
        "subscribe": 0
    }
    await ws.send(json.dumps(req))
    # wait for a proposal or error
    while True:
        raw = await ws.recv()
        try:
            msg = json.loads(raw)
        except Exception:
            print("Non-json response while requesting proposal:", raw)
            continue
        if msg.get("proposal"):
            return msg["proposal"]
        if msg.get("error"):
            return {"error": msg["error"]}
        # ignore other messages and keep waiting

async def buy_contract(ws, proposal_id):
    req = {"buy": proposal_id}
    await ws.send(json.dumps(req))
    raw = await ws.recv()
    try:
        resp = json.loads(raw)
    except Exception:
        print("Non-json buy response:", raw)
        resp = {"raw": raw}
    return resp

# -----------------------
# Main run loop (debuggable)
# -----------------------
async def run():
    load_state()
    if not API_TOKEN:
        print("ERRO: Defina DERIV_TOKEN como variável de ambiente no Render.")
        return

    # ML model
    ml = MLModel()  # carrega model.pkl se existir, senão SGD incremental

    # TEST BUY mode (use apenas em demo)
    TEST_BUY = os.getenv("TEST_BUY", "0") == "1"
    TEST_SYMBOL = os.getenv("TEST_SYMBOL", "frxEURUSD")
    TEST_SIDE = os.getenv("TEST_SIDE", "CALL")  # "CALL" ou "PUT"

    print("Conectando ao Deriv WS:", WS_URL)
    async with websockets.connect(WS_URL) as ws:
        # authorize
        await ws.send(json.dumps({"authorize": API_TOKEN}))
        raw_auth = await ws.recv()
        try:
            auth = json.loads(raw_auth)
        except Exception:
            auth = {"raw": raw_auth}
        print("Authorize:", auth)

        # subscribe ticks for all symbols
        for s in SYMBOLS:
            await ws.send(json.dumps({"ticks": s, "subscribe": 1}))
        print("Subscribed to:", SYMBOLS)

        # state containers for tracking
        pending_proposals = {}   # proposal_id -> info
        open_contracts = {}      # contract_id -> info

        # handle incoming message helper
        def process_contract_result(msg_key, content):
            # content: dict for proposal_open_contract or sell/transaction keys
            # Try to detect profit/loss in several common fields
            profit = None
            contract_id = None
            try:
                # common fields used by Deriv responses
                contract_id = content.get("contract_id") or content.get("id") or content.get("transaction_id")
                # some responses include 'profit' directly
                if "profit" in content:
                    profit = float(content.get("profit", 0))
                # some include 'payout' and 'buy_price'
                elif "payout" in content and "buy_price" in content:
                    profit = float(content.get("payout", 0)) - float(content.get("buy_price", 0))
                # some include 'sell_price' and 'buy_price'
                elif "sell_price" in content and "buy_price" in content:
                    profit = float(content.get("sell_price", 0)) - float(content.get("buy_price", 0))
                # fallback: try 'profit' in nested fields
                else:
                    for k in ("profit", "payout", "sell_price"):
                        if k in content:
                            try:
                                profit = float(content.get(k, 0))
                                break
                            except Exception:
                                pass
            except Exception as e:
                print("Erro ao processar resultado de contrato:", e, content)

            if contract_id:
                open_contracts[contract_id] = content
                print(f"Contract stored: {contract_id} | profit={profit}")

            # update state if we can determine result
            if profit is not None:
                if profit > 0:
                    state["wins"] += 1
                    state["consecutive_losses"] = 0
                    print("Registrado WIN (profit > 0).")
                else:
                    state["losses"] += 1
                    state["consecutive_losses"] += 1
                    print("Registrado LOSS (profit <= 0).")
                save_state()
                # if consecutive losses threshold reached, pause
                if state["consecutive_losses"] >= CONSECUTIVE_LOSSES_TO_PAUSE:
                    state["paused_until"] = time.time() + PAUSE_SECONDS_AFTER_LOSSES
                    save_state()
                    print("Pausando devido a perdas consecutivas.")

        # If TEST_BUY requested, make a single quick buy attempt (demo only)
        if TEST_BUY:
            print("TEST_BUY mode ativado — tentando proposal+buy em", TEST_SYMBOL, TEST_SIDE)
            test_proposal = await request_proposal(ws, TEST_SYMBOL, TEST_SIDE)
            print("Test proposal:", test_proposal)
            if test_proposal.get("error"):
                print("Test proposal error:", test_proposal["error"])
                return
            pid = test_proposal.get("id")
            if not pid:
                print("Test proposal sem id, abortando teste.")
                return
            test_buy = await buy_contract(ws, pid)
            print("Test buy response:", test_buy)
            return

        # main message loop
        while True:
            raw = await ws.recv()
            try:
                msg = json.loads(raw)
            except Exception:
                print("Mensagem não-json recebida:", raw)
                continue

            # Log messages of interest generically
            if any(k in msg for k in ("proposal", "buy", "error", "proposal_open_contract", "transaction", "sell")):
                print("Incoming special msg:", {k: msg.get(k) for k in ("proposal", "buy", "error", "proposal_open_contract", "transaction", "sell")})

            # If we get contract result notifications, process them
            if msg.get("proposal_open_contract"):
                process_contract_result("proposal_open_contract", msg["proposal_open_contract"])
            if msg.get("sell"):
                process_contract_result("sell", msg["sell"])
            if msg.get("transaction"):
                # some transaction payloads include results
                process_contract_result("transaction", msg["transaction"])

            # tick processing
            if msg.get("tick"):
                symbol = msg["tick"].get("symbol")
                epoch = msg["tick"].get("epoch")
                price = float(msg["tick"].get("quote"))
                update_candle(symbol, epoch, price)

                # only when we have enough candles
                if len(candles[symbol]) >= HISTORY_CANDLES:
                    df = pd.DataFrame(list(candles[symbol]))
                    try:
                        signal, rsi, upper, lower = calc_indicators(df)
                    except Exception as e:
                        print("Erro ao calcular indicadores:", e)
                        continue

                    print(f"TECH signal for {symbol}: {signal} | RSI={rsi:.2f}")

                    features = [
                        float(rsi),
                        float(df["close"].iloc[-1]),
                        float(upper - lower),
                        float((df["close"].iloc[-1] - lower) / (upper - lower + 1e-9))
                    ]

                    # checks (risk / pause)
                    now = time.time()
                    if state.get("paused_until", 0) > now:
                        # paused, skip
                        # only print occasionally
                        continue
                    if state.get("losses", 0) >= DAILY_MAX_LOSSES:
                        print("Limite diário de perdas alcançado, pausando.")
                        state["paused_until"] = now + PAUSE_SECONDS_AFTER_LOSSES
                        save_state()
                        continue
                    if state.get("wins", 0) >= DAILY_MAX_WINS:
                        print("Limite diário de ganhos alcançado, pausando.")
                        state["paused_until"] = now + PAUSE_SECONDS_AFTER_LOSSES
                        save_state()
                        continue

                    # Technical decision
                    tech_decision = signal
                    if tech_decision is None:
                        continue

                    # ML prediction
                    try:
                        proba, ml_pred = ml.predict_proba_and_label(features)
                    except Exception as e:
                        print("Erro no ML predict:", e)
                        continue

                    print(f"ML predict for {symbol}: pred={ml_pred}, prob={proba:.3f}")

                    if ml_pred != tech_decision or proba < MODEL_PROBA_THRESHOLD:
                        print("ML não concorda ou prob abaixo do limiar -> pulando")
                        continue

                    # Request proposal and buy
                    print(f"Requisitando proposal para {symbol} {ml_pred}")
                    proposal = await request_proposal(ws, symbol, ml_pred)
                    print("Proposal recebido:", proposal)
                    if proposal.get("error"):
                        print("Proposal error:", proposal["error"])
                        continue

                    prop_id = proposal.get("id")
                    if not prop_id:
                        print("Proposal sem id, pulando")
                        continue

                    # Track proposal
                    pending_proposals[prop_id] = {"symbol": symbol, "side": ml_pred, "time": time.time()}

                    buy_resp = await buy_contract(ws, prop_id)
                    print("Buy response:", buy_resp)

                    if buy_resp.get("error"):
                        print("Erro ao comprar:", buy_resp.get("error"))
                        state["losses"] += 1
                        state["consecutive_losses"] += 1
                        save_state()
                        if state["consecutive_losses"] >= CONSECUTIVE_LOSSES_TO_PAUSE:
                            state["paused_until"] = time.time() + PAUSE_SECONDS_AFTER_LOSSES
                            save_state()
                    else:
                        # try to extract contract id if present and store
                        cid = None
                        # Buy responses can vary — try common locations
                        if isinstance(buy_resp.get("buy"), dict):
                            cid = buy_resp["buy"].get("contract_id") or buy_resp["buy"].get("transaction_id") or buy_resp["buy"].get("id")
                        if cid:
                            open_contracts[cid] = {"symbol": symbol, "side": ml_pred, "open_time": time.time()}
                            print("Contrato armazenado (cid):", cid)
                        else:
                            print("Buy OK, porém sem contract_id detectado — aguardando mensagens posteriores para confirmação.")

            # small sleep to avoid 100% cpu loop
            await asyncio.sleep(0.01)

# -----------------------
# Thread starter + Flask runner
# -----------------------
def start_bot_loop():
    try:
        asyncio.run(run())
    except Exception as e:
        print("Erro no loop principal:", e)

if __name__ == "__main__":
    threading.Thread(target=start_bot_loop, daemon=True).start()
    # Flask runs in main thread to keep Render alive
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
