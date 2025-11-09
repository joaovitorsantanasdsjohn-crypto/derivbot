import os
import time
import json
import threading
import asyncio
import websockets
import pandas as pd
from flask import Flask

from indicators import get_rsi_bollinger
from model import MLModel
from utils import request_proposal, buy_contract, save_state, load_state

# Configurações gerais
SYMBOLS = ["frxEURUSD", "frxUSDCHF", "frxGBPCHF", "frxUSDJPY", "BTCUSD"]
MODEL_PROBA_THRESHOLD = 0.7
STAKE = 1.0
LOSS_LIMIT = 3
WIN_LIMIT = 5
CONSECUTIVE_LOSSES_TO_PAUSE = 2
PAUSE_SECONDS_AFTER_LOSSES = 2 * 60 * 60  # 2 horas
DERIV_TOKEN = os.getenv("DERIV_TOKEN")
if not DERIV_TOKEN:
    raise ValueError("Token DERIV_TOKEN não encontrado. Configure no Render.")

# Flask app (para UptimeRobot)
app = Flask(__name__)

@app.route('/')
def home():
    return "BotDoJoão rodando com Flask e WebSocket na Deriv ✅", 200

# Estado do bot
state = load_state()

ml = MLModel("model.pkl")

async def run():
    uri = "wss://ws.derivws.com/websockets/v3?app_id=1089"
    async with websockets.connect(uri) as ws:
        # Autenticação
        await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
        auth_resp = await ws.recv()
        print("Autorização:", auth_resp)

        while True:
            for symbol in SYMBOLS:
                # Verifica limites diários
                if state["losses"] >= LOSS_LIMIT:
                    print("Limite de perda diária atingido.")
                    await asyncio.sleep(60)
                    continue
                if state["wins"] >= WIN_LIMIT:
                    print("Limite de ganho diário atingido.")
                    await asyncio.sleep(60)
                    continue

                # Pausa se necessário
                if state.get("paused_until", 0) > time.time():
                    print(f"Pausado até {time.ctime(state['paused_until'])}")
                    await asyncio.sleep(60)
                    continue

                # Coleta candles
                await ws.send(json.dumps({
                    "ticks_history": symbol,
                    "end": "latest",
                    "count": 20,
                    "style": "candles",
                    "granularity": 300  # 5 minutos
                }))
                msg = json.loads(await ws.recv())
                candles = msg.get("candles", [])
                if not candles or len(candles) < 20:
                    print(f"Sem candles suficientes para {symbol}")
                    continue

                df = pd.DataFrame(candles)
                df["close"] = df["close"].astype(float)

                # Indicadores
                rsi, signal = get_rsi_bollinger(df)
                if signal is None:
                    continue

                features = [rsi, df["close"].iloc[-1]]
                proba, ml_pred = ml.predict_proba_and_label(features)

                if ml_pred != signal or proba < MODEL_PROBA_THRESHOLD:
                    continue

                print(f"Abrindo operação {ml_pred} em {symbol} | prob={proba:.2f} | RSI={rsi:.2f}")
                proposal = await request_proposal(ws, symbol, ml_pred)
                prop_id = proposal.get("id")
                if not prop_id:
                    print("Proposal sem id, pulando")
                    continue

                buy_resp = await buy_contract(ws, prop_id)
                print("Buy response:", buy_resp)

                if buy_resp.get("error"):
                    print("Erro ao comprar:", buy_resp.get("error"))
                    state["losses"] += 1
                    state["consecutive_losses"] += 1
                    save_state(state)
                    if state["consecutive_losses"] >= CONSECUTIVE_LOSSES_TO_PAUSE:
                        state["paused_until"] = time.time() + PAUSE_SECONDS_AFTER_LOSSES
                        save_state(state)
                else:
                    # Para simplificação, considera win imediato (na prática, aguardar resultado real)
                    state["wins"] += 1
                    state["consecutive_losses"] = 0
                    save_state(state)

                await asyncio.sleep(5)

            await asyncio.sleep(2)

def start_bot_loop():
    asyncio.run(run())

if __name__ == "__main__":
    threading.Thread(target=start_bot_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
