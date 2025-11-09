import os
save_state()
continue


# Technical agreement: RSI + Bollinger must agree
tech_decision = signal # CALL / PUT / None
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
