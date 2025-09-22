import os
import time
import hmac
import hashlib
import base64
from urllib.parse import urlencode
import requests
import pandas as pd

# ---------------- CONFIG ----------------
PAIR = "XBTEUR"  # Change if needed
API_KEY = os.getenv("KRAKEN_API_KEY")
API_SECRET = os.getenv("KRAKEN_API_SECRET")
# ----------------------------------------

API_BASE = "https://api.kraken.com"

def require_keys():
    if not API_KEY or not API_SECRET:
        raise RuntimeError("KRAKEN_API_KEY and KRAKEN_API_SECRET must be set in environment.")

def _sign(path: str, data: dict, secret: str):
    post_data = urlencode(data)
    encoded = (str(data['nonce']) + post_data).encode()
    message = path.encode() + hashlib.sha256(encoded).digest()
    mac = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
    return base64.b64encode(mac.digest()).decode()

def public_get(endpoint: str, params: dict = None):
    url = f"{API_BASE}/0/public/{endpoint}"
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def private_post(endpoint: str, data: dict):
    require_keys()
    path = f"/0/private/{endpoint}"
    data['nonce'] = int(time.time() * 1000)
    signature = _sign(path, data, API_SECRET)
    headers = {
        "API-Key": API_KEY,
        "API-Sign": signature,
        "Content-Type": "application/x-www-form-urlencoded",
    }
    url = f"{API_BASE}{path}"
    r = requests.post(url, headers=headers, data=data, timeout=15)
    r.raise_for_status()
    return r.json()

def get_balance():
    resp = private_post("Balance", {})
    return resp['result']

def get_min_order_volume(pair):
    info = public_get("AssetPairs")
    return float(info['result'][pair]['ordermin'])

def place_market_order(pair: str, side: str, amount_fiat: float):
    ticker = public_get("Ticker", {"pair": pair})
    price = float(ticker['result'][pair]['c'][0])
    vol = round(amount_fiat / price, 6)
    min_vol = get_min_order_volume(pair)
    if vol < min_vol:
        return None, min_vol
    order = {
        "ordertype": "market",
        "type": side,
        "volume": str(vol),
        "pair": pair
    }
    resp = private_post("AddOrder", order)
    return resp, min_vol

def main():
    print("Starting test order...")

    balances = get_balance()
    fiat_balance = float(balances.get("ZEUR", 0))
    btc_balance = float(balances.get("XXBT", 0))
    print(f"Detected balance: {fiat_balance} ZEUR, {btc_balance} BTC")

    # Use all available fiat for BUY
    resp, min_vol = place_market_order(PAIR, 'buy', fiat_balance)
    if resp:
        print("BUY order executed:", resp)
        # Calculate bought BTC amount
        vol_bought = float(resp['descr']['order'].split()[1])
        # Immediately sell the same amount
        btc_price = float(public_get("Ticker", {"pair": PAIR})['result'][PAIR]['c'][0])
        sell_fiat = vol_bought * btc_price
        sell_resp, _ = place_market_order(PAIR, 'sell', sell_fiat)
        if sell_resp:
            print("SELL order executed:", sell_resp)
        else:
            print("SELL order failed: balance too low")
    else:
        print(f"BUY order skipped: balance too low for minimum ({min_vol} BTC)")

if __name__ == "__main__":
    main()
