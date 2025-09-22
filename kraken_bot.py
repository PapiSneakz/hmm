from dotenv import load_dotenv
load_dotenv()  # loads variables from .env into os.environ

import os
import time
import hmac
import hashlib
import base64
from urllib.parse import urlencode
import requests
import pandas as pd

# ---------------- CONFIG ----------------
PAIR = "XXBTZEUR"          # Correct Kraken pair for BTC/EUR
TRADE_EUR = 30.0            # Max euros per trade
SHORT_SMA = 10
LONG_SMA = 30
POLL_INTERVAL = 30          # seconds between checks
LAST_ACTION_FILE = "last_action.txt"

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

def fetch_ohlc(pair: str, interval: int = 1, count: int = 200):
    resp = public_get("OHLC", {"pair": pair, "interval": interval})
    data = resp['result'][pair]
    df = pd.DataFrame(data, columns=["time","open","high","low","close","vwap","volume","count"])
    df['close'] = df['close'].astype(float)
    return df.tail(count)

def generate_signal(df: pd.DataFrame):
    df['sma_short'] = df['close'].rolling(window=SHORT_SMA).mean()
    df['sma_long'] = df['close'].rolling(window=LONG_SMA).mean()
    if len(df) < LONG_SMA + 2:
        return None
    prev, last = df.iloc[-2], df.iloc[-1]
    if prev['sma_short'] <= prev['sma_long'] and last['sma_short'] > last['sma_long']:
        return 'buy'
    if prev['sma_short'] >= prev['sma_long'] and last['sma_short'] < last['sma_long']:
        return 'sell'
    return None

def get_balance():
    resp = private_post("Balance", {})
    return resp['result']

def get_min_order_volume(pair):
    info = public_get("AssetPairs")
    return float(info['result'][pair]['ordermin'])

def place_market_order(pair: str, side: str, eur_amount: float):
    ticker = public_get("Ticker", {"pair": pair})
    price = float(ticker['result'][pair]['c'][0])
    vol = round(eur_amount / price, 6)
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

def load_last_action():
    if os.path.exists(LAST_ACTION_FILE):
        with open(LAST_ACTION_FILE, "r") as f:
            return f.read().strip()
    return None

def save_last_action(action):
    with open(LAST_ACTION_FILE, "w") as f:
        f.write(action)

def main():
    print("Running LIVE Kraken bot with max 30 EUR per trade...")
    last_action = load_last_action()

    while True:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            df = fetch_ohlc(PAIR)
            signal = generate_signal(df)
            balances = get_balance()
            fiat_balance = float(balances.get("ZEUR", 0))
            btc_balance = float(balances.get("XXBT", 0))

            print(f"{timestamp} | Signal: {signal} | Fiat: {fiat_balance} EUR | BTC: {btc_balance}")

            # ----- BUY -----
            if signal == 'buy' and last_action != 'buy' and fiat_balance > 0:
                trade_amount = min(fiat_balance, TRADE_EUR)
                resp, min_vol = place_market_order(PAIR, 'buy', trade_amount)
                if resp:
                    print(f"{timestamp} | BUY executed:", resp)
                    last_action = 'buy'
                    save_last_action(last_action)
                else:
                    print(f"{timestamp} | Balance too low for minimum order ({min_vol} BTC). Skipping BUY.")

            # ----- SELL -----
            elif signal == 'sell' and last_action == 'buy' and btc_balance > 0:
                btc_price = float(public_get("Ticker", {"pair": PAIR})['result'][PAIR]['c'][0])
                trade_amount = min(btc_balance * btc_price, TRADE_EUR)
                resp, min_vol = place_market_order(PAIR, 'sell', trade_amount)
                if resp:
                    print(f"{timestamp} | SELL executed:", resp)
                    last_action = 'sell'
                    save_last_action(last_action)
                else:
                    print(f"{timestamp} | Balance too low for minimum order ({min_vol} BTC). Skipping SELL.")

            else:
                print(f"{timestamp} | No trade executed. Last action: {last_action}")

        except Exception as e:
            print(f"{timestamp} | Error in main loop:", e)

        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
