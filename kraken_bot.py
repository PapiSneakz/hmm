import os
import time
import hmac
import hashlib
import base64
from urllib.parse import urlencode
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

API_BASE = "https://api.kraken.com"
SHORT_SMA = 10
LONG_SMA = 30
POLL_INTERVAL = 30  # seconds
FEE_BUFFER = 0.995  # 0.5% buffer for fees

API_KEY = os.getenv("KRAKEN_API_KEY")
API_SECRET = os.getenv("KRAKEN_API_SECRET")

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

def private_post(endpoint: str, data: dict = {}):
    require_keys()
    data['nonce'] = int(time.time() * 1000)
    path = f"/0/private/{endpoint}"
    signature = _sign(path, data, API_SECRET)
    headers = {"API-Key": API_KEY, "API-Sign": signature}
    url = f"{API_BASE}/0/private/{endpoint}"
    r = requests.post(url, headers=headers, data=data, timeout=15)
    r.raise_for_status()
    return r.json()

def get_fiat_balance():
    balances = private_post("Balance").get("result", {})
    if 'ZUSD' in balances and float(balances['ZUSD']) > 0:
        return 'ZUSD', float(balances['ZUSD']), 'XBTUSD'
    elif 'ZEUR' in balances and float(balances['ZEUR']) > 0:
        return 'ZEUR', float(balances['ZEUR']), 'XBTEUR'
    else:
        return None, 0, None

def get_pair_info(pair_input):
    pair_info = public_get("AssetPairs", {"pair": pair_input})
    pair_key = list(pair_info['result'].keys())[0]
    min_volume = float(pair_info['result'][pair_key]['ordermin'])
    return pair_key, min_volume

def fetch_ohlc(pair_input: str, interval: int = 1, count: int = 200):
    pair_key, _ = get_pair_info(pair_input)
    resp = public_get("OHLC", {"pair": pair_key, "interval": interval})
    data = resp['result'][pair_key]
    df = pd.DataFrame(data, columns=["time","open","high","low","close","vwap","volume","count"])
    df['close'] = df['close'].astype(float)
    return df.tail(count), pair_key

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

def place_market_order(side: str, pair_input: str):
    fiat_code, balance, _ = get_fiat_balance()
    if not fiat_code:
        print("No USD or EUR balance available for trading.")
        return None

    pair_key, min_volume = get_pair_info(pair_input)
    price_resp = public_get("Ticker", {"pair": pair_input})
    price = float(price_resp['result'][pair_key]['c'][0])
    volume = round(balance * FEE_BUFFER / price, 6)

    if volume < min_volume:
        print(f"Balance too low for minimum order ({min_volume} BTC). Skipping {side.upper()} order.")
        return None

    order = {
        "ordertype": "market",
        "type": side,
        "volume": str(volume),
        "pair": pair_key
    }

    resp = private_post("AddOrder", order)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if resp['error']:
        print(f"{timestamp} | {side.upper()} order rejected:", resp['error'])
        return None
    else:
        descr = resp['result'].get('descr', {}).get('order', '')
        txid = resp['result'].get('txid', [])
        print(f"{timestamp} | {side.upper()} executed: {volume} BTC (~{round(volume*price, 2)} {fiat_code}) | txid: {txid} | {descr}")
        return resp['result']

def main():
    last_action = None
    print("Running LIVE Kraken bot with logging, internal pair keys, and fee-safe orders...")
    while True:
        try:
            fiat_code, balance, pair_input = get_fiat_balance()
            if not fiat_code:
                print("No USD or EUR balance available. Skipping iteration.")
                time.sleep(POLL_INTERVAL)
                continue

            df, pair_key = fetch_ohlc(pair_input)
            signal = generate_signal(df)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"{timestamp} | Signal: {signal}")

            if signal == 'buy' and last_action != 'buy':
                place_market_order('buy', pair_input)
                last_action = 'buy'
            elif signal == 'sell' and last_action == 'buy':
                place_market_order('sell', pair_input)
                last_action = 'sell'
            else:
                print(f"{timestamp} | No trade executed. Last action: {last_action}")

        except Exception as e:
            print("Error in main loop:", e)

        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
