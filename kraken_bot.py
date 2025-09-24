# kraken_bot.py
from dotenv import load_dotenv
load_dotenv()
import os
import time
import hmac
import hashlib
import base64
from urllib.parse import urlencode
import requests
import pandas as pd
import json

# ---------------- CONFIG (Faster EMA / Higher trades) ----------------
ASSETS = ["ETH", "DOGE", "XRP"]
QUOTE = "EUR"

TRADE_EUR = 50.0           # Max EUR per trade per coin
MIN_PROFIT = 0.005         # Minimum profit threshold: 0.5% (~covers fees)

SHORT_EMA = 3
LONG_EMA = 8
OHLC_INTERVAL = 1
OHLC_COUNT = 200

POLL_INTERVAL = 30          # seconds between checks
LAST_ACTION_FILE = "last_action.json"

API_KEY = os.getenv("KRAKEN_API_KEY")
API_SECRET = os.getenv("KRAKEN_API_SECRET")

# Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# -----------------------------------------------------------
API_BASE = "https://api.kraken.com"

# ------------------- Telegram Helper ----------------------
def send_telegram(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print("⚠️ Telegram send failed:", e)

# ------------------- Kraken API ---------------------------
def require_keys():
    if not API_KEY or not API_SECRET:
        raise RuntimeError("KRAKEN_API_KEY and KRAKEN_API_SECRET must be set.")

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

# ------------------- Market Functions ---------------------
def fetch_ohlc(pair: str, interval: int = OHLC_INTERVAL, count: int = OHLC_COUNT):
    resp = public_get("OHLC", {"pair": pair, "interval": interval})
    result = resp.get('result', {})
    data = None
    for k, v in result.items():
        if k == 'last':
            continue
        data = v
        break
    if data is None:
        raise RuntimeError(f"No OHLC data for {pair}")
    df = pd.DataFrame(data, columns=["time","open","high","low","close","vwap","volume","count"])
    df['close'] = df['close'].astype(float)
    return df.tail(count)

def generate_scalp_signal(df: pd.DataFrame):
    if len(df) < LONG_EMA + 2:
        return None
    df = df.copy()
    df['ema_short'] = df['close'].ewm(span=SHORT_EMA, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=LONG_EMA, adjust=False).mean()
    prev = df.iloc[-2]
    last = df.iloc[-1]
    if prev['ema_short'] <= prev['ema_long'] and last['ema_short'] > last['ema_long']:
        return 'buy'
    if prev['ema_short'] >= prev['ema_long'] and last['ema_short'] < last['ema_long']:
        return 'sell'
    return None

def get_balance():
    resp = private_post("Balance", {})
    return resp['result']

def get_all_assetpairs():
    resp = public_get("AssetPairs")
    return resp['result']

def resolve_pairs(bases, quote=QUOTE):
    pairs_info = get_all_assetpairs()
    resolved = {}
    for base in bases:
        match = None
        for pair_key, info in pairs_info.items():
            if not isinstance(info, dict):
                continue
            alt = info.get('altname', '')
            ws = info.get('wsname', '')
            pair_text = f"{pair_key} {alt} {ws}".upper()
            if base.upper() in pair_text and quote.upper() in pair_text:
                match = (pair_key, info)
                break
        if not match:
            for pair_key, info in pairs_info.items():
                if base.upper() in pair_key.upper() and quote.upper() in pair_key.upper():
                    match = (pair_key, info)
                    break
        if not match:
            raise RuntimeError(f"Could not resolve pair for {base}/{quote}.")
        pair_key, info = match
        resolved[base.upper()] = {
            'pair': pair_key,
            'base_asset': info.get('base'),
            'pair_info': info
        }
    return resolved

def get_min_order_volume(pair):
    info = public_get("AssetPairs")
    return float(info['result'][pair]['ordermin'])

def get_price(pair):
    ticker = public_get("Ticker", {"pair": pair})
    result = ticker.get('result', {})
    for k, v in result.items():
        if k == 'last':
            continue
        return float(v['c'][0])
    return 0

def place_market_order(pair: str, side: str, eur_amount: float):
    price = get_price(pair)
    if price == 0:
        return None, 0
    vol = round(eur_amount / price, 8)
    min_vol = get_min_order_volume(pair)
    if vol < min_vol:
        if side == "buy":
            needed_eur = price * min_vol
            fiat_balance = float(get_balance().get("Z" + QUOTE, 0))
            if needed_eur > fiat_balance:
                return None, min_vol
            vol = min_vol
        else:
            return None, min_vol
    order = {
        "ordertype": "market",
        "type": side,
        "volume": str(vol),
        "pair": pair
    }
    resp = private_post("AddOrder", order)
    return resp, min_vol

# ------------------- Load / Save Last Actions ----------------
def load_last_action():
    if os.path.exists(LAST_ACTION_FILE):
        with open(LAST_ACTION_FILE, "r") as f:
            try:
                data = json.load(f)
                # Normalize old string entries to dict
                for k, v in data.items():
                    if isinstance(v, str):
                        data[k] = {"side": v}
                return data
            except Exception:
                return {}
    return {}

def save_last_action(actions_dict):
    with open(LAST_ACTION_FILE, "w") as f:
        json.dump(actions_dict, f)

# ------------------- Main Loop ---------------------------
def main():
    send_telegram("🤖 Kraken scalping bot started!")
    print("Running PER-COIN scalping bot (ETH+DOGE+XRP, fast EMA)...")
    last_action = load_last_action()
    try:
        resolved = resolve_pairs(ASSETS, QUOTE)
    except Exception as e:
        print("Error resolving pairs:", e)
        send_telegram(f"⚠️ Error resolving pairs: {e}")
        return

    print("Resolved pairs:")
    for base, info in resolved.items():
        print(f"  {base} -> pair {info['pair']}")

    while True:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            balances = get_balance()
            fiat_balance = float(balances.get("Z" + QUOTE, 0) or 0)
            per_asset_balances = {base: float(balances.get(info['base_asset'], 0) or 0)
                                  for base, info in resolved.items()}

            asset_signals = {}
            for base, info in resolved.items():
                df = fetch_ohlc(info['pair'], interval=OHLC_INTERVAL, count=OHLC_COUNT)
                asset_signals[base] = generate_scalp_signal(df)

            print(f"{timestamp} | Signals: {asset_signals} | Fiat: {fiat_balance:.2f} {QUOTE} | Balances: {per_asset_balances}")
            executed_any = False

            for base, signal in asset_signals.items():
                pair = resolved[base]['pair']
                balance = per_asset_balances.get(base, 0)
                last = last_action.get(base, {})

                # BUY
                if signal == 'buy' and (last.get('side') != 'buy') and fiat_balance > 0:
                    eur_amount = min(TRADE_EUR, fiat_balance)
                    resp, min_vol = place_market_order(pair, 'buy', eur_amount)
                    if resp:
                        price = get_price(pair)
                        last_action[base] = {"side": "buy", "price": price}
                        fiat_balance -= eur_amount
                        executed_any = True
                        msg = f"🚀 BUY {base} executed at {price:.2f} {QUOTE}"
                        print(f"{timestamp} | {msg}", resp)
                        send_telegram(msg)
                    else:
                        print(f"{timestamp} | Not enough balance for min order ({min_vol} {base}). Skipping BUY.")

                # SELL
                elif signal == 'sell' and last.get('side') == 'buy' and balance > 0:
                    buy_price = last['price']
                    price = get_price(pair)
                    target_price = buy_price * (1 + MIN_PROFIT)
                    if price >= target_price:
                        eur_equivalent = balance * price
                        eur_amount = min(eur_equivalent, TRADE_EUR)
                        resp, min_vol = place_market_order(pair, 'sell', eur_amount)
                        if resp:
                            last_action[base] = {"side": "sell"}
                            executed_any = True
                            msg = f"💰 SELL {base} executed at {price:.2f} {QUOTE}"
                            print(f"{timestamp} | {msg}", resp)
                            send_telegram(msg)
                        else:
                            print(f"{timestamp} | Not enough for min order ({min_vol} {base}). Skipping SELL.")
                    else:
                        print(f"{timestamp} | {base} sell skipped: current {price:.6f} < target {target_price:.6f}")

            if executed_any:
                save_last_action(last_action)
            else:
                print(f"{timestamp} | No trades executed. Last actions: {last_action}")

        except Exception as e:
            print(f"{timestamp} | Error in main loop:", e)
            send_telegram(f"⚠️ Error in main loop: {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
