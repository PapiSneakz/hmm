# kraken_bot.py
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
import json

# ---------------- CONFIG (Balanced Profile) ----------------
ASSETS = ["BTC", "LTC", "DOGE"]  # base tickers to monitor
QUOTE = "EUR"                    # quote currency

TRADE_EUR = 30.0                 # Target euros per trade (max total per trade decision)

# Balanced scalping profile
SHORT_EMA = 5                    # fast EMA
LONG_EMA = 20                    # slow EMA
OHLC_INTERVAL = 1                # use 1-minute candles
OHLC_COUNT = 200                 # number of candles to fetch per asset

AGREE_THRESHOLD = 2              # majority agreement (2 of 3 assets)
POLL_INTERVAL = 30               # seconds between checks

LAST_ACTION_FILE = "last_action.json"

API_KEY = os.getenv("KRAKEN_API_KEY")
API_SECRET = os.getenv("KRAKEN_API_SECRET")
# -----------------------------------------------------------

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
        raise RuntimeError(f"No OHLC data returned for {pair}")
    df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
    df['close'] = df['close'].astype(float)
    return df.tail(count)


def generate_scalp_signal(df: pd.DataFrame):
    """
    Balanced scalping signal using EMA crossover (5 vs 20).
    Returns 'buy', 'sell', or None.
    """
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


def place_market_order(pair: str, side: str, eur_amount: float):
    ticker = public_get("Ticker", {"pair": pair})
    result = ticker.get('result', {})
    price = None
    for k, v in result.items():
        if k == 'last':
            continue
        price = float(v['c'][0])
        break
    if price is None:
        raise RuntimeError(f"Could not fetch ticker price for pair {pair}")

    vol = round(eur_amount / price, 8)
    min_vol = get_min_order_volume(pair)

    if vol < min_vol:
        if side == "buy":
            needed_eur = price * min_vol
            fiat_balance = float(get_balance().get("ZEUR", 0))
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


def load_last_action():
    if os.path.exists(LAST_ACTION_FILE):
        with open(LAST_ACTION_FILE, "r") as f:
            try:
                return json.load(f)
            except Exception:
                return {}
    return {}


def save_last_action(actions_dict):
    with open(LAST_ACTION_FILE, "w") as f:
        json.dump(actions_dict, f)


def main():
    print("Running MULTI-asset Kraken scalping bot (Balanced Profile: 5/20 EMA, 1m candles, threshold=2)...")
    last_action = load_last_action()
    try:
        resolved = resolve_pairs(ASSETS, QUOTE)
    except Exception as e:
        print("Error resolving asset pairs:", e)
        return

    print("Resolved pairs:")
    for base, info in resolved.items():
        print(f"  {base} -> pair {info['pair']} (base asset code: {info['base_asset']})")

    while True:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            balances = get_balance()
            fiat_balance = float(balances.get("Z" + QUOTE, balances.get(QUOTE, 0)) or 0)
            per_asset_balances = {}
            for base, info in resolved.items():
                base_asset_code = info['base_asset']
                per_asset_balances[base] = float(balances.get(base_asset_code, 0) or 0)

            asset_signals = {}
            for base, info in resolved.items():
                pair = info['pair']
                df = fetch_ohlc(pair, interval=OHLC_INTERVAL, count=OHLC_COUNT)
                signal = generate_scalp_signal(df)
                asset_signals[base] = signal

            buys = [b for b, s in asset_signals.items() if s == 'buy']
            sells = [b for b, s in asset_signals.items() if s == 'sell']

            print(f"{timestamp} | Signals: {asset_signals} | Fiat: {fiat_balance:.4f} {QUOTE} | Asset balances: {per_asset_balances}")

            executed_any = False

            # BUY
            if len(buys) >= AGREE_THRESHOLD and fiat_balance > 0:
                eur_per_asset = min(TRADE_EUR / max(1, len(buys)), fiat_balance / max(1, len(buys)))
                print(f"{timestamp} | Majority BUY ({len(buys)}). Attempting: {buys} each {eur_per_asset:.2f} {QUOTE}")
                for base in buys:
                    if last_action.get(base) == 'buy':
                        print(f"{timestamp} | {base}: last action already BUY. Skipping.")
                        continue
                    pair = resolved[base]['pair']
                    resp, min_vol = place_market_order(pair, 'buy', eur_per_asset)
                    if resp:
                        print(f"{timestamp} | BUY {base} executed:", resp)
                        last_action[base] = 'buy'
                        executed_any = True
                    else:
                        print(f"{timestamp} | Not enough balance for minimum order ({min_vol} {base}). Skipping.")
                if executed_any:
                    save_last_action(last_action)

            # SELL
            elif len(sells) >= AGREE_THRESHOLD:
                print(f"{timestamp} | Majority SELL ({len(sells)}). Attempting: {sells}")
                for base in sells:
                    base_bal = per_asset_balances.get(base, 0)
                    if base_bal <= 0:
                        print(f"{timestamp} | {base}: no balance to sell. Skipping.")
                        continue
                    if last_action.get(base) == 'sell':
                        print(f"{timestamp} | {base}: last action already SELL. Skipping.")
                        continue

                    pair = resolved[base]['pair']
                    ticker = public_get("Ticker", {"pair": pair})
                    result = ticker.get('result', {})
                    price = None
                    for k, v in result.items():
                        if k == 'last':
                            continue
                        price = float(v['c'][0])
                        break
                    if price is None:
                        print(f"{timestamp} | Could not fetch price for {base}. Skipping.")
                        continue
                    eur_equivalent = base_bal * price
                    eur_to_sell = min(eur_equivalent, TRADE_EUR / max(1, len(sells)))
                    resp, min_vol = place_market_order(pair, 'sell', eur_to_sell)
                    if resp:
                        print(f"{timestamp} | SELL {base} executed:", resp)
                        last_action[base] = 'sell'
                        executed_any = True
                    else:
                        print(f"{timestamp} | Not enough for minimum order ({min_vol} {base}). Skipping.")
                if executed_any:
                    save_last_action(last_action)

            else:
                print(f"{timestamp} | No majority signal. Last actions: {last_action}")

        except Exception as e:
            print(f"{timestamp} | Error in main loop:", e)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
