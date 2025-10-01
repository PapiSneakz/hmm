# kraken_bot.py
"""
Improved Kraken scalping bot with robust error handling, retry/backoff, and Discord alerts.
- Retries network/HTTP errors with exponential backoff + jitter
- Falls back to cached prices when public API temporarily fails (to allow trading)
- Handles Kraken API "error" payloads gracefully
- Keeps attempting trades aggressively but safely (respects minimum order volume)
- Sends Telegram and Discord alerts on failures and trades

Notes:
- Keep your KRAKEN_API_KEY, KRAKEN_API_SECRET, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, and DISCORD_WEBHOOK in environment or .env
"""
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
import logging
import random
from typing import Tuple, Optional

# ---------------- CONFIG ----------------
ASSETS = ["ETH", "DOGE", "XRP"]
QUOTE = "EUR"

TRADE_EUR = 50.0           # Max EUR per trade per coin
MIN_PROFIT = 0.012         # 1.2% profit target (covers ~0.5% Kraken fees)

SHORT_EMA = 5
LONG_EMA = 20
OHLC_INTERVAL = 1
OHLC_COUNT = 200

POLL_INTERVAL = 30         # seconds between checks
LAST_ACTION_FILE = "last_action.json"

API_KEY = os.getenv("KRAKEN_API_KEY")
API_SECRET = os.getenv("KRAKEN_API_SECRET")

# Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Discord
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")

API_BASE = "https://api.kraken.com"

# Retry settings
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0
BACKOFF_FACTOR = 2.0
JITTER = 0.3

# Failure thresholds
ALERT_FAILURE_THRESHOLD = 6  # consecutive public/private errors before Telegram alert

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("kraken_bot")

# In-memory caches
price_cache = {}
ordermin_cache = {}

consecutive_public_errors = 0
consecutive_private_errors = 0

# ------------------- Telegram ----------------------
def send_telegram(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=10)
    except Exception as e:
        logger.warning("⚠️ Telegram send failed: %s", e)

# ------------------- Discord ----------------------
def send_discord(msg: str, ping_everyone: bool = False):
    if not DISCORD_WEBHOOK:
        return
    content = f"@everyone {msg}" if ping_everyone else msg
    try:
        requests.post(DISCORD_WEBHOOK, json={"content": content}, timeout=10)
    except Exception as e:
        logger.warning("⚠️ Discord send failed: %s", e)

# ------------------- Retry ------------------
def _sleep_backoff(attempt: int):
    base = INITIAL_BACKOFF * (BACKOFF_FACTOR ** attempt)
    jitter = base * JITTER
    wait = base + random.uniform(-jitter, jitter)
    time.sleep(max(wait, 0.1))

def request_with_retry(fn, *args, is_private=False, **kwargs):
    global consecutive_public_errors, consecutive_private_errors
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            result = fn(*args, **kwargs)
            if is_private:
                consecutive_private_errors = 0
            else:
                consecutive_public_errors = 0
            return result
        except requests.exceptions.RequestException as e:
            last_exc = e
            logger.warning("Request attempt %d failed: %s", attempt + 1, e)
            _sleep_backoff(attempt)
        except Exception as e:
            last_exc = e
            logger.exception("Unexpected error during request: %s", e)
            _sleep_backoff(attempt)

    if is_private:
        consecutive_private_errors += 1
        if consecutive_private_errors >= ALERT_FAILURE_THRESHOLD:
            send_telegram(f"⚠️ Kraken private API failing repeatedly ({consecutive_private_errors} errors)")
    else:
        consecutive_public_errors += 1
        if consecutive_public_errors >= ALERT_FAILURE_THRESHOLD:
            send_telegram(f"⚠️ Kraken public API failing repeatedly ({consecutive_public_errors} errors)")
    raise last_exc

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

def public_get(endpoint: str, params: dict = None) -> dict:
    url = f"{API_BASE}/0/public/{endpoint}"
    def _fn():
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    return request_with_retry(_fn, is_private=False)

def private_post(endpoint: str, data: dict) -> dict:
    require_keys()
    path = f"/0/private/{endpoint}"
    data = dict(data)
    data['nonce'] = int(time.time() * 1000)
    signature = _sign(path, data, API_SECRET)
    headers = {"API-Key": API_KEY, "API-Sign": signature}
    url = f"{API_BASE}{path}"
    def _fn():
        r = requests.post(url, headers=headers, data=data, timeout=15)
        r.raise_for_status()
        return r.json()
    return request_with_retry(_fn, is_private=True)

# ------------------- Market Functions ---------------------
def fetch_ohlc(pair: str, interval=OHLC_INTERVAL, count=OHLC_COUNT) -> pd.DataFrame:
    resp = public_get("OHLC", {"pair": pair, "interval": interval})
    if resp.get("error"):
        raise RuntimeError(f"OHLC error: {resp['error']}")
    result = resp.get("result", {})
    data = next((v for k, v in result.items() if k != "last"), None)
    if data is None:
        raise RuntimeError(f"No OHLC data for {pair}")
    df = pd.DataFrame(data, columns=["time","open","high","low","close","vwap","volume","count"])
    df['close'] = df['close'].astype(float)
    return df.tail(count)

def generate_scalp_signal(df: pd.DataFrame):
    if len(df) < LONG_EMA + 2:
        return None
    df['ema_short'] = df['close'].ewm(span=SHORT_EMA, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=LONG_EMA, adjust=False).mean()
    prev, last = df.iloc[-2], df.iloc[-1]
    if prev['ema_short'] <= prev['ema_long'] and last['ema_short'] > last['ema_long']:
        return 'buy'
    if prev['ema_short'] >= prev['ema_long'] and last['ema_short'] < last['ema_long']:
        return 'sell'
    return None

def get_balance():
    resp = private_post("Balance", {})
    if resp.get("error"):
        raise RuntimeError(f"Balance error: {resp['error']}")
    return resp.get("result", {})

def get_all_assetpairs():
    resp = public_get("AssetPairs")
    if resp.get("error"):
        raise RuntimeError(f"AssetPairs error: {resp['error']}")
    return resp.get("result", {})

def resolve_pairs(bases, quote=QUOTE):
    pairs_info = get_all_assetpairs()
    resolved = {}
    for base in bases:
        match = None
        for pair_key, info in pairs_info.items():
            alt, ws = info.get('altname',''), info.get('wsname','')
            if base.upper() in f"{pair_key} {alt} {ws}".upper() and quote.upper() in f"{pair_key} {alt} {ws}".upper():
                match = (pair_key, info)
                break
        if not match:
            for pair_key, info in pairs_info.items():
                if base.upper() in pair_key.upper() and quote.upper() in pair_key.upper():
                    match = (pair_key, info)
                    break
        if not match:
            raise RuntimeError(f"Could not resolve pair for {base}/{quote}.")
        resolved[base.upper()] = {"pair": match[0], "base_asset": match[1].get("base")}
    return resolved

def get_min_order_volume(pair):
    if pair in ordermin_cache:
        return ordermin_cache[pair]
    info = public_get("AssetPairs")
    result = info.get("result", {})
    minv = float(result[pair].get("ordermin", 0))
    ordermin_cache[pair] = minv
    return minv

def get_price(pair):
    try:
        ticker = public_get("Ticker", {"pair": pair})
        if ticker.get("error"):
            raise RuntimeError(f"Ticker error: {ticker['error']}")
        for k, v in ticker.get("result", {}).items():
            if k != "last":
                price = float(v['c'][0])
                price_cache[pair] = price
                return price
    except Exception as e:
        logger.warning("Failed to fetch price for %s: %s", pair, e)
        if pair in price_cache:
            return price_cache[pair]
        raise

def place_market_order(pair, side, eur_amount) -> Tuple[Optional[dict], float]:
    try:
        price = get_price(pair)
    except Exception:
        return None, get_min_order_volume(pair)

    vol = round(eur_amount / price, 8)
    min_vol = get_min_order_volume(pair)
    if vol < min_vol:
        return None, min_vol

    order = {"ordertype": "market","type": side,"volume": str(vol),"pair": pair}
    resp = private_post("AddOrder", order)
    if resp.get("error"):
        return None, min_vol
    return resp, min_vol

# ------------------- Last Actions ----------------
def load_last_action():
    if os.path.exists(LAST_ACTION_FILE):
        try:
            with open(LAST_ACTION_FILE) as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_last_action(actions):
    try:
        with open(LAST_ACTION_FILE, "w") as f:
            json.dump(actions, f)
    except Exception as e:
        logger.warning("Failed to save last_action: %s", e)

# ------------------- Main Loop ---------------------------
def main():
    send_discord("🤖 Kraken scalping bot started with profit protection!")
    logger.info("Running scalping bot...")
    last_action = load_last_action()
    resolved = resolve_pairs(ASSETS, QUOTE)

    while True:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            balances = get_balance()
            fiat_balance = float(balances.get("Z" + QUOTE, 0) or 0)
            per_asset_balances = {b: float(balances.get(info['base_asset'], 0) or 0) for b, info in resolved.items()}

            asset_signals = {}
            for base, info in resolved.items():
                try:
                    df = fetch_ohlc(info['pair'])
                    asset_signals[base] = generate_scalp_signal(df)
                except Exception as e:
                    logger.warning("OHLC fail %s: %s", base, e)
                    asset_signals[base] = None

            # Log the cycle but don't ping @everyone
            cycle_msg = f"{timestamp} | Signals: {asset_signals} | Fiat: {fiat_balance:.2f} {QUOTE} | Balances: {per_asset_balances}"
            logger.info(cycle_msg)
            send_discord(cycle_msg, ping_everyone=False)

            executed_any = False

            for base, signal in asset_signals.items():
                pair = resolved[base]['pair']
                balance = per_asset_balances[base]
                last = last_action.get(base, {})

                # -------- BUY --------
                if signal == 'buy' and last.get('side') != 'buy':
                    if fiat_balance < 1:
                        logger.info("%s | Skipping BUY %s: not enough fiat (%.2f %s)", timestamp, base, fiat_balance, QUOTE)
                        continue
                    eur_amount = min(TRADE_EUR, fiat_balance)
                    resp, min_vol = place_market_order(pair, 'buy', eur_amount)
                    if resp:
                        price = get_price(pair)
                        last_action[base] = {"side": "buy", "price": price}
                        fiat_balance -= eur_amount
                        executed_any = True
                        msg = f"🚀 BUY {base} at {price:.6f} {QUOTE}"
                        logger.info("%s | %s", timestamp, msg)
                        send_discord(msg, ping_everyone=True)
                    else:
                        logger.info("%s | Skipping BUY %s: below min order (%.8f < %.8f)", timestamp, base, eur_amount, min_vol)

                # -------- SELL --------
                elif signal == 'sell' and last.get('side') == 'buy':
                    buy_price = last.get('price')
                    price = None
                    try:
                        price = get_price(pair)
                    except Exception as e:
                        logger.warning("No price for %s: %s", base, e)

                    if balance <= 0:
                        logger.info("%s | Skipping SELL %s: balance is zero", timestamp, base)
                    elif buy_price is None:
                        logger.info("%s | Skipping SELL %s: no stored buy price", timestamp, base)
                    elif price is None:
                        logger.info("%s | Skipping SELL %s: no price available", timestamp, base)
                    else:
                        target_price = buy_price * (1 + MIN_PROFIT)
                        if price < buy_price:
                            logger.info("%s | Skipping SELL %s: price %.6f < buy price %.6f", timestamp, base, price, buy_price)
                        elif price < target_price:
                            logger.info("%s | Skipping SELL %s: price %.6f < target profit %.6f", timestamp, base, price, target_price)
                        else:
                            eur_equivalent = balance * price
                            eur_amount = min(eur_equivalent, TRADE_EUR)
                            resp, min_vol = place_market_order(pair, 'sell', eur_amount)
                            if resp:
                                last_action[base] = {"side": "sell"}
                                executed_any = True
                                msg = f"💰 SELL {base} at {price:.6f} {QUOTE}"
                                logger.info("%s | %s", timestamp, msg)
                                send_discord(msg, ping_everyone=True)
                            else:
                                logger.info("%s | Skipping SELL %s: order below min volume %.8f", timestamp, base, min_vol)

            if executed_any:
                save_last_action(last_action)
            else:
                logger.info("%s | No trades executed.", timestamp)

        except Exception as e:
            logger.exception("%s | Error in main loop: %s", timestamp, e)
            send_discord(f"⚠️ Error in main loop: {e}")

        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()

