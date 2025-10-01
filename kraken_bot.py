# kraken_bot.py
"""
Improved Kraken scalping bot with robust error handling and retry/backoff.
- Retries network/HTTP errors with exponential backoff + jitter
- Falls back to cached prices when public API temporarily fails (to allow trading)
- Handles Kraken API "error" payloads gracefully
- Keeps attempting trades aggressively but safely (respects minimum order volume)
- Sends Telegram alerts on repeated failures

Notes:
- Keep your KRAKEN_API_KEY, KRAKEN_API_SECRET, TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in environment or .env
- This file replaces the previous version; do NOT duplicate functionality elsewhere.
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

SHORT_EMA = 5              # smoother, fewer fake trades
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

API_BASE = "https://api.kraken.com"

# Retry settings
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0  # seconds
BACKOFF_FACTOR = 2.0
JITTER = 0.3           # add +/- jitter to backoff

# Failure thresholds
ALERT_FAILURE_THRESHOLD = 6  # consecutive public/private errors before Telegram alert

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("kraken_bot")

# Simple in-memory caches to allow trading when public API hiccups
price_cache = {}
ordermin_cache = {}

# counters for consecutive failures
consecutive_public_errors = 0
consecutive_private_errors = 0

# ------------------- Telegram ----------------------
def send_telegram(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram not configured, skipping message")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        logger.warning("⚠️ Telegram send failed: %s", e)

# ------------------- Utilities & Retry ------------------
def _sleep_backoff(attempt: int):
    base = INITIAL_BACKOFF * (BACKOFF_FACTOR ** attempt)
    jitter = base * JITTER
    wait = base + random.uniform(-jitter, jitter)
    if wait < 0:
        wait = 0.1
    time.sleep(wait)

def request_with_retry(fn, *args, is_private=False, **kwargs):
    """Call `fn(*args, **kwargs)` with retries on network/HTTP errors.
    Returns the function's return value or raises last exception after retries.
    Updates global error counters for alerts.
    """
    global consecutive_public_errors, consecutive_private_errors
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            result = fn(*args, **kwargs)
            # reset failure counters on success
            if is_private:
                consecutive_private_errors = 0
            else:
                consecutive_public_errors = 0
            return result
        except requests.exceptions.RequestException as e:
            last_exc = e
            logger.warning("Request attempt %d failed: %s", attempt + 1, e)
            _sleep_backoff(attempt)
            continue
        except Exception as e:
            last_exc = e
            logger.exception("Unexpected error during request: %s", e)
            _sleep_backoff(attempt)
            continue
    # exhausted retries
    if is_private:
        consecutive_private_errors += 1
        logger.error("Private API failed %d times", consecutive_private_errors)
        if consecutive_private_errors >= ALERT_FAILURE_THRESHOLD:
            send_telegram(f"⚠️ Kraken private API failing repeatedly: {consecutive_private_errors} errors")
    else:
        consecutive_public_errors += 1
        logger.error("Public API failed %d times", consecutive_public_errors)
        if consecutive_public_errors >= ALERT_FAILURE_THRESHOLD:
            send_telegram(f"⚠️ Kraken public API failing repeatedly: {consecutive_public_errors} errors")
    # raise last exception for caller to decide; many callers will handle errors
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
    data = dict(data)  # copy
    data['nonce'] = int(time.time() * 1000)
    signature = _sign(path, data, API_SECRET)
    headers = {
        "API-Key": API_KEY,
        "API-Sign": signature,
        "Content-Type": "application/x-www-form-urlencoded",
    }
    url = f"{API_BASE}{path}"

    def _fn():
        r = requests.post(url, headers=headers, data=data, timeout=15)
        r.raise_for_status()
        return r.json()

    return request_with_retry(_fn, is_private=True)

# ------------------- Market Functions ---------------------

def fetch_ohlc(pair: str, interval: int = OHLC_INTERVAL, count: int = OHLC_COUNT) -> pd.DataFrame:
    resp = public_get("OHLC", {"pair": pair, "interval": interval})
    if not isinstance(resp, dict):
        raise RuntimeError("Unexpected OHLC response")
    if resp.get('error'):
        # Kraken may return temporary errors in the payload
        logger.warning("OHLC API returned errors: %s", resp.get('error'))
        raise RuntimeError("OHLC returned error: %s" % resp.get('error'))

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


def get_balance() -> dict:
    resp = private_post("Balance", {})
    if not isinstance(resp, dict):
        raise RuntimeError("Unexpected Balance response")
    if resp.get('error'):
        raise RuntimeError("Balance error: %s" % resp.get('error'))
    return resp.get('result', {})


def get_all_assetpairs() -> dict:
    resp = public_get("AssetPairs")
    if resp.get('error'):
        raise RuntimeError("AssetPairs error: %s" % resp.get('error'))
    return resp.get('result', {})


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


def get_min_order_volume(pair) -> float:
    # cache ordemin to reduce public requests
    if pair in ordermin_cache:
        return ordermin_cache[pair]
    info = public_get("AssetPairs")
    result = info.get('result', {})
    if pair not in result:
        raise RuntimeError(f"Pair {pair} not found in AssetPairs")
    minv = float(result[pair].get('ordermin', 0))
    ordermin_cache[pair] = minv
    return minv


def get_price(pair) -> float:
    # attempt to fetch real-time price; fall back to cache if public API fails
    try:
        ticker = public_get("Ticker", {"pair": pair})
        if ticker.get('error'):
            raise RuntimeError("Ticker error: %s" % ticker.get('error'))
        result = ticker.get('result', {})
        for k, v in result.items():
            if k == 'last':
                continue
            price = float(v['c'][0])
            price_cache[pair] = price
            return price
    except Exception as e:
        logger.warning("Failed to fetch price for %s: %s", pair, e)
        # fall back to last cached price if available
        if pair in price_cache:
            logger.info("Using cached price for %s: %s", pair, price_cache[pair])
            return price_cache[pair]
        # if no cached price, propagate exception
        raise


def place_market_order(pair: str, side: str, eur_amount: float) -> Tuple[Optional[dict], float]:
    """Place a market order by EUR amount. Returns (response, min_vol)
    If private API fails, this function will retry (private_post already retries), and if caller still gets exception,
    it will return (None, min_vol) to indicate failure while providing min_vol info when possible.
    """
    # obtain a price (may use cache)
    try:
        price = get_price(pair)
    except Exception as e:
        logger.warning("Cannot determine price for %s to compute volume: %s", pair, e)
        # try to recover by looking at ordemin only; place minimal volume if side == buy and fiat enough
        try:
            min_vol = get_min_order_volume(pair)
            return None, min_vol
        except Exception:
            return None, 0

    vol = round(eur_amount / price, 8)
    try:
        min_vol = get_min_order_volume(pair)
    except Exception as e:
        logger.warning("Cannot determine ordemin for %s: %s", pair, e)
        min_vol = 0

    # if calculated volume is below min -> try to increase to min if buying and fiat available
    if vol < min_vol:
        if side == "buy":
            needed_eur = price * min_vol
            try:
                fiat_balance = float(get_balance().get("Z" + QUOTE, 0))
            except Exception as e:
                logger.warning("Failed to get fiat balance while handling min order: %s", e)
                return None, min_vol
            if needed_eur > fiat_balance:
                logger.info("Not enough fiat to meet min order volume for %s: needed %.2f, have %.2f", pair, needed_eur, fiat_balance)
                return None, min_vol
            vol = min_vol
        else:
            # for sell, if below min, refuse
            return None, min_vol

    order = {
        "ordertype": "market",
        "type": side,
        "volume": str(vol),
        "pair": pair
    }

    try:
        resp = private_post("AddOrder", order)
    except Exception as e:
        logger.warning("Failed to place order %s %s on %s: %s", side, vol, pair, e)
        return None, min_vol

    # Kraken returns errors in payload as well
    if resp.get('error'):
        logger.warning("AddOrder returned errors: %s", resp.get('error'))
        return None, min_vol

    return resp, min_vol

# ------------------- Load / Save Last Actions ----------------

def load_last_action():
    if os.path.exists(LAST_ACTION_FILE):
        with open(LAST_ACTION_FILE, "r") as f:
            try:
                data = json.load(f)
                for k, v in data.items():
                    if isinstance(v, str):
                        data[k] = {"side": v}
                return data
            except Exception:
                return {}
    return {}


def save_last_action(actions_dict):
    try:
        with open(LAST_ACTION_FILE, "w") as f:
            json.dump(actions_dict, f)
    except Exception as e:
        logger.warning("Failed to save last_action: %s", e)

# ------------------- Main Loop ---------------------------

def main():
    send_telegram("🤖 Kraken scalping bot (robust) started with profit protection!")
    logger.info("Running scalping bot (ETH+DOGE+XRP, EMA 5/20, profit-protected)...")
    last_action = load_last_action()

    try:
        resolved = resolve_pairs(ASSETS, QUOTE)
    except Exception as e:
        logger.exception("Error resolving pairs: %s", e)
        send_telegram(f"⚠️ Error resolving pairs: {e}")
        return

    logger.info("Resolved pairs:")
    for base, info in resolved.items():
        logger.info("  %s -> pair %s", base, info['pair'])

    while True:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            # fetch balances (private) - wrapped in retry
            balances = get_balance()
            fiat_balance = float(balances.get("Z" + QUOTE, 0) or 0)
            per_asset_balances = {base: float(balances.get(info['base_asset'], 0) or 0)
                                  for base, info in resolved.items()}

            asset_signals = {}
            for base, info in resolved.items():
                # fetch OHLC; if it fails, we'll attempt to use cached price to make a decision
                try:
                    df = fetch_ohlc(info['pair'], interval=OHLC_INTERVAL, count=OHLC_COUNT)
                    asset_signals[base] = generate_scalp_signal(df)
                except Exception as e:
                    logger.warning("Failed to fetch OHLC for %s: %s (will attempt to trade using cached price if available)", base, e)
                    # if OHLC is unavailable, don't block trading — fallback: None signal (conservative)
                    asset_signals[base] = None

            logger.info("%s | Signals: %s | Fiat: %.2f %s | Balances: %s", timestamp, asset_signals, fiat_balance, QUOTE, per_asset_balances)
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
                        try:
                            price = get_price(pair)
                        except Exception:
                            price = None
                        last_action[base] = {"side": "buy", "price": price}
                        fiat_balance -= eur_amount
                        executed_any = True
                        msg = f"🚀 BUY {base} executed at {price if price else 'unknown'} {QUOTE}"
                        logger.info("%s | %s %s", timestamp, msg, resp)
                        send_telegram(msg)
                    else:
                        logger.info("%s | Not enough balance for min order (%s %s). Skipping BUY.", timestamp, min_vol, base)

                # SELL — only if profitable
                elif signal == 'sell' and last.get('side') == 'buy' and balance > 0:
                    buy_price = last.get('price')
                    # if buy_price isn't available (e.g. cached or missing), try to use cached or current
                    try:
                        price = get_price(pair)
                    except Exception as e:
                        logger.warning("Could not fetch price for sell decision for %s: %s", base, e)
                        price = None

                    if buy_price is None:
                        # If we don't know buy_price, attempt to sell if price exists and achieves MIN_PROFIT vs cached buy price (not available)
                        logger.info("No stored buy price for %s; attempting conservative sell if price exists", base)

                    if price is not None and buy_price is not None:
                        target_price = buy_price * (1 + MIN_PROFIT)
                        if price > buy_price and price >= target_price:
                            eur_equivalent = balance * price
                            eur_amount = min(eur_equivalent, TRADE_EUR)
                            resp, min_vol = place_market_order(pair, 'sell', eur_amount)
                            if resp:
                                last_action[base] = {"side": "sell"}
                                executed_any = True
                                msg = f"💰 SELL {base} executed at {price:.6f} {QUOTE}"
                                logger.info("%s | %s %s", timestamp, msg, resp)
                                send_telegram(msg)
                            else:
                                logger.info("%s | Not enough for min order (%s %s). Skipping SELL.", timestamp, min_vol, base)
                        else:
                            logger.info("%s | %s sell skipped: price %s < target %s or not profitable", timestamp, base, price, target_price)
                    else:
                        logger.info("%s | Cannot evaluate sell for %s: missing price or buy_price", timestamp, base)

            if executed_any:
                save_last_action(last_action)
            else:
                logger.info("%s | No trades executed. Last actions: %s", timestamp, last_action)

        except Exception as e:
            logger.exception("%s | Error in main loop: %s", timestamp, e)
            # send a compact message to Telegram but avoid spamming
            send_telegram(f"⚠️ Error in main loop: {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()

