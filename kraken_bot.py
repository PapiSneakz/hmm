# kraken_bot.py
"""
Kraken scalping bot with multiple buys, average entry, profit-target selling,
robust error handling, retry/backoff, Telegram/Discord alerts, and hourly summary.

Features:
- ASSETS = ["ETH","DOGE","XRP"]
- Up to MAX_BUYS_PER_COIN buys per coin (4)
- Average entry price tracked per coin
- Sell all at once when average entry + MIN_PROFIT reached
- Detailed "Skipping ..." reasons logged and also sent to Discord (but @everyone only for real trades/hourly summary)
- Persistent state: last_action.json and profit_log.json
- Discord webhook hardcoded (user requested); Telegram optional via env
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

TRADE_EUR = 50.0           # base EUR per buy
MIN_PROFIT = 0.012         # 1.2% profit target

SHORT_EMA = 5
LONG_EMA = 20
OHLC_INTERVAL = 1
OHLC_COUNT = 200

POLL_INTERVAL = 30         # seconds between checks
LAST_ACTION_FILE = "last_action.json"
PROFIT_LOG_FILE = "profit_log.json"

MAX_BUYS_PER_COIN = 4      # user requested 4 buys max

# ---------------- API KEYS ----------------
API_KEY = os.getenv("KRAKEN_API_KEY")
API_SECRET = os.getenv("KRAKEN_API_SECRET")

# Telegram (optional)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Discord webhook - hardcoded as requested (change if needed)
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1423014611380076686/7r6lxPcnVVBIi567_L9aSEkLSm7cbPOEEI3lEtBUz-prATtaHhbDN5Gtu1DRuLHXwJPo"

API_BASE = "https://api.kraken.com"

# Retry/backoff
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0
BACKOFF_FACTOR = 2.0
JITTER = 0.3

ALERT_FAILURE_THRESHOLD = 6  # consecutive failures to alert via Telegram

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("kraken_bot")

# In-memory caches and counters
price_cache = {}
ordermin_cache = {}
consecutive_public_errors = 0
consecutive_private_errors = 0

# ---------------- TELEGRAM / DISCORD ----------------
def send_telegram(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram not configured; skipping Telegram msg.")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=10)
    except Exception as e:
        logger.warning("⚠️ Telegram send failed: %s", e)

def send_discord(msg: str, ping_everyone: bool = False):
    """Send message to Discord webhook. If ping_everyone=True prepend @everyone."""
    if not DISCORD_WEBHOOK:
        logger.debug("Discord webhook not configured; skipping Discord msg.")
        return
    content = f"@everyone {msg}" if ping_everyone else msg
    try:
        requests.post(DISCORD_WEBHOOK, json={"content": content}, timeout=10)
    except Exception as e:
        logger.warning("⚠️ Discord send failed: %s", e)

# ---------------- UTILS & RETRY ----------------
def _sleep_backoff(attempt: int):
    base = INITIAL_BACKOFF * (BACKOFF_FACTOR ** attempt)
    jitter = base * JITTER
    wait = base + random.uniform(-jitter, jitter)
    time.sleep(max(wait, 0.1))

def request_with_retry(fn, *args, is_private=False, **kwargs):
    """Call fn with retries. Updates global error counters and possibly sends Telegram alerts."""
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
    # exhausted
    if is_private:
        consecutive_private_errors += 1
        logger.error("Private API failed %d times", consecutive_private_errors)
        if consecutive_private_errors >= ALERT_FAILURE_THRESHOLD:
            send_telegram(f"⚠️ Kraken private API failing repeatedly ({consecutive_private_errors})")
    else:
        consecutive_public_errors += 1
        logger.error("Public API failed %d times", consecutive_public_errors)
        if consecutive_public_errors >= ALERT_FAILURE_THRESHOLD:
            send_telegram(f"⚠️ Kraken public API failing repeatedly ({consecutive_public_errors})")
    raise last_exc

# ---------------- KRAKEN API ----------------
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
    headers = {"API-Key": API_KEY, "API-Sign": signature, "Content-Type": "application/x-www-form-urlencoded"}
    url = f"{API_BASE}{path}"
    def _fn():
        r = requests.post(url, headers=headers, data=data, timeout=15)
        r.raise_for_status()
        return r.json()
    return request_with_retry(_fn, is_private=True)

# ---------------- MARKET FUNCTIONS ----------------
def fetch_ohlc(pair: str, interval=OHLC_INTERVAL, count=OHLC_COUNT) -> pd.DataFrame:
    resp = public_get("OHLC", {"pair": pair, "interval": interval})
    if not isinstance(resp, dict):
        raise RuntimeError("Unexpected OHLC response")
    if resp.get("error"):
        raise RuntimeError(f"OHLC error: {resp.get('error')}")
    result = resp.get("result", {})
    data = None
    for k, v in result.items():
        if k == "last":
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
        raise RuntimeError(f"Balance error: {resp.get('error')}")
    return resp.get('result', {})

def get_all_assetpairs() -> dict:
    resp = public_get("AssetPairs")
    if resp.get("error"):
        raise RuntimeError(f"AssetPairs error: {resp.get('error')}")
    return resp.get('result', {})

def resolve_pairs(bases, quote=QUOTE):
    pairs_info = get_all_assetpairs()
    resolved = {}
    for base in bases:
        match = None
        for pair_key, info in pairs_info.items():
            if not isinstance(info, dict):
                continue
            alt = info.get('altname','')
            ws = info.get('wsname','')
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
        resolved[base.upper()] = {'pair': pair_key, 'base_asset': info.get('base'), 'pair_info': info}
    return resolved

def get_min_order_volume(pair) -> float:
    if pair in ordermin_cache:
        return ordermin_cache[pair]
    info = public_get("AssetPairs")
    result = info.get("result", {})
    if pair not in result:
        raise RuntimeError(f"Pair {pair} not found in AssetPairs")
    minv = float(result[pair].get('ordermin', 0))
    ordermin_cache[pair] = minv
    return minv

def get_price(pair) -> float:
    try:
        ticker = public_get("Ticker", {"pair": pair})
        if ticker.get('error'):
            raise RuntimeError(f"Ticker error: {ticker.get('error')}")
        result = ticker.get('result', {})
        for k, v in result.items():
            if k == 'last':
                continue
            price = float(v['c'][0])
            price_cache[pair] = price
            return price
    except Exception as e:
        logger.warning("Failed to fetch price for %s: %s", pair, e)
        if pair in price_cache:
            logger.info("Using cached price for %s: %s", pair, price_cache[pair])
            return price_cache[pair]
        raise

def place_market_order(pair: str, side: str, eur_amount: float) -> Tuple[Optional[dict], float, str]:
    """Place a market order by EUR amount. Returns (response, min_vol, fail_reason)."""
    try:
        price = get_price(pair)
    except Exception as e:
        # can't compute volume w/out price
        try:
            min_vol = get_min_order_volume(pair)
        except Exception:
            min_vol = 0
        return None, min_vol, f"Failed to fetch price: {e}"

    vol = round(eur_amount / price, 8)
    try:
        min_vol = get_min_order_volume(pair)
    except Exception as e:
        min_vol = 0
        logger.warning("Cannot determine ordemin for %s: %s", pair, e)

    if vol < min_vol:
        return None, min_vol, f"Order volume too small: {vol:.8f} < min {min_vol:.8f}"

    order = {"ordertype": "market", "type": side, "volume": str(vol), "pair": pair}
    try:
        resp = private_post("AddOrder", order)
    except Exception as e:
        return None, min_vol, f"Failed to place order: {e}"

    if resp.get('error'):
        return None, min_vol, f"Kraken AddOrder error: {resp.get('error')}"
    return resp, min_vol, ""

# ---------------- PERSISTENCE ----------------
def load_json_file(path: str, default):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)
            return default
    else:
        return default

def save_json_file(path: str, obj):
    try:
        with open(path, "w") as f:
            json.dump(obj, f)
    except Exception as e:
        logger.warning("Failed to save %s: %s", path, e)

def load_last_action() -> dict:
    data = load_json_file(LAST_ACTION_FILE, {})
    # Handle older formats: if values are floats/strings convert to structured dict
    fixed = {}
    for k, v in data.items():
        if isinstance(v, dict):
            # ensure buys list exists
            d = dict(v)
            if 'buys' not in d:
                # older format might have side/price
                if d.get('side') == 'buy':
                    # try to read price
                    price = d.get('price')
                    if price:
                        d['buys'] = [price]
                        d['avg_price'] = float(price)
                    else:
                        d['buys'] = []
                        d['avg_price'] = None
                else:
                    d['buys'] = d.get('buys', [])
                    d['avg_price'] = d.get('avg_price')
            fixed[k] = d
        elif isinstance(v, list):
            # convert list to buys
            fixed[k] = {'buys': v, 'avg_price': (sum(v)/len(v) if v else None)}
        elif isinstance(v, (int, float, str)):
            # unknown older format: wrap into buys if numeric
            try:
                p = float(v)
                fixed[k] = {'buys': [p], 'avg_price': p}
            except Exception:
                fixed[k] = {'buys': [], 'avg_price': None}
        else:
            fixed[k] = {'buys': [], 'avg_price': None}
    return fixed

def save_last_action(actions: dict):
    save_json_file(LAST_ACTION_FILE, actions)

def load_profit_log():
    data = load_json_file(PROFIT_LOG_FILE, {})
    # expected keys: realized_eur (float), trades (int)
    realized = float(data.get('realized_eur', 0.0))
    trades = int(data.get('trades', 0))
    return {'realized_eur': realized, 'trades': trades}

def save_profit_log(log: dict):
    save_json_file(PROFIT_LOG_FILE, log)

# ---------------- MAIN LOOP ----------------
def main():
    send_discord("🤖 Kraken scalping bot started with profit protection!", ping_everyone=True)
    logger.info("Running scalping bot...")
    last_action = load_last_action()           # dict: base -> {'buys': [prices], 'avg_price': x}
    profit_log = load_profit_log()             # {'realized_eur': X, 'trades': N}
    try:
        resolved = resolve_pairs(ASSETS, QUOTE)
    except Exception as e:
        logger.exception("Error resolving pairs: %s", e)
        send_discord(f"⚠️ Error resolving pairs: {e}")
        send_telegram(f"⚠️ Error resolving pairs: {e}")
        return

    # sanity: ensure every asset key exists
    for a in ASSETS:
        if a not in last_action:
            last_action[a] = {'buys': [], 'avg_price': None}

    last_summary_time = time.time()

    while True:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            balances = get_balance()
            fiat_balance = float(balances.get("Z" + QUOTE, 0) or 0)
            per_asset_balances = {}
            for base, info in resolved.items():
                base_asset_code = info.get('base_asset') or ''
                # Kraken balance keys may be like 'XXBT' etc. We just pick by name if present
                bal = balances.get(base_asset_code)
                # fallback: try by common prefixes mapping (Z = fiat prefix)
                if bal is None:
                    # try with asset itself
                    bal = balances.get(base)
                per_asset_balances[base] = float(bal or 0)

            # compute signals
            asset_signals = {}
            for base, info in resolved.items():
                try:
                    df = fetch_ohlc(info['pair'], interval=OHLC_INTERVAL, count=OHLC_COUNT)
                    asset_signals[base] = generate_scalp_signal(df)
                except Exception as e:
                    logger.warning("OHLC fail %s: %s", base, e)
                    asset_signals[base] = None

            # Log cycle to Discord (no @everyone)
            cycle_msg = f"{timestamp} | Signals: {asset_signals} | Fiat: {fiat_balance:.2f} {QUOTE} | Balances: {per_asset_balances}"
            logger.info(cycle_msg)
            send_discord(cycle_msg, ping_everyone=False)

            executed_any = False

            # iterate assets
            for base, signal in asset_signals.items():
                pair = resolved[base]['pair']
                balance = per_asset_balances.get(base, 0.0)
                info = last_action.get(base, {})
                # ensure structure
                buys = info.get('buys') if isinstance(info.get('buys'), list) else []
                avg_price = info.get('avg_price') if info.get('avg_price') is not None else (sum(buys)/len(buys) if buys else None)

                # -------- BUY logic --------
                if signal == 'buy':
                    # check max buy count
                    if len(buys) >= MAX_BUYS_PER_COIN:
                        msg = f"{timestamp} | Skipping BUY {base}: reached max {MAX_BUYS_PER_COIN} buys (open buys={len(buys)})"
                        logger.info(msg); send_discord(msg)
                    elif fiat_balance < 1:
                        msg = f"{timestamp} | Skipping BUY {base}: insufficient fiat ({fiat_balance:.2f} {QUOTE})"
                        logger.info(msg); send_discord(msg)
                    else:
                        eur_amount = min(TRADE_EUR, fiat_balance)
                        resp, min_vol, reason = place_market_order(pair, 'buy', eur_amount)
                        if resp:
                            # record executed buy price
                            try:
                                price = get_price(pair)
                            except Exception:
                                price = None
                            if price is None:
                                # if price missing, attempt to derive from order result if possible
                                price = None
                                try:
                                    # attempt to parse resp for price from Kraken response
                                    descr = resp.get('result', {}).get('descr', {})
                                    if isinstance(descr, dict):
                                        price = float(descr.get('price') or descr.get('close') or 0) or None
                                except Exception:
                                    price = None
                            if price:
                                buys.append(price)
                                avg_price = sum(buys) / len(buys)
                                info['buys'] = buys
                                info['avg_price'] = avg_price
                            else:
                                # fallback: store buy as 0 so we still count the buy (rare)
                                buys.append(0.0)
                                info['buys'] = buys
                                info['avg_price'] = avg_price
                            fiat_balance -= eur_amount
                            executed_any = True
                            msg = f"🚀 BUY {base} executed at {price if price else 'unknown'} {QUOTE} | Buys: {len(buys)} | Avg: {avg_price if avg_price else '-'}"
                            logger.info("%s | %s %s", timestamp, msg, resp)
                            send_discord(msg, ping_everyone=True)
                            send_telegram(msg)
                        else:
                            # explain why buy didn't occur
                            msg = f"{timestamp} | Skipping BUY {base}: {reason}"
                            logger.info(msg); send_discord(msg)

                # -------- SELL logic: sell all if profit target hit on average entry --------
                elif signal == 'sell' and buys:
                    # ensure avg_price exists
                    if avg_price is None:
                        msg = f"{timestamp} | Skipping SELL {base}: no stored avg entry price"
                        logger.info(msg); send_discord(msg)
                    else:
                        try:
                            price = get_price(pair)
                        except Exception as e:
                            msg = f"{timestamp} | No price for {base}: {e}"
                            logger.warning(msg); send_discord(msg)
                            price = None

                        if price is None:
                            # can't evaluate
                            continue

                        # compute target based on average price
                        target_price = avg_price * (1 + MIN_PROFIT)
                        # Provide detailed reasons on skip
                        if balance <= 0:
                            msg = f"{timestamp} | Skipping SELL {base}: balance is zero (balance={balance})"
                            logger.info(msg); send_discord(msg)
                        elif price < avg_price:
                            msg = f"{timestamp} | Skipping SELL {base}: price {price:.6f} < avg entry {avg_price:.6f}"
                            logger.info(msg); send_discord(msg)
                        elif price < target_price:
                            msg = f"{timestamp} | Skipping SELL {base}: price {price:.6f} < target profit {target_price:.6f}"
                            logger.info(msg); send_discord(msg)
                        else:
                            # proceed to sell ALL open exposure (sell amount equals holdings)
                            eur_equivalent = balance * price
                            # cap eur amount to TRADE_EUR * number_of_buys so we don't exceed expected chunking
                            eur_amount = min(eur_equivalent, TRADE_EUR * len(buys))
                            resp, min_vol, reason = place_market_order(pair, 'sell', eur_amount)
                            if resp:
                                # compute realized profit: approx = (sell_proceeds - invested)
                                # invested ~ TRADE_EUR * number_of_buys (we used that per buy)
                                invested = TRADE_EUR * len(buys)
                                proceeds = eur_amount
                                profit_eur = proceeds - invested
                                profit_log['realized_eur'] = profit_log.get('realized_eur', 0.0) + profit_eur
                                profit_log['trades'] = profit_log.get('trades', 0) + 1
                                save_profit_log(profit_log)

                                msg = f"💰 SELL {base} executed at {price:.6f} {QUOTE} | Proceeds: {proceeds:.2f} € | Invested: {invested:.2f} € | P/L: {profit_eur:+.2f} €"
                                logger.info("%s | %s %s", timestamp, msg, resp)
                                send_discord(msg, ping_everyone=True)
                                send_telegram(msg)
                                # reset buys
                                info['buys'] = []
                                info['avg_price'] = None
                                executed_any = True
                            else:
                                msg = f"{timestamp} | Skipping SELL {base}: {reason}"
                                logger.info(msg); send_discord(msg)

                # store back
                last_action[base] = {'buys': info.get('buys', []), 'avg_price': info.get('avg_price')}

            # Hourly summary with @everyone and lifetime profit
            if time.time() - last_summary_time >= 3600:
                summary = f"⏱ Hourly Summary | Fiat: {fiat_balance:.2f} {QUOTE}\n"
                for base, info in last_action.items():
                    b = info.get('buys', []) or []
                    avg = info.get('avg_price') if info.get('avg_price') else "-"
                    summary += f"{base}: {len(b)} open buys, avg entry {avg}\n"
                rp = profit_log.get('realized_eur', 0.0)
                trades = profit_log.get('trades', 0)
                summary += f"Lifetime realized P/L: {rp:.2f} EUR across {trades} closed trades"
                send_discord(summary, ping_everyone=True)
                send_telegram(summary)
                last_summary_time = time.time()

            # persist state if anything executed
            if executed_any:
                save_last_action(last_action)

        except Exception as e:
            logger.exception("%s | Error in main loop: %s", timestamp, e)
            send_discord(f"⚠️ Error in main loop: {e}")
            send_telegram(f"⚠️ Error in main loop: {e}")

        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()

