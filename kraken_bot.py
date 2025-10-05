# kraken_bot.py
"""
Kraken scalping bot with multiple buys, average entry, profit-target selling,
robust error handling, retry/backoff, Telegram/Discord alerts, and hourly summary.

Features added:
- Detailed skip reasons sent to Discord (non-@everyone).
- @everyone ping on successful BUY, SELL, and hourly summary.
- Multi-buy support (max 4 buys), average entry calculation.
- Realized profit calculation persisted to profit_log.json (lifetime).
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

TRADE_EUR = 50.0           # Max EUR per trade per coin (per buy)
MIN_PROFIT = 0.012         # 1.2% profit target

SHORT_EMA = 5
LONG_EMA = 20
OHLC_INTERVAL = 1
OHLC_COUNT = 200

POLL_INTERVAL = 30         # seconds between checks
LAST_ACTION_FILE = "last_action.json"
PROFIT_FILE = "profit_log.json"

# ---------------- API KEYS ----------------
API_KEY = os.getenv("KRAKEN_API_KEY")
API_SECRET = os.getenv("KRAKEN_API_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")

API_BASE = "https://api.kraken.com"

# Retry settings
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0
BACKOFF_FACTOR = 2.0
JITTER = 0.3

ALERT_FAILURE_THRESHOLD = 6

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("kraken_bot")

# ---------------- STATE ----------------
price_cache = {}
ordermin_cache = {}
consecutive_public_errors = 0
consecutive_private_errors = 0

# persistent profit tracking (loaded from file)
def load_profit():
    if os.path.exists(PROFIT_FILE):
        try:
            with open(PROFIT_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    # default structure
    return {"total_realized_eur": 0.0, "history": []}

def save_profit(data):
    try:
        with open(PROFIT_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        logger.warning("Failed to save profit file: %s", e)

profit_data = load_profit()

# ---------------- TELEGRAM / DISCORD ----------------
def send_telegram(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram not configured")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=10)
    except Exception as e:
        logger.warning("⚠️ Telegram send failed: %s", e)

def send_discord(msg: str, ping_everyone: bool = False):
    if not DISCORD_WEBHOOK:
        logger.debug("Discord webhook not configured")
        return
    content = f"@everyone {msg}" if ping_everyone else msg
    try:
        requests.post(DISCORD_WEBHOOK, json={"content": content}, timeout=10)
    except Exception as e:
        logger.warning("⚠️ Discord send failed: %s", e)

# ---------------- RETRY ----------------
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
            logger.exception("Unexpected error: %s", e)
            _sleep_backoff(attempt)
    # exhausted
    if is_private:
        consecutive_private_errors += 1
        if consecutive_private_errors >= ALERT_FAILURE_THRESHOLD:
            send_telegram(f"⚠️ Kraken private API failing repeatedly ({consecutive_private_errors})")
    else:
        consecutive_public_errors += 1
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
    headers = {"API-Key": API_KEY, "API-Sign": signature}
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
    if resp.get('error'):
        logger.warning("OHLC API returned errors: %s", resp.get('error'))
        raise RuntimeError("OHLC returned error: %s" % resp.get('error'))
    result = resp.get('result', {})
    data = next((v for k, v in result.items() if k != "last"), None)
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
    if not isinstance(resp, dict):
        raise RuntimeError("Unexpected Balance response")
    if resp.get('error'):
        raise RuntimeError("Balance error: %s" % resp.get('error'))
    return resp.get('result', {})

def get_all_assetpairs():
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
        if pair in price_cache:
            logger.info("Using cached price for %s: %s", pair, price_cache[pair])
            return price_cache[pair]
        raise

# place_market_order returns (resp, min_vol, reason, executed_vol)
def place_market_order(pair: str, side: str, eur_amount: float) -> Tuple[Optional[dict], float, str, float]:
    """Place a market order by EUR amount. Returns (response, min_vol, fail_reason, executed_vol)
       executed_vol is the volume we expect was bought/sold (EUR/price).
    """
    # obtain price (may use cache)
    try:
        price = get_price(pair)
    except Exception as e:
        try:
            minv = get_min_order_volume(pair)
        except Exception:
            minv = 0
        return None, minv, f"Failed to fetch price: {e}", 0.0

    vol = round(eur_amount / price, 8)
    try:
        min_vol = get_min_order_volume(pair)
    except Exception as e:
        logger.warning("Cannot determine ordemin for %s: %s", pair, e)
        min_vol = 0

    if vol < min_vol:
        return None, min_vol, f"Order volume too small: {vol:.8f} < min {min_vol:.8f}", vol

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
        return None, min_vol, f"Failed to place order: {e}", vol

    if resp.get('error'):
        return None, min_vol, f"AddOrder returned errors: {resp.get('error')}", vol

    # success — we assume executed volume equals vol we requested
    return resp, min_vol, "", vol

# ---------------- LAST ACTIONS ----------------
def load_last_action():
    if os.path.exists(LAST_ACTION_FILE):
        with open(LAST_ACTION_FILE, "r") as f:
            try:
                data = json.load(f)
                # normalize old format to new: ensure buys is list of dicts
                for k, v in data.items():
                    if isinstance(v, dict) and 'buys' in v:
                        # keep as-is
                        continue
                    # older entries may have {"side":"buy"} — convert to buys list empty
                    data[k] = v if isinstance(v, dict) else {}
                        # fallback
                return data
            except Exception:
                return {}
    return {}

def save_last_action(actions):
    try:
        with open(LAST_ACTION_FILE, "w") as f:
            json.dump(actions, f)
    except Exception as e:
        logger.warning("Failed to save last_action: %s", e)

# ---------------- HELPERS ----------------
def weighted_avg_price(buys):
    """buys is list of {'price': float, 'vol': float}. Return weighted avg price or None."""
    if not buys:
        return None
    total_vol = sum(b['vol'] for b in buys)
    if total_vol == 0:
        return None
    return sum(b['price'] * b['vol'] for b in buys) / total_vol

def record_realized_profit(base: str, profit_eur: float, sell_price: float, avg_entry: float, vol: float):
    global profit_data
    profit_data['total_realized_eur'] = round(profit_data.get('total_realized_eur', 0.0) + profit_eur, 8)
    profit_data['history'].append({
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "asset": base,
        "profit_eur": round(profit_eur, 8),
        "sell_price": sell_price,
        "avg_entry": avg_entry,
        "volume": vol
    })
    save_profit(profit_data)

# ---------------- MAIN LOOP ----------------
def main():
    send_discord("🤖 Kraken scalping bot started with profit protection!", ping_everyone=True)
    logger.info("Running scalping bot...")
    last_action = load_last_action()
    resolved = resolve_pairs(ASSETS, QUOTE)
    last_summary_time = time.time()

    # normalize last_action entries
    for base in ASSETS:
        info = last_action.get(base, {})
        if not isinstance(info, dict):
            info = {}
        if 'buys' not in info:
            info['buys'] = []  # each buy: {'price': float, 'vol': float, 'eur': float}
        last_action[base] = info

    while True:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            balances = get_balance()
            fiat_balance = float(balances.get("Z" + QUOTE, 0) or 0)
            per_asset_balances = {b: float(balances.get(resolved[b]['base_asset'], 0) or 0) for b in resolved}

            asset_signals = {}
            for base, info in resolved.items():
                try:
                    df = fetch_ohlc(info['pair'], interval=OHLC_INTERVAL, count=OHLC_COUNT)
                    asset_signals[base] = generate_scalp_signal(df)
                except Exception as e:
                    logger.warning("OHLC fail %s: %s", base, e)
                    asset_signals[base] = None

            # Log cycle -> send to Discord (no @everyone)
            cycle_msg = f"{timestamp} | Signals: {asset_signals} | Fiat: {fiat_balance:.2f} {QUOTE} | Balances: {per_asset_balances}"
            logger.info(cycle_msg)
            send_discord(cycle_msg, ping_everyone=False)

            executed_any = False

            for base, signal in asset_signals.items():
                pair = resolved[base]['pair']
                balance = per_asset_balances.get(base, 0)
                last = last_action.get(base, {"buys": []})
                buys = last.get('buys', [])

                # -------- BUY --------
                if signal == 'buy':
                    if len(buys) >= 4:
                        msg = f"{timestamp} | Skipping BUY {base}: reached max 4 buys"
                        logger.info(msg)
                        send_discord(msg)
                        continue
                    if fiat_balance < 1:
                        msg = f"{timestamp} | Skipping BUY {base}: insufficient fiat ({fiat_balance:.2f} {QUOTE})"
                        logger.info(msg)
                        send_discord(msg)
                        continue

                    eur_amount = min(TRADE_EUR, fiat_balance)
                    resp, min_vol, reason, vol = place_market_order(pair, 'buy', eur_amount)
                    if resp:
                        # store the executed volume & price (we used vol = eur/price)
                        price = get_price(pair)
                        buy_entry = {"price": price, "vol": vol, "eur": round(vol * price, 8)}
                        buys.append(buy_entry)
                        last['buys'] = buys
                        last['avg_price'] = weighted_avg_price(buys)
                        fiat_balance -= eur_amount
                        executed_any = True
                        msg = f"🚀 BUY {base} at {price:.6f} | Avg: {last['avg_price']:.6f} | Buys: {len(buys)}"
                        logger.info("%s | %s", timestamp, msg)
                        send_telegram(msg)
                        send_discord(msg, ping_everyone=True)
                    else:
                        msg = f"{timestamp} | Skipping BUY {base}: {reason}"
                        logger.info(msg)
                        send_discord(msg)

                # -------- SELL --------
                elif signal == 'sell' and buys:
                    avg_price = weighted_avg_price(buys)
                    if avg_price is None:
                        msg = f"{timestamp} | Skipping SELL {base}: no avg entry price"
                        logger.info(msg)
                        send_discord(msg)
                        continue

                    try:
                        price = get_price(pair)
                    except Exception as e:
                        msg = f"{timestamp} | No price for {base}: {e}"
                        logger.warning(msg)
                        send_discord(msg)
                        continue

                    if balance <= 0:
                        msg = f"{timestamp} | Skipping SELL {base}: balance is zero"
                        logger.info(msg)
                        send_discord(msg)
                        continue

                    target_price = avg_price * (1 + MIN_PROFIT)
                    # give detailed skip reasons similar to original
                    if price < avg_price:
                        msg = f"{timestamp} | Skipping SELL {base}: price {price:.6f} < avg entry {avg_price:.6f}"
                        logger.info(msg)
                        send_discord(msg)
                        continue
                    if price < target_price:
                        msg = f"{timestamp} | Skipping SELL {base}: price {price:.6f} < target profit {target_price:.6f}"
                        logger.info(msg)
                        send_discord(msg)
                        continue

                    # Execute sell of total holdings up to TRADE_EUR * number_of_buys (or all)
                    total_vol = sum(b['vol'] for b in buys)
                    eur_equivalent = total_vol * price
                    # choose to sell up to all open volume (you wanted to "Sell all at once when profit target hit")
                    eur_amount = min(eur_equivalent, TRADE_EUR * len(buys))
                    resp, min_vol, reason, sold_vol = place_market_order(pair, 'sell', eur_amount)
                    if resp:
                        # compute realized profit: assume we sold proportionally across buys
                        # sold_vol is volume sold (approx). We'll proportionally take from buys from oldest to newest.
                        remaining_to_sell = sold_vol
                        cost_basis = 0.0
                        sold_volume_accounted = 0.0
                        # iterate buys to compute cost basis for the portion sold
                        new_buys = []
                        for b in buys:
                            if remaining_to_sell <= 0:
                                new_buys.append(b)
                                continue
                            take = min(b['vol'], remaining_to_sell)
                            cost_basis += take * b['price']
                            sold_volume_accounted += take
                            remaining_to_sell -= take
                            leftover = b['vol'] - take
                            if leftover > 0:
                                new_buys.append({"price": b['price'], "vol": leftover, "eur": leftover * b['price']})
                        # If we didn't sell full requested (edge cases), sold_volume_accounted may be < sold_vol; handle gracefully.
                        if sold_volume_accounted == 0:
                            # nothing sold (shouldn't happen if resp success), but handle
                            msg = f"{timestamp} | Sell reported success but sold volume accounted is zero for {base}"
                            logger.warning(msg)
                            send_discord(msg)
                        else:
                            proceeds = sold_volume_accounted * price
                            profit_eur = proceeds - cost_basis
                            profit_pct = (price / (cost_basis / sold_volume_accounted) - 1) * 100 if cost_basis > 0 else 0.0
                            # record realized profit
                            record_realized_profit(base, profit_eur, price, (cost_basis / sold_volume_accounted), sold_volume_accounted)
                            msg = f"💰 SELL {base} at {price:.6f} | Realized: {profit_eur:.2f} EUR ({profit_pct:.2f}%)"
                            logger.info("%s | %s", timestamp, msg)
                            send_telegram(msg)
                            send_discord(msg, ping_everyone=True)
                            # update buys to leftover
                            last['buys'] = new_buys
                            last['avg_price'] = weighted_avg_price(new_buys)
                            executed_any = True
                    else:
                        msg = f"{timestamp} | Skipping SELL {base}: {reason}"
                        logger.info(msg)
                        send_discord(msg)

                # save per-asset state
                last_action[base] = last

            # hourly summary (ping @everyone)
            if time.time() - last_summary_time >= 3600:
                summary_msg = f"⏱ Hourly Summary | Fiat: {fiat_balance:.2f} {QUOTE} | Total realized: {profit_data.get('total_realized_eur', 0.0):.2f} EUR\n"
                for base, info in last_action.items():
                    buys = info.get('buys', [])
                    avg = weighted_avg_price(buys)
                    summary_msg += f"{base}: {len(buys)} open buys, avg entry {avg if avg else '-'}\n"
                send_discord(summary_msg.strip(), ping_everyone=True)
                last_summary_time = time.time()

            if executed_any:
                save_last_action(last_action)
            else:
                # also log and send a short "no trades executed" to Discord (non-ping) to match your desired cycle reporting
                msg = f"{timestamp} | No trades executed."
                logger.info(msg)
                send_discord(msg, ping_everyone=False)

        except Exception as e:
            logger.exception("%s | Error in main loop: %s", timestamp, e)
            send_discord(f"⚠️ Error in main loop: {e}")

        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
