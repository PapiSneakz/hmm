# kraken_bot.py
"""
Kraken scalping bot with multiple buys, average entry, profit-target selling,
robust error handling, retry/backoff, Telegram/Discord alerts, hourly summary,
and lifetime profit tracking.

Environment variables expected (put in .env):
- KRAKEN_API_KEY
- KRAKEN_API_SECRET
- TELEGRAM_TOKEN (optional)
- TELEGRAM_CHAT_ID (optional)
- DISCORD_WEBHOOK (optional)

Behavior highlights:
- Assets: ETH, DOGE, XRP (use ASSETS)
- Max 4 buys per asset, up to TRADE_EUR each buy
- Sells entire position when average entry + MIN_PROFIT is hit
- Sends Discord messages each cycle (no @everyone) and @everyone on buy/sell and hourly summary
- Persists last_action.json and lifetime_profit.json
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

TRADE_EUR = 50.0           # EUR per buy
MAX_BUYS_PER_ASSET = 4     # max buys per coin
MIN_PROFIT = 0.012         # 1.2% profit target

SHORT_EMA = 5
LONG_EMA = 20
OHLC_INTERVAL = 1
OHLC_COUNT = 200

POLL_INTERVAL = 30         # seconds
LAST_ACTION_FILE = "last_action.json"
LIFETIME_FILE = "lifetime_profit.json"

# Hourly summary (sends @everyone)
SUMMARY_INTERVAL = 3600  # seconds

# ---------------- API KEYS ----------------
API_KEY = os.getenv("KRAKEN_API_KEY")
API_SECRET = os.getenv("KRAKEN_API_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")

API_BASE = "https://api.kraken.com"

# Retry/backoff
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0
BACKOFF_FACTOR = 2.0
JITTER = 0.3

ALERT_FAILURE_THRESHOLD = 6

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("kraken_bot")

# Caches & counters
price_cache = {}
ordermin_cache = {}
consecutive_public_errors = 0
consecutive_private_errors = 0

# ---------------- TELEGRAM / DISCORD ----------------
def send_telegram(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=10)
    except Exception as e:
        logger.warning("⚠️ Telegram send failed: %s", e)

def send_discord(msg: str, ping_everyone: bool = False):
    if not DISCORD_WEBHOOK:
        return
    # Discord requires the message to be plain text content for webhooks
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

    # Exhausted retries -> update counters & alert
    if is_private:
        consecutive_private_errors += 1
        logger.error("Private API failed %d times", consecutive_private_errors)
        if consecutive_private_errors >= ALERT_FAILURE_THRESHOLD:
            send_telegram(f"⚠️ Kraken private API failing repeatedly ({consecutive_private_errors} errors)")
    else:
        consecutive_public_errors += 1
        logger.error("Public API failed %d times", consecutive_public_errors)
        if consecutive_public_errors >= ALERT_FAILURE_THRESHOLD:
            send_telegram(f"⚠️ Kraken public API failing repeatedly ({consecutive_public_errors} errors)")
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
    if not isinstance(resp, dict) or resp.get("error"):
        raise RuntimeError(f"OHLC error: {resp.get('error')}")
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
    if not isinstance(resp, dict) or resp.get("error"):
        raise RuntimeError(f"Balance error: {resp.get('error')}")
    return resp.get("result", {})

def get_all_assetpairs():
    resp = public_get("AssetPairs")
    if resp.get("error"):
        raise RuntimeError(f"AssetPairs error: {resp.get('error')}")
    return resp.get("result", {})

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
            # fallback search
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
    """
    Place market order by EUR amount.
    Returns (response, min_vol, fail_reason). On success fail_reason = "".
    """
    # get price (may use cache)
    try:
        price = get_price(pair)
    except Exception as e:
        reason = f"Failed to fetch price: {e}"
        return None, get_min_order_volume(pair), reason

    vol = round(eur_amount / price, 8)
    try:
        min_vol = get_min_order_volume(pair)
    except Exception as e:
        min_vol = 0
        logger.warning("Could not get ordemin for %s: %s", pair, e)

    if vol < min_vol:
        return None, min_vol, f"Order volume too small: {vol:.8f} < min {min_vol:.8f}"

    order = {
        "ordertype": "market",
        "type": side,
        "volume": str(vol),
        "pair": pair
    }
    try:
        resp = private_post("AddOrder", order)
    except Exception as e:
        return None, min_vol, f"AddOrder failed: {e}"

    if isinstance(resp, dict) and resp.get('error'):
        return None, min_vol, f"AddOrder returned errors: {resp.get('error')}"
    return resp, min_vol, ""

# ---------------- LAST ACTIONS & LIFETIME PROFIT ----------------
def load_json_file_safely(path: str, default):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning("Failed reading %s: %s", path, e)
    return default

def load_last_action():
    data = load_json_file_safely(LAST_ACTION_FILE, {})
    # ensure structure: dict with per-asset dict containing 'buys' list and 'avg_price' maybe
    sanitized = {}
    for base in ASSETS:
        val = data.get(base, {})
        if isinstance(val, dict):
            buys = val.get("buys") or []
            # sanitize buys list -> list of dicts {price, eur, vol}
            buys2 = []
            for b in buys:
                if isinstance(b, dict) and "price" in b and "eur" in b and "vol" in b:
                    buys2.append(b)
                else:
                    # if legacy format (list of prices), convert to dict with eur=TRADE_EUR, vol estimate
                    try:
                        price = float(b)
                        vol = round(TRADE_EUR / price, 8)
                        buys2.append({"price": price, "eur": TRADE_EUR, "vol": vol})
                    except Exception:
                        continue
            avg = val.get("avg_price")
            if not avg and buys2:
                avg = sum(b["price"] for b in buys2) / len(buys2)
            sanitized[base] = {"buys": buys2, "avg_price": avg}
        else:
            sanitized[base] = {"buys": [], "avg_price": None}
    return sanitized

def save_last_action(actions: dict):
    try:
        with open(LAST_ACTION_FILE, "w") as f:
            json.dump(actions, f)
    except Exception as e:
        logger.warning("Failed to save last_action: %s", e)

def load_lifetime_profit():
    data = load_json_file_safely(LIFETIME_FILE, {"realized_eur": 0.0})
    # ensure float
    try:
        data["realized_eur"] = float(data.get("realized_eur", 0.0))
    except Exception:
        data["realized_eur"] = 0.0
    return data

def save_lifetime_profit(data: dict):
    try:
        with open(LIFETIME_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        logger.warning("Failed to save lifetime_profit: %s", e)

# ---------------- MAIN LOOP ----------------
def main():
    send_telegram("🤖 Kraken scalping bot started with profit protection!")
    send_discord("🤖 Kraken scalping bot started with profit protection!", ping_everyone=True)
    logger.info("Running scalping bot...")
    last_action = load_last_action()
    lifetime = load_lifetime_profit()

    try:
        resolved = resolve_pairs(ASSETS, QUOTE)
    except Exception as e:
        logger.exception("Error resolving pairs on startup: %s", e)
        send_telegram(f"⚠️ Error resolving pairs: {e}")
        send_discord(f"⚠️ Error resolving pairs: {e}")
        return

    logger.info("Resolved pairs:")
    for base, info in resolved.items():
        logger.info("  %s -> pair %s", base, info['pair'])

    last_summary_time = time.time()

    while True:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            # fetch balances
            balances = get_balance()
            # fiat on Kraken uses 'Z' + currency (EUR)
            fiat_key = "Z" + QUOTE
            fiat_balance = float(balances.get(fiat_key, 0) or 0)
            # per-asset balances: use info['base_asset'] as returned by Kraken (e.g., XETH)
            per_asset_balances = {}
            for base, info in resolved.items():
                key = info.get('base_asset')
                # balances may use codes like 'XETH' or 'ETH' depending; fallback to base symbol
                val = balances.get(key) if key in balances else balances.get(base)
                per_asset_balances[base] = float(val or 0)

            # generate signals
            asset_signals = {}
            for base, info in resolved.items():
                try:
                    df = fetch_ohlc(info['pair'])
                    asset_signals[base] = generate_scalp_signal(df)
                except Exception as e:
                    logger.warning("Failed to fetch OHLC for %s: %s (will not block trading)", base, e)
                    asset_signals[base] = None

            # Log cycle (no @everyone)
            cycle_msg = f"{timestamp} | Signals: {asset_signals} | Fiat: {fiat_balance:.2f} {QUOTE} | Balances: {per_asset_balances}"
            logger.info(cycle_msg)
            send_discord(cycle_msg, ping_everyone=False)

            executed_any = False

            for base, signal in asset_signals.items():
                pair = resolved[base]['pair']
                balance = per_asset_balances.get(base, 0)
                last = last_action.get(base, {"buys": [], "avg_price": None})
                # ensure structure
                if "buys" not in last:
                    last["buys"] = []
                if "avg_price" not in last:
                    last["avg_price"] = None

                # -------- BUY --------
                if signal == 'buy':
                    if len(last["buys"]) >= MAX_BUYS_PER_ASSET:
                        msg = f"{timestamp} | Skipping BUY {base}: reached max {MAX_BUYS_PER_ASSET} buys"
                        logger.info(msg)
                        send_discord(msg)
                        continue
                    if fiat_balance < 1:
                        msg = f"{timestamp} | Skipping BUY {base}: insufficient fiat ({fiat_balance:.2f} {QUOTE})"
                        logger.info(msg)
                        send_discord(msg)
                        continue

                    eur_amount = min(TRADE_EUR, fiat_balance)
                    resp, min_vol, reason = place_market_order(pair, 'buy', eur_amount)
                    if resp:
                        # compute volume from eur_amount/price (we might also parse resp for executed volume if needed)
                        price = None
                        try:
                            price = get_price(pair)
                        except Exception:
                            # fallback if price unavailable (should not happen)
                            logger.warning("Could not read price after buy for %s", pair)
                        vol = round(eur_amount / price, 8) if price and price > 0 else 0.0
                        buy_entry = {"price": price, "eur": eur_amount, "vol": vol}
                        last["buys"].append(buy_entry)
                        # recompute avg price weighted by volume (preferred)
                        total_vol = sum(b["vol"] for b in last["buys"] if isinstance(b, dict))
                        if total_vol > 0:
                            weighted = sum(b["price"] * b["vol"] for b in last["buys"])
                            last["avg_price"] = weighted / total_vol
                        else:
                            last["avg_price"] = sum(b["price"] for b in last["buys"]) / len(last["buys"])
                        fiat_balance -= eur_amount
                        executed_any = True
                        # compute small balance update displayed in message
                        msg = f"🚀 BUY {base} at {price:.6f} {QUOTE} | Avg: {last['avg_price']:.6f} | Buys: {len(last['buys'])}"
                        logger.info("%s | %s", timestamp, msg)
                        # send @everyone for executed trade
                        send_discord(msg, ping_everyone=True)
                        # also Telegram
                        send_telegram(msg)
                    else:
                        # explain reason
                        msg = f"{timestamp} | Skipping BUY {base}: {reason}"
                        logger.info(msg)
                        send_discord(msg)

                # -------- SELL --------
                elif signal == 'sell' and last.get('buys'):
                    avg_price = last.get('avg_price')
                    # fetch current price
                    price = None
                    try:
                        price = get_price(pair)
                    except Exception as e:
                        msg = f"{timestamp} | No price for {base}: {e}"
                        logger.warning(msg)
                        send_discord(msg)
                        continue

                    # checks with verbose reasons
                    if balance <= 0:
                        msg = f"{timestamp} | Skipping SELL {base}: balance is zero"
                        logger.info(msg)
                        send_discord(msg)
                        continue
                    if not avg_price:
                        msg = f"{timestamp} | Skipping SELL {base}: no stored average buy price"
                        logger.info(msg)
                        send_discord(msg)
                        continue

                    target_price = avg_price * (1 + MIN_PROFIT)
                    if price < avg_price:
                        msg = f"{timestamp} | Skipping SELL {base}: price {price:.6f} < avg buy {avg_price:.6f}"
                        logger.info(msg)
                        send_discord(msg)
                        continue
                    if price < target_price:
                        msg = f"{timestamp} | Skipping SELL {base}: price {price:.6f} < target profit {target_price:.6f}"
                        logger.info(msg)
                        send_discord(msg)
                        continue

                    # price >= target -> attempt to sell entire position (or up to TRADE_EUR * len(buys))
                    total_vol = sum(b["vol"] for b in last["buys"])
                    if total_vol <= 0:
                        msg = f"{timestamp} | Skipping SELL {base}: total volume calculated zero"
                        logger.info(msg)
                        send_discord(msg)
                        continue
                    # compute EUR equivalent of full position, cap to TRADE_EUR * len(buys)
                    eur_equivalent = total_vol * price
                    eur_amount = min(eur_equivalent, TRADE_EUR * max(1, len(last["buys"])))
                    resp, min_vol, reason = place_market_order(pair, 'sell', eur_amount)
                    if resp:
                        # compute realized profit for portion sold
                        sell_vol = eur_amount / price if price > 0 else 0.0
                        # proportion of total volume sold
                        proportion = sell_vol / total_vol if total_vol > 0 else 0.0
                        total_cost = sum(b["eur"] for b in last["buys"])
                        cost_basis = total_cost * proportion
                        proceeds = sell_vol * price
                        profit = proceeds - cost_basis
                        lifetime["realized_eur"] = lifetime.get("realized_eur", 0.0) + profit
                        save_lifetime_profit(lifetime)
                        # clear buys (we sell all at once per your rules)
                        last["buys"] = []
                        last["avg_price"] = None
                        executed_any = True
                        # nice @everyone message including profit/loss
                        sign = "+" if profit >= 0 else "-"
                        msg = f"💰 SELL {base} at {price:.6f} {QUOTE} | Realized {sign}{abs(profit):.2f} {QUOTE} | Lifetime: {lifetime['realized_eur']:.2f} {QUOTE}"
                        logger.info("%s | %s", timestamp, msg)
                        send_discord(msg, ping_everyone=True)
                        send_telegram(msg)
                    else:
                        msg = f"{timestamp} | Skipping SELL {base}: {reason}"
                        logger.info(msg)
                        send_discord(msg)

                # save per-asset last state
                last_action[base] = last

            # hourly summary
            if time.time() - last_summary_time >= SUMMARY_INTERVAL:
                summary_msg = f"⏱ Hourly Summary | Fiat: {fiat_balance:.2f} {QUOTE}\n"
                for base, info in last_action.items():
                    buys = info.get('buys', [])
                    avg = info.get('avg_price')
                    summary_msg += f"{base}: {len(buys)} open buys, avg entry {avg if avg else '-'}\n"
                summary_msg = summary_msg.strip()
                send_discord(summary_msg, ping_everyone=True)
                last_summary_time = time.time()

            # persist state if trades executed (or always to be safe)
            save_last_action(last_action)

            if not executed_any:
                # send small cycle-no-trades message to maintain visibility (you currently prefer cycles every loop)
                pass  # already sent cycle_msg earlier

        except Exception as e:
            logger.exception("%s | Error in main loop: %s", timestamp, e)
            send_discord(f"⚠️ Error in main loop: {e}")

        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()

