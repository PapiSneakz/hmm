import os
import time
import hmac
import hashlib
import base64
from urllib.parse import urlencode
import requests
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()
API_KEY = os.getenv("KRAKEN_API_KEY")
API_SECRET = os.getenv("KRAKEN_API_SECRET")
API_BASE = "https://api.kraken.com"
API_VERSION = "0"

FEE_BUFFER = 0.995  # 0.5% buffer for fees

def _sign(path: str, data: dict, secret: str):
    post_data = urlencode(data)
    encoded = (str(data['nonce']) + post_data).encode()
    message = path.encode() + hashlib.sha256(encoded).digest()
    mac = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
    return base64.b64encode(mac.digest()).decode()

def private_post(endpoint: str, data: dict = {}):
    data['nonce'] = int(time.time() * 1000)
    signature = _sign(f"/{API_VERSION}/private/{endpoint}", data, API_SECRET)
    headers = {"API-Key": API_KEY, "API-Sign": signature}
    url = f"{API_BASE}/{API_VERSION}/private/{endpoint}"
    r = requests.post(url, headers=headers, data=data)
    return r.json()

def public_get(endpoint: str, params: dict = None):
    url = f"{API_BASE}/{API_VERSION}/public/{endpoint}"
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()

# 1️⃣ Get account balances
balances_resp = private_post("Balance")
balances = balances_resp.get("result", {})

# 2️⃣ Detect available fiat (USD or EUR)
if 'ZUSD' in balances and float(balances['ZUSD']) > 0:
    fiat_currency = 'ZUSD'
    pair_input = 'XBTUSD'
    available_balance = float(balances['ZUSD'])
elif 'ZEUR' in balances and float(balances['ZEUR']) > 0:
    fiat_currency = 'ZEUR'
    pair_input = 'XBTEUR'
    available_balance = float(balances['ZEUR'])
else:
    print("No USD or EUR balance available. Deposit funds first.")
    exit()

print(f"Detected balance: {available_balance} {fiat_currency}")
print(f"Trading pair: {pair_input}")

# 3️⃣ Fetch minimum order size and actual Kraken pair key
pair_info = public_get("AssetPairs", {"pair": pair_input})
pair_key = list(pair_info['result'].keys())[0]  # Kraken internal key
min_volume = float(pair_info['result'][pair_key]['ordermin'])
print(f"Minimum BTC volume for {pair_key}: {min_volume}")

# 4️⃣ Get current BTC price
ticker = public_get("Ticker", {"pair": pair_input})
price = float(ticker['result'][pair_key]['c'][0])
print(f"Current BTC price: {price}")

# 5️⃣ Calculate BTC volume with fee buffer
volume = round(available_balance * FEE_BUFFER / price, 6)
if volume < min_volume:
    print(f"Balance too low for minimum order ({min_volume} BTC).")
    exit()

print(f"Placing market BUY for {volume} BTC (~{available_balance} {fiat_currency})")

# 6️⃣ Place BUY order
buy_order = {
    "ordertype": "market",
    "type": "buy",
    "volume": str(volume),
    "pair": pair_key
}

buy_resp = private_post("AddOrder", buy_order)
if buy_resp['error']:
    print("BUY order rejected:", buy_resp['error'])
    exit()
else:
    print("BUY order executed:", buy_resp['result'])

# 7️⃣ Immediately place SELL order
print("Placing immediate SELL order to test...")
sell_order = {
    "ordertype": "market",
    "type": "sell",
    "volume": str(volume),
    "pair": pair_key
}

sell_resp = private_post("AddOrder", sell_order)
if sell_resp['error']:
    print("SELL order rejected:", sell_resp['error'])
else:
    print("SELL order executed:", sell_resp['result'])
