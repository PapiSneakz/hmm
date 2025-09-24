import os
import requests

TELEGRAM_TOKEN = "8438533380:AAEFInXrJzGV2q3i2DXQf27MTW6HTDTqVZo"
TELEGRAM_CHAT_ID = "1646255457"

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    r = requests.post(url, data=payload)
    print(r.status_code, r.text)

send_telegram("✅ Test message from bot")
