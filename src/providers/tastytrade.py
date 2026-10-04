import os
import aiohttp
import asyncio
from datetime import datetime
from .base import MarketDataSource

class TastytradeSource(MarketDataSource):
    def __init__(self):
        super().__init__("Tastytrade")
        self.client_id = os.getenv("TASTY_CLIENT_ID")
        self.client_secret = os.getenv("TASTY_CLIENT_SECRET")
        self.refresh_token = os.getenv("TASTY_REFRESH_TOKEN")
        self.access_token = None

    async def connect(self) -> bool:
        url = "https://api.tastyworks.com/oauth/token"
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        headers = {
            "User-Agent": "dunnavan-engine/1.0", 
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self.access_token = data.get("access_token")
                    self.is_connected = True
                    return True
                
                print(f"\n[ERROR] Tastytrade Auth Failed ({resp.status}): {await resp.text()}")
                return False

    async def stream_data(self, symbol: str, queue: asyncio.Queue):
        url = f"https://api.tastyworks.com/market-data/by-type?equity={symbol}"
        headers = {
            "Authorization": f"Bearer {self.access_token}", 
            "User-Agent": "dunnavan-engine/1.0"
        }
        
        async with aiohttp.ClientSession() as session:
            while True:
                async with session.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        raw = await resp.json()
                        print(raw)  # Inspecting the exact payload shape
                    else:
                        await queue.put({
                            "provider": self.provider_name,
                            "symbol": symbol,
                            "type": f"ERROR {resp.status}",
                            "bid": 0.0, "ask": 0.0,
                            "ts": datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        })
                await asyncio.sleep(1.0)
