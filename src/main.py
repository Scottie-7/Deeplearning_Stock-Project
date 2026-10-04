import asyncio
import os
import aiohttp
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

ENVIRONMENT = os.getenv("TASTY_ENV", "production").lower()
API_BASE = "https://api.cert.tastyworks.com" if ENVIRONMENT == "sandbox" else "https://api.tastyworks.com"

def display_market_data(payload: dict, console: Console, symbol: str):
    data_block = payload.get("data", {})
    items = data_block.get("items", [])
    
    if not items:
        console.print(f"[yellow]Warning: Received empty items list for {symbol}. Raw payload: {payload}[/yellow]")
        return
        
    data = items[0]
    volume_raw = data.get('volume', 0)
    volume_fmt = f"{float(volume_raw):,.0f}" if volume_raw else "N/A"
    
    table = Table(
        title=f"Live Market Data [{ENVIRONMENT.upper()}]: {data.get('symbol', symbol)}", 
        show_header=True, 
        header_style="bold green"
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta", justify="right")
    
    table.add_row("Last / Mark", f"{data.get('last', 'N/A')} / {data.get('mark', 'N/A')}")
    table.add_row("Bid x Size", f"{data.get('bid', 'N/A')} x {data.get('bid-size', 'N/A')}")
    table.add_row("Ask x Size", f"{data.get('ask', 'N/A')} x {data.get('ask-size', 'N/A')}")
    table.add_row("Day High", str(data.get('day-high-price') or data.get('day-high', 'N/A')))
    table.add_row("Day Low", str(data.get('day-low-price') or data.get('day-low', 'N/A')))
    table.add_row("Volume", volume_fmt)
    table.add_row("Updated", str(data.get('updated-at', 'N/A')))
    
    console.clear()
    console.print(table)

async def tastytrade_producer(queue: asyncio.Queue, symbol: str, console: Console):
    base_headers = {"User-Agent": "market-data-engine/1.0"}
    timeout = aiohttp.ClientTimeout(total=10)
    
    async with aiohttp.ClientSession(headers=base_headers, timeout=timeout) as session:
        auth_payload = {
            "grant_type": "refresh_token",
            "refresh_token": os.getenv("TASTY_REFRESH_TOKEN"),
            "client_secret": os.getenv("TASTY_CLIENT_SECRET"),
            "client_id": os.getenv("TASTY_CLIENT_ID")
        }
        
        if not auth_payload["refresh_token"]:
            console.print("[red]Error: TASTY_REFRESH_TOKEN missing in .env[/red]")
            await queue.put(None)
            return

        console.print(f"[cyan]Authenticating with Tastytrade ({ENVIRONMENT}) via OAuth...[/cyan]")
        try:
            async with session.post(f"{API_BASE}/oauth/token", json=auth_payload) as auth_req:
                if auth_req.status != 200:
                    error_text = await auth_req.text()
                    console.print(f"[red]OAuth Authentication failed ({auth_req.status}): {error_text}[/red]")
                    await queue.put(None)
                    return
                
                auth_response = await auth_req.json()
                access_token = auth_response["access_token"] 
                console.print("[green]Authentication successful.[/green]")
        except asyncio.TimeoutError:
            console.print("[red]Authentication timed out connecting to Tastytrade.[/red]")
            await queue.put(None)
            return
        except Exception as e:
            console.print(f"[red]Connection error during auth: {e}[/red]")
            await queue.put(None)
            return
            
        auth_headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json"
        }
        
        url = f"{API_BASE}/market-data/by-type?equity={symbol}"
        
        while True:
            try:
                async with session.get(url, headers=auth_headers) as data_req:
                    response_text = await data_req.text()
                    
                    if data_req.status == 200:
                        import json
                        payload = json.loads(response_text)
                        await queue.put({"provider": "tastytrade", "payload": payload, "symbol": symbol})
                    elif data_req.status == 429:
                        console.print("[yellow]Rate limited (429), backing off...[/yellow]")
                        await asyncio.sleep(5)
                    else:
                        console.print(f"[yellow]API Error {data_req.status}: {response_text}[/yellow]")
            except asyncio.TimeoutError:
                console.print("[yellow]Market data request timed out, retrying...[/yellow]")
            except Exception as e:
                console.print(f"[red]Data fetch error: {e}[/red]")
                        
            await asyncio.sleep(1)

async def dashboard_consumer(queue: asyncio.Queue, console: Console):
    while True:
        data_event = await queue.get()
        if data_event is None:
            break
            
        provider = data_event.get("provider")
        payload = data_event.get("payload")
        symbol = data_event.get("symbol", "IWM")
        
        if provider == "tastytrade":
            display_market_data(payload, console, symbol)
            
        queue.task_done()

async def main():
    console = Console()
    data_queue = asyncio.Queue()
    
    # Dynamic Ticker Input
    user_input = input("Enter target ticker symbol [Default: SPY]: ").strip().upper()
    target_symbol = user_input if user_input else "IWM"
    
    console.print(f"[yellow]Initializing Live Market Data Engine for {target_symbol} ({ENVIRONMENT})...[/yellow]")
    
    producer_task = asyncio.create_task(tastytrade_producer(data_queue, target_symbol, console))
    consumer_task = asyncio.create_task(dashboard_consumer(data_queue, console))
    
    try:
        await asyncio.gather(producer_task, consumer_task)
    except asyncio.CancelledError:
        console.print("[red]Shutting down engine tasks.[/red]")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExited gracefully.")
