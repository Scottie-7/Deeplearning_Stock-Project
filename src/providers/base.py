from abc import ABC, abstractmethod

class MarketDataSource(ABC):
    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        self.is_connected = False

    @abstractmethod
    async def connect(self) -> bool:
        """Authenticate and establish websocket connections."""
        pass

    @abstractmethod
    async def stream_data(self, symbol: str, queue):
        """Stream L1/L2 data into the shared async queue."""
        pass
