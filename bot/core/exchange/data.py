from dataclasses import dataclass, field
from sortedcontainers import SortedDict

@dataclass
class Orderbook:
    """Структура для хранения ордербука одного актива"""
    symbol: str
    # SortedDict для bids (цена DESC - максимальная цена первая)
    bids: SortedDict = field(default_factory=lambda: SortedDict(lambda x: -x))
    # SortedDict для asks (цена ASC - минимальная цена первая)
    asks: SortedDict = field(default_factory=SortedDict)
    update_ts: int = 0

    def update_snapshot(self, bids, asks, ts):
        """Полное обновление ордербука (snapshot)"""
        self.bids.clear()
        self.asks.clear()

        # Быстрая конвертация через float()
        for price_str, amount_str in bids:
            self.bids[float(price_str)] = float(amount_str)

        for price_str, amount_str in asks:
            self.asks[float(price_str)] = float(amount_str)

        self.update_ts = ts

    def update_delta(self, bids, asks, ts):
        """Инкрементальное обновление ордербука (delta)"""
        for price_str, amount_str in bids:
            price = float(price_str)
            amount = float(amount_str)
            if amount == 0.0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = amount

        for price_str, amount_str in asks:
            price = float(price_str)
            amount = float(amount_str)
            if amount == 0.0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = amount

        self.update_ts = ts

    def get_top_n_bids(self, n = 10):
        """Получить топ N лучших bid цен"""
        return [(p, v) for p, v in list(self.bids.items())[:n]]

    def get_top_n_asks(self, n = 10):
        """Получить топ N лучших ask цен"""
        return [(p, v) for p, v in list(self.asks.items())[:n]]
