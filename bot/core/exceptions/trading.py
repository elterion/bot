class TradingError(Exception):
    """Базовый класс для торговых исключений"""
    pass

class PlaceOrderError(TradingError):
    """Ошибки при постановке ордера"""
    pass

class SetLeverageError(TradingError):
    """Ошибки при изменении плеча"""
    pass

class NoSuchOrderError(TradingError):
    """Попытка обратиться к несуществующему ордеру"""
    pass