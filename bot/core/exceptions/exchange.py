class ExchangeError(Exception):
    """Базовый класс для исключений, связанных с биржами"""
    pass

class UnknownEndpointError(ExchangeError):
    """Unknown endpoint"""
    pass