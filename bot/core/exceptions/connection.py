class ConnectionError(Exception):
    """Базовый класс для исключений, связанных с биржами"""
    pass

class ConnectionLostError(ConnectionError):
    """Unknown endpoint"""
    pass