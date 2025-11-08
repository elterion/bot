class DBError(Exception):
    """Базовый класс для исключений, связанных с БД"""
    pass

class OrderNotFoundError(DBError):
    """Такого ордера не существует."""
    pass

class OrderDuplicationError(DBError):
    """Такой ордер уже существует в базе данных."""
    pass