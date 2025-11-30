import polars as pl

def sma(df: pl.DataFrame, window: int = 14, col_name: str = None) -> pl.DataFrame:
    """
    Рассчитывает скользящую среднюю (simple moving average).

    Args:
    ----------
    df : pl.DataFrame
        DataFrame с историческими данными, должен содержать столбец 'Close'
    window : int, optional
        Период расчета (по умолчанию 14)
    col_name : str, optional
        Имя результирующего столбца. Автоформат: f'sma_{window}' если None

    Return:
    -----------
    pl.DataFrame
        Исходный DataFrame с добавленным столбцом col_name

    Пример:
    -------
    >>> import jaref_bot
    >>> df = pl.DataFrame({'Close': [...]})
    >>> result = jaref_bot.indicators.sma(df, window=30, col_name='sma_30min')
    """

    col_name = col_name if col_name is not None else f"sma_{window}"

    return df.with_columns(pl.col('Close').rolling_mean(window).alias(col_name))

def ema(df: pl.DataFrame, window: int = 14, col_name: str = None) -> pl.DataFrame:
    """
    Рассчитывает экспоненциальную скользящую среднюю (exponential moving average).

    Args:
    ----------
    df : pl.DataFrame
        DataFrame с историческими данными, должен содержать столбец 'Close'
    window : int, optional
        Период расчета (по умолчанию 14)
    col_name : str, optional
        Имя результирующего столбца. Автоформат: f'ema_{window}' если None

    Return:
    -----------
    pl.DataFrame
        Исходный DataFrame с добавленным столбцом col_name

    Пример:
    -------
    >>> import jaref_bot
    >>> df = pl.DataFrame({'Close': [...]})
    >>> result = jaref_bot.indicators.ema(df, window=30, col_name='ema_30min')
    """

    col_name = col_name if col_name is not None else f"ema_{window}"

    return df.with_columns(df['Close'].ewm_mean(span=14).alias(col_name))


def ma_signal(df: pl.DataFrame,
              col_name: str,
              res_col: str = None,
              threshold: float = 0.0) -> pl.DataFrame:
    """
    Функция генерирует торговые сигналы на основе предварительно вычисленных
    значений Moving Average.

    Сигнальная логика:
    - Покупка (1):   Восходящий тренд (ewm > threshold)
    - Продажа (-1):  Нисходящий тренд (ewm < threshold)

    Args:
    ----------
    df : pl.DataFrame
        DataFrame с рассчитанными значениями RSI
    col_name : str
        Название столбца с EWM
    res_col : str
        Название итогового столбца с торговым сигналом
    threshold : float
        Расстояние от 0, при котором происходит покупка/продажа актива

    Return:
    -----------
    pl.DataFrame
        Исходный DataFrame с добавленным столбцом сигналов
    """

    signal_col = res_col if res_col is not None else f"ma_{window}_signal"

    return df.with_columns(
            pl.when(pl.col('Close') - pl.col(col_name) > threshold).then(1)
                .when(pl.col('Close') - pl.col(col_name) < threshold).then(-1)
                .alias(signal_col))


def ma_cross_signal(df: pl.DataFrame,
    col_name_fast: str = 'ewm_fast',
    col_name_slow: str = 'ewm_slow') -> pl.DataFrame:
    """
    Функция генерирует торговые сигналы на основе предварительно вычисленных значений EWM.

    Сигнальная логика:
    - Покупка (1):   Fast EWM > Slow EWM
    - Продажа (-1):  Fast EWM < Slow EWM

    Args:
    ----------
    df : pl.DataFrame
        DataFrame с рассчитанными значениями RSI
    col_name_fast : str
        Название "быстрой" EWM
    col_name_slow : str
        Название "медленной" EWM

    Return:
    -----------
    pl.DataFrame
        Исходный DataFrame с добавленным столбцом сигналов
    """

    signal_col = f'ewm_cross_signal'

    return df.with_columns(
            pl.when(pl.col(col_name_fast) > pl.col(col_name_slow)).then(1)
                .when(pl.col(col_name_fast) < pl.col(col_name_slow)).then(-1)  # Сигнал к продаже
                .alias(signal_col))
