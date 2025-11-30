import polars as pl

def rsi(df: pl.DataFrame, window: int = 14, col_name: str = None) -> pl.DataFrame:
    """
    Рассчитывает индекс относительной силы (RSI) с использованием сглаживания EMA.

    Args:
    ----------
    df : pl.DataFrame
        DataFrame с историческими данными, должен содержать столбец 'Close'
    window : int, optional
        Период расчета RSI (по умолчанию 14)
    col_name : str, optional
        Имя результирующего столбца. Автоформат: f'rsi_{window}' если None

    Return:
    -----------
    pl.DataFrame
        Исходный DataFrame с добавленным столбцом RSI

    Пример:
    -------
    >>> from jaref_bot.indicators import rsi
    >>> df = pl.DataFrame({'Close': [...]})
    >>> result = rsi(df, window=30, col_name='rsi_30min')
    """

    col_name = col_name if col_name is not None else f"rsi_{window}"

    return (
        df.lazy().with_columns(
        # Шаг 1: Изменение цены
            pl.col("Close").diff(1).alias('return'),
        ).with_columns(
        # Шаг 2: Разделяем на gain/loss
            pl.when(pl.col("return") > 0).then(pl.col("return")).otherwise(0).alias('gain'),
            pl.when(pl.col("return") < 0).then(-pl.col("return")).otherwise(0).alias('loss'),
        ).with_columns(
        # Шаг 3: Сглаживание через EMA (alpha = 1/window)
            pl.col("gain").ewm_mean(alpha=1/window, adjust=False).alias('avg_gain'),
            pl.col("loss").ewm_mean(alpha=1/window, adjust=False).alias('avg_loss'),
        ).with_columns(
        # Шаг 4: Относительная сила (RS)
            (pl.col("avg_gain") / pl.col("avg_loss")).alias('rs'),
        ).with_columns(
        # Шаг 5: Расчет RSI
            (100 - (100 / (1 + pl.col("rs")))).alias('rsi')
        ).with_columns(pl.col("rsi").alias(col_name)
        ).drop(["gain", "loss", 'avg_gain', 'avg_loss', 'rs', 'rsi', 'return']
        ).filter((pl.col(col_name) > 0) & (pl.col(col_name) < 100)
        ).collect()
        )

def rsi_signal(df: pl.DataFrame,
               window: int = 14,
               lower_bound: float = 30,
               upper_bound: float = 70,
               col_name: str = 'rsi') -> pl.DataFrame:
    """
    Функция рассчитывает индикатор RSI и генерирует торговые сигналы.

    Сигнальная логика:
    - Покупка (1):   когда RSI опускается ниже нижней границы (перепроданность)
    - Продажа (-1):  когда RSI поднимается выше верхней границы (перекупленность)
    - Нейтральная позиция (0): когда RSI находится между границами

    Args:
    ----------
    df : pl.DataFrame
        DataFrame с рассчитанными значениями RSI
    window : int, optional
        Период расчета RSI (по умолчанию 14)
    lower_bound : float
        Нижняя граница RSI для генерации сигнала покупки (0 < lower_bound < upper_bound)
    upper_bound : float
        Верхняя граница RSI для генерации сигнала продажи (lower_bound < upper_bound < 100)
    col_name : str, optional
        Название столбца с RSI (по умолчанию 'rsi')

    Return:
    -----------
    pl.DataFrame
        Исходный DataFrame с добавленным столбцом сигналов

    Особенности:
    ------------
    - Сигналы генерируются по принципу пересечения границ
    - Не реализует логику "удержания позиции" (только точки входа/выхода)
    - Требует предварительного расчета RSI в DataFrame
    - Автоматически именует столбец сигналов как f'{col_name}_signal'

    Примеры использования:
    ----------------------
    Стандартные границы (30/70):
    >>> df = rsi_signal(df, lower_bound=30, upper_bound=70)
    """

    signal_col = f'rsi{window}m_signal'
    res_df = rsi(df, window)

    return res_df.with_columns(
            pl.when(pl.col(col_name) < lower_bound).then(1)  # Сигнал к покупке
                .when(pl.col(col_name) > upper_bound).then(-1)  # Сигнал к продаже
                .otherwise(0)  # Нейтральная зона
                .alias(signal_col))
