import polars as pl

def rsi(df: pl.DataFrame, window: int = 14, col_name: str = 'Close') -> pl.DataFrame:
    """
    Рассчитывает индекс относительной силы (RSI) с использованием сглаживания EMA.

    Args:
    ----------
    df : pl.DataFrame
        DataFrame с историческими данными
    window : int, optional
        Период расчета RSI (по умолчанию 14)
    col_name : str, optional
        Имя столбца с ценой

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
    return (
        df.with_columns(
        # Шаг 1: Изменение цены
            pl.col(col_name).diff(1).alias('return'),
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
        ).drop(["gain", "loss", 'avg_gain', 'avg_loss', 'rs', 'return']
        ).filter((pl.col('rsi') > 0) & (pl.col('rsi') < 100)
        )[window:])