import polars as pl
from functools import reduce
from decimal import Decimal, ROUND_FLOOR

def round_size(value, dp):
        if not isinstance(value, Decimal):
            value = Decimal(value)
        if not isinstance(dp, Decimal):
            dp = Decimal(dp)
        return float((value / dp).quantize(Decimal('1'), rounding=ROUND_FLOOR) * dp)

def calculate_profit(open_price, close_price, n_coins, side='long', fee_rate=0.00055):
    usdt_open = n_coins * open_price
    open_fee = usdt_open * fee_rate

    usdt_close = n_coins * close_price
    close_fee = usdt_close * fee_rate

    if side == 'long':
        profit = usdt_close - usdt_open - open_fee - close_fee
    elif side == 'short':
        profit = usdt_open - usdt_close - open_fee - close_fee
    return profit

def create_ohlcv_from_trades(df, timeframe='1m', calculate_cvd=False):
    col = 'price' if 'price' in df.columns else 'close'

    if calculate_cvd:
        df = df.with_columns(
            (pl.when(pl.col("buyer") == False)
                .then(pl.col("qty"))
                .otherwise(-pl.col("qty")))
            .alias("delta")
        ).with_columns(
            (
               (pl.col("datetime") - pl.duration(hours=7))
               .dt.truncate("1d")
               + pl.duration(hours=7)
            )
            .alias("session_start")
        ).with_columns(pl.col("delta").cum_sum().over("session_start").alias("cvd")
        ).drop('delta', 'session_start')

    # Базовые агрегации (всегда присутствуют)
    base_aggs = [
        pl.col(col).first().alias("open"),
        pl.col(col).max().alias("high"),
        pl.col(col).min().alias("low"),
        pl.col(col).last().alias("close"),
        pl.col("qty").sum().alias("volume"),
        pl.len().alias("n_trades")
    ]

    # Агрегации для CVD (только при calculate_cvd == True)
    cvd_aggs = []
    if calculate_cvd:
        cvd_aggs = [
            pl.col("cvd").first().alias("cvd_open"),
            pl.col("cvd").max().alias("cvd_high"),
            pl.col("cvd").min().alias("cvd_low"),
            pl.col("cvd").last().alias("cvd_close")
        ]

    # Объединяем агрегации
    all_aggs = base_aggs + cvd_aggs

    # Обработка для индексного таймфрейма (оканчивается на 'i')
    if timeframe.endswith('i'):
        return df.with_row_index("id").with_columns(pl.col('id').cast(pl.Int32)).group_by_dynamic(
            index_column="id",
            every=timeframe,
        ).agg(
            pl.col("datetime").first().alias("time"),
            *all_aggs
        ).drop('id')
    # Обработка временного таймфрейма
    else:
        return df.group_by_dynamic(
                index_column="datetime",
                every=timeframe,
                closed="both",
            ).agg(
                *all_aggs
            ).rename({'datetime': 'time'})

def make_price_df_from_orderbooks_bulk(dfs: list[pl.DataFrame], tokens: list[str], trunc: str | None = None) -> pl.DataFrame:
    """
    Создаёт датафрейм с ценами произвольного числа токенов из их ордербуков.

    Параметры
    ----------
    dfs : list[pl.DataFrame]
        Список датафреймов с ордербуками. Каждый должен содержать колонки:
        - "time" или "bucket" : datetime
        - "bid_price" : float
        - "ask_price" : float
    tokens : list[str]
        Список тикеров, соответствующий каждому датафрейму.
    trunc : str, optional
        Интервал агрегации для group_by_dynamic (например, "1m", "5m").

    Возвращает
    ----------
    pl.DataFrame
        Датафрейм с колонкой времени и колонками цен для каждого токена.
    """

    if len(dfs) != len(tokens):
        raise ValueError("Количество датафреймов должно совпадать с количеством токенов")

    # Определяем имя колонки с датой по первому фрейму
    date_col = 'bucket' if 'bucket' in dfs[0].columns else 'time'

    # Преобразуем каждый ордербук в формат [time, <token_price>]
    processed = []
    for df, token in zip(dfs, tokens):
        processed.append(df.select(date_col, 'price').rename({'price': token}))

    # Последовательно объединяем датафреймы по времени
    df = reduce(
        lambda left, right: left.join(
            right, on=date_col, how='full', coalesce=True
        ),
        processed
    )

    # Сортируем и заполняем пропуски вперёд
    df = df.sort(by=date_col).fill_null(strategy='forward')

    # Агрегируем, если надо
    if trunc:
        df = df.group_by_dynamic(index_column=date_col, every=trunc).agg(
            *[pl.col(token).last() for token in tokens]
        )

    return df.drop_nulls()

def normalize(df, method, shift_to_zero=False):
    price_cols = [c for c in df.columns if c != "time"]

    if method == 'minimax':
        func = [((pl.col(c) - pl.col(c).min()) / (pl.col(c).max() - pl.col(c).min())) for c in price_cols]
    elif method == 'z_score':
        func = [((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()) for c in price_cols]
    else:
        raise NameError('Неизвестный способ нормализации. Доступные методы: "minimax", "z_score"')

    df = df.with_columns(func)

    if shift_to_zero:
        df = df.with_columns([(pl.col(c) - pl.col(c).first()) for c in price_cols])
    return df
