import polars as pl
import numpy as np
from datetime import datetime, timedelta

from bot.core.db.postgres_manager import DBManager
from bot.config.credentials import host, user, password, db_name
db_params = {'host': host, 'user': user, 'password': password, 'dbname': db_name}
db_manager = DBManager(db_params)


def get_duration_string(dur: int):
    if dur < 60:
        return f'{dur} seconds'
    elif dur < 3600:
        mins = dur // 60
        secs = dur - mins * 60
        return f'{mins} minutes {secs} seconds'
    elif dur < 3600 * 24:
        hours = dur // 3600
        mins = (dur - hours * 3600) // 60
        secs = dur - hours * 3600 - 60 * mins
        return f'{hours} hours {mins} minutes {secs} seconds'
    else:
        days = int(dur // 86400)
        remainder = dur % 86400
        hours = int(remainder // 3600)
        remainder %= 3600
        minutes = int(remainder // 60)
        seconds = int(remainder % 60)
        return f'{days} days {hours} hours {minutes} minutes {seconds} seconds'

def analyze_strategy(df: pl.DataFrame, start_date, end_date,
                     initial_balance: float = 1000.0) -> dict:
    """
    Анализирует торговую стратегию на основе данных о сделках и возвращает ключевые метрики.

    Args:
    -------
    trades_df (pl.DataFrame): DataFrame с историей торговых сделок, содержащий колонки:
        - open_ts (i64): Unix timestamp открытия сделки
        - close_ts (i64): Unix timestamp закрытия сделки
        - total_profit (f64): Суммарная прибыль/убыток сделки в USDT
        - fees (f64): Комиссии за сделку
        - reason (i64): Причина закрытия (1 - z_score, 2 - стоп-лосс, 3 - ликвидация)
        - и другие колонки (qty_1, qty_2, prices, etc.)

    initial_balance (float): Начальный баланс счета в USDT
    order_size (float): Размер ордера в USDT

    Return:
    -------
    dict: Словарь с рассчитанными метриками стратегии
    """
    if df is None or df.height == 0:
        return {}

    metrics = dict()
    df = df.sort(by="open_ts")
    df = df.with_columns(
        (pl.col('close_time') - pl.col('open_time')).alias('duration')
    )

    hour_duration = df['duration'].sum().days * 24 + df['duration'].sum().seconds / 3600

    # --- Базовая информация ---
    total_seconds = (end_date - start_date).total_seconds()

    metrics['total_days'] = round(total_seconds / 86400.0, 2)
    metrics['n_trades'] = len(df)

    # --- Рассчитываем длительности сделок ---
    df = df.with_columns((pl.col("close_ts") - pl.col("open_ts")).alias('duration'))

    metrics["duration_min"] = timedelta(seconds=int(df['duration'].min()))
    metrics["duration_max"] = timedelta(seconds=int(df['duration'].max()))
    metrics["duration_avg"] = timedelta(seconds=int(df['duration'].mean()))

    # --- Проверяем наличие стоп-лоссов и ликвидаций ---
    metrics['stop_losses'] = df.filter(pl.col("reason") == 2).height
    metrics['liquidations'] = df.filter(pl.col("reason") == 3).height

    # --- Расчет баланса по истории сделок ---
    df = df.with_columns(
        pl.col("total_profit").cum_sum().alias("cum_profit")
    ).with_columns(
        (pl.lit(float(initial_balance)) + pl.col("cum_profit")).alias("balance")
    )

    metrics['initial_balance'] = initial_balance
    metrics['final_balance'] = round(df['balance'][-1], 4)
    metrics['profit'] = round(df['total_profit'].sum(), 2)
    metrics['$_hour'] = round(df['total_profit'].sum() / hour_duration, 2)

    # --- Доходность ---
    metrics["total_perc_return"] = round((metrics['final_balance']
                                / metrics['initial_balance'] - 1) * 100, 2)

    # try:
    #     metrics["annual_return"] = round(((metrics['final_balance'] / metrics['initial_balance']
    #                                       ) ** (365 / metrics['total_days']) - 1) * 100
    #                                       , 2)
    # except TypeError:
    #     metrics["annual_return"] = -100
    # except ZeroDivisionError:
    #     metrics["annual_return"] = 0
    #     return metrics

    # --- Рассчет изменений баланса ---
    # df = df.with_columns(
    #     (pl.col("balance") - pl.col("balance").shift(1)).alias("absolute_change"),
    #     (((pl.col("balance") / pl.col("balance").shift(1)) - 1) * 100).alias("percent_change")
    # )

    # --- Показатели просадки (Drawdown) ---
    df = df.with_columns(
        pl.col("balance").cum_max().alias("cum_max")
    )
    df = df.with_columns(
        (pl.col("balance") - pl.col("cum_max")).alias("drawdown") # В абсолютных величинах
    )

    metrics['max_drawdown'] = round(
        df.select("drawdown",'cum_profit').min_horizontal().min(),
        2)

    # --- Информация по сделкам ---
    metrics['max_profit'] = round(df['total_profit'].max(), 2)
    metrics['max_loss'] = round(min(df['total_profit'].min(), 0), 2)
    metrics['avg_profit'] = round(df['total_profit'].mean(), 2)


    std = df['total_profit'].std()
    if std is None:
        std = 0
    metrics['profit_std'] = round(std, 2)

    add_param = 10 # Настраиваемый параметр, который добавляется к знаменателю, чтобы сгладить
                   # разницу между слабоплюсовым, но безубыточным trades_df,
                   # и сильно более плюсовым, но имеющим просадку датафреймом
                   # Чем больше add_param, тем меньше преимущество у безубыточного df.

    profit_ratio = 100 * (df['total_profit'].sum() /
                          (abs(metrics['max_drawdown']) + add_param) /
                          (std + 5 * add_param))
    metrics['profit_ratio'] = round(profit_ratio, 3)

    return metrics

    # Коэффициент Шарпа
    metrics['sharpe_ratio'] = round((metrics['avg_return'] / metrics['std_return'])
                                    * np.sqrt(metrics['trades_per_year']), 4)

    # Коэффициент Сортино
    downside_returns = df.filter(pl.col("percent_change") < 0)["percent_change"]

    if downside_returns.len() > 0:
        downside_std = downside_returns.std()
        metrics['sortino_ratio'] = round(
            (metrics['avg_return'] / downside_std) * np.sqrt(metrics['trades_per_year']),
            4
        ) if downside_std != 0 else float('nan')
    else:
        metrics['sortino_ratio'] = float('nan')

    # Коэффициент Кальмара: отношение CAGR к абсолютной величине максимальной просадки
    if isinstance(metrics['max_drawdown'], float) and metrics['max_drawdown'] < 1:
        metrics['calmar_ratio'] = round(metrics["annual_return"] / abs(metrics['max_drawdown'])
                                        if metrics['max_drawdown'] != 0 else np.nan, 4)
    elif metrics['max_drawdown'] > 1:
        raise Exception('Величина max_drawdown не может быть больше 1!')

    # --- Аналитика по сделкам ---
    metrics['win_ratio'] = round(metrics['winning_trades'] / metrics['n_trades']
                                 if metrics['n_trades'] > 0 else 0, 2)

    return metrics

def create_pair_trades_df(log_file='./logs/trades.jsonl'):
    pair_trades_df = pl.DataFrame()

    orders = pl.read_ndjson(log_file).with_columns(
        pl.col('ct').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").dt.replace_time_zone("Europe/Moscow")
    )
    trading_history = db_manager.get_table('trading_history', df_type='polars')

    for i, row in enumerate(orders.iter_rows(named=True)):
        if row['action'] == 'close':
            continue

        ct = row['ct']
        token_1 = row['token_1']
        token_2 = row['token_2']
        tf = row['tf']
        wind = row['wind']
        thresh_in = row['thresh_in']
        thresh_out = row['thresh_out']
        side = row['side']
        beta = row['beta']
        z_score_open = row['z_score']

        for row in orders[i:].iter_rows(named=True):
            if row['token_1'] == token_1 and row['token_2'] == token_2 and row['action'] == 'close':
                z_score_close = row['z_score']
                break

        t1_data = trading_history.filter(
            (pl.col('token') == token_1 + '_USDT') & (abs(pl.col('created_at') - ct) < timedelta(seconds=10))
        )
        t2_data = trading_history.filter(
            (pl.col('token') == token_2 + '_USDT') & (abs(pl.col('created_at') - ct) < timedelta(seconds=10))
        )

        if t1_data.is_empty() or t2_data.is_empty():
            continue

        t1_op = t1_data['open_price'][0]
        t2_op = t2_data['open_price'][0]
        t1_cp = t1_data['close_price'][0]
        t2_cp = t2_data['close_price'][0]
        t1_qty = t1_data['qty'][0]
        t2_qty = t2_data['qty'][0]
        t1_profit = t1_data['profit'][0]
        t2_profit = t2_data['profit'][0]

        open_ts = int(datetime.timestamp(ct))
        close_time = max(t1_data['closed_at'][0], t2_data['closed_at'][0])
        close_ts = int(datetime.timestamp(close_time))
        fees = t1_data['realized_pnl'][0] + t2_data['realized_pnl'][0]

        pair_trades_df = pair_trades_df.vstack(pl.DataFrame({
            'open_time': ct,
            'open_ts': open_ts,
            'close_time': close_time,
            'close_ts': close_ts,
            'token_1': token_1,
            'token_2': token_2,
            'side': side,
            'tf': tf,
            'wind': wind,
            'thresh_in': thresh_in,
            'thresh_out': thresh_out,
            'beta': beta,
            'z_score_open': z_score_open,
            'z_score_close': z_score_close,
            'qty_1': t1_qty,
            'qty_2': t2_qty,
            'open_price_1': t1_op,
            'close_price_1': t1_cp,
            'open_price_2': t2_op,
            'close_price_2': t2_cp,
            'fees': fees,
            'profit_1': t1_profit,
            'profit_2': t2_profit,
            'total_profit': t1_profit + t2_profit,
            'reason': 1
        }))

    pair_trades_df = pair_trades_df.with_columns(
        (pl.col('close_time') - pl.col('open_time')).alias('duration'),
    )

    return pair_trades_df.sort(by='open_time')
