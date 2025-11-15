from bot.utils.pair_trading import make_df_from_orderbooks, make_trunc_df, create_zscore_df
from datetime import datetime
from zoneinfo import ZoneInfo
import polars as pl
import numpy as np
from tqdm import tqdm
import os

from bot.core.db.postgres_manager import DBManager
from bot.config.credentials import host, user, password, db_name
db_params = {'host': host, 'user': user, 'password': password, 'dbname': db_name}
db_manager = DBManager(db_params)

from bot.core.exchange.http_api import ExchangeManager, BybitRestAPI

def create_dfs(filename, spread_method, tf, winds, start_time, valid_time, end_time, min_order):
    token_pairs = []
    with open(filename, 'r') as file:
        for line in file:
            a, b = line.strip().split()
            token_pairs.append((a, b))

    for token_1, token_2 in tqdm(token_pairs):
        t1_name = token_1 + '_USDT'
        t2_name = token_2 + '_USDT'

        token_1_first_date = db_manager.get_oldest_date_in_orderbook(t1_name)
        token_2_first_date = db_manager.get_oldest_date_in_orderbook(t2_name)

        if token_1_first_date > start_time or token_2_first_date > start_time:
            tqdm.write(f'Для пары {token_1} - {token_2} не хватает тренировочной выборки.')
            continue

        df_1 = db_manager.get_tick_ob(token=t1_name,
                                        start_time=start_time,
                                        end_time=end_time)
        df_2 = db_manager.get_tick_ob(token=t2_name,
                                        start_time=start_time,
                                        end_time=end_time)

        df = make_df_from_orderbooks(df_1, df_2, token_1, token_2, start_time=start_time)
        agg_df = make_trunc_df(df, timeframe=tf, token_1=token_1, token_2=token_2, method='triple')
        tick_df = make_df_from_orderbooks(df_1, df_2, token_1, token_2, start_time=start_time)

        start_ts = int(datetime.timestamp(valid_time))
        spread_df = create_zscore_df(token_1, token_2, tick_df, agg_df, tf, winds,
                                     min_order, start_ts, median_length=6, spr_method=spread_method)

        spread_df.write_parquet(f'./data/pair_backtest/{token_1}_{token_2}_{tf}_{spread_method}.parquet')

    def merge_files():
        pass

if __name__ == '__main__':
    spread_method = 'dist'
    min_order = 40

    end_time = datetime(2025, 11, 12, 0, 0, tzinfo=ZoneInfo("Europe/Moscow"))
    valid_time = datetime(2025, 10, 22, 0, 0, tzinfo=ZoneInfo("Europe/Moscow"))
    start_time = datetime(2025, 10, 12, 0, 0, tzinfo=ZoneInfo("Europe/Moscow"))

    filename = './data/token_pairs.txt' # Файл, в котором указаны пары токенов

    for tf, winds in (('4h', np.array([12, 14, 16, 18, 24, 30])),
                      ('1h', np.array([18, 24, 36, 48, 64, 72, 96, 120])),
                      # ('5m', np.array([60, 90, 120, 180, 240, 300, 450, 600]))
                      ):
        create_dfs(filename, spread_method, tf, winds, start_time, valid_time, end_time, min_order)

    db_manager.close()
