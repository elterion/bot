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

# from bot.core.exchange.http_api import ExchangeManager, BybitRestAPI
from bot.utils.files import load_config

def create_dfs(token_pairs, spread_method, tf, winds, start_time, valid_time, end_time, min_order):
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

def merge_files(n_tf, token_pairs, method, start_time, end_time):
    tf_1 = '4h'
    tf_2 = '1h'

    if n_tf == 3:
        tf_3 = '5m'

    time_series = pl.datetime_range(start=start_time, end=end_time, interval="5s", eager=True)
    main_df = pl.DataFrame({'time': time_series})

    for token_1, token_2 in tqdm(token_pairs):
        cols_to_drop = ['time', token_1, token_2, f'{token_1}_size', f'{token_2}_size',
                   f'{token_1}_bid_price', f'{token_2}_bid_price',
                   f'{token_1}_ask_price', f'{token_2}_ask_price',
                   f'{token_1}_bid_size', f'{token_2}_bid_size',
                   f'{token_1}_ask_size', f'{token_2}_ask_size']

        try:
            spread_df_1 = pl.read_parquet(f'./data/pair_backtest/{token_1}_{token_2}_{tf_1}_{method}.parquet',
                            low_memory=True, rechunk=True, use_pyarrow=True).filter(
                            (pl.col('time') >= start_time) & (pl.col('time') < end_time)
                        )
            spread_df_2 = pl.read_parquet(f'./data/pair_backtest/{token_1}_{token_2}_{tf_2}_{method}.parquet',
                                low_memory=True, rechunk=True, use_pyarrow=True).filter(
                                (pl.col('time') >= start_time) & (pl.col('time') < end_time)
                            )
            if n_tf == 3:
                spread_df_3 = pl.read_parquet(f'./data/pair_backtest/{token_1}_{token_2}_{tf_3}_{method}.parquet',
                                    low_memory=True, rechunk=True, use_pyarrow=True).filter(
                                    (pl.col('time') >= start_time) & (pl.col('time') < end_time)
                                )

            df = spread_df_1.join(spread_df_2.drop(cols_to_drop), on='ts', coalesce=True)

            if n_tf == 3:
                df = df.join(spread_df_3.drop(cols_to_drop), on='ts', coalesce=True)

            df.write_parquet(f'./data/pair_backtest/{token_1}_{token_2}_{method}_full.parquet')
        except FileNotFoundError:
            print(f'FileNotFoundError: {token_1} - {token_2}')
            continue

def clean_files(search_space, token_pairs, method):
    for tf in search_space.keys():
        for token_1, token_2 in token_pairs:
            file_path = f'./data/pair_backtest/{token_1}_{token_2}_{tf}_{method}.parquet'
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f'Удалён: {file_path}')


if __name__ == '__main__':
    config = load_config('./bot/config/config.yaml')

    spread_method = config['backtest_spr_method']
    min_order = config['min_order']

    end_time = config['end_time']
    valid_time = config['valid_time']
    start_time = config['start_time']

    search_space = config['search_space']
    n_tf = len(search_space)

    filename = config['token_pairs_file']

    token_pairs = []
    with open(filename, 'r') as file:
        for line in file:
            a, b = line.strip().split()
            token_pairs.append((a, b))

    for tf, winds in search_space.items():
        winds = np.array(winds)
        create_dfs(token_pairs, spread_method, tf, winds, start_time, valid_time, end_time, min_order)
    db_manager.close()

    # merge_files(n_tf, token_pairs, spread_method, valid_time, end_time)
    # clean_files(search_space, token_pairs, spread_method)
