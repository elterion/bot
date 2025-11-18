from bot.analysis.pair_trading import backtest
from bot.analysis.strategy_analysis import analyze_strategy
from datetime import datetime
from zoneinfo import ZoneInfo
import polars as pl
import numpy as np
from tqdm import tqdm
import heapq
import pickle

from bot.core.db.postgres_manager import DBManager
from bot.config.credentials import host, user, password, db_name
db_params = {'host': host, 'user': user, 'password': password, 'dbname': db_name}
db_manager = DBManager(db_params)

# from bot.core.exchange.http_api import ExchangeManager, BybitRestAPI
from bot.utils.files import load_config

def base(token: str) -> str:
    return token.split('_')[0] if '_' in token else token

def find_best_params(df, token_1, token_2, dp_1, dp_2, ps_1, ps_2,
                     in_params, out_params, dist_in_params, dist_out_params,
                     stop_loss_std=5.0, sl_method=None, max_order=50,
                     min_trades=10,
                     leverage=1, n_best_params=2, verbose=0):
    """
    Параметры in_params и out_params передаются в виде положительных чисел.

    """

    heap = []
    end_date = df['time'][-1]
    start_date = df['time'][0]

    for thresh_in in in_params:
        for thresh_out in out_params:
            thresh_low_in = -thresh_in
            thresh_high_in = thresh_in
            thresh_low_out = -thresh_out
            thresh_high_out = thresh_out

            if abs(thresh_out) > abs(thresh_in):
                continue

            for dist_in in dist_in_params:
                for dist_out in dist_out_params:
                    tr = backtest(df, token_1, token_2, dp_1, dp_2, ps_1, ps_2,
                        thresh_low_in=thresh_low_in, thresh_high_in=thresh_high_in,
                        thresh_low_out=thresh_low_out, thresh_high_out=thresh_high_out,
                        long_possible=True, short_possible=True,
                        balance=max_order * 2, order_size=max_order, fee_rate=0.00055,
                        dist_in=dist_in, dist_out=dist_out,
                        stop_loss_std=stop_loss_std, sl_method=sl_method,
                        leverage=leverage
                        )

                    if tr.height >= min_trades:
                        metrics = analyze_strategy(tr, start_date=start_date,
                                                    end_date=end_date,
                                                    initial_balance=max_order * 2)
                        profit_ratio = metrics['profit_ratio']

                        if len(heap) < n_best_params:
                            heapq.heappush(heap, (profit_ratio, tr.height, thresh_in,
                                                  thresh_out, dist_in, dist_out))
                        else:
                            if profit_ratio > heap[0][0]:
                                heapq.heapreplace(heap, (profit_ratio, tr.height, thresh_in,
                                                         thresh_out, dist_in, dist_out))
    return heap

def grid_search(spread_df, token_1, token_2, method, start_time, end_time, min_trades, n_top_params,
        search_space, in_params, out_params, dist_in_params, dist_out_params,
        leverage, verbose=0):

    top_params = []

    # Загружаем техническую информацию по монетам (шаг цены, округление цены в usdt etc.)
    with open("./data/coin_information.pkl", "rb") as f:
        coin_information = pickle.load(f)

    # Сохраним информацию о шаге цены монет в переменных
    dp_1 = float(coin_information['bybit_linear'][token_1 + '_USDT']['qty_step'])
    ps_1 = int(coin_information['bybit_linear'][token_1 + '_USDT']['price_scale'])
    dp_2 = float(coin_information['bybit_linear'][token_2 + '_USDT']['qty_step'])
    ps_2 = int(coin_information['bybit_linear'][token_2 + '_USDT']['price_scale'])

    for tf, wind in search_space:
        if verbose > 0:
            print(f'Параметры модели. tf: {tf}, wind: {wind}')
        df = pl.DataFrame()

        df = spread_df.select('time', 'ts', token_1, token_2, f'{token_1}_size', f'{token_2}_size',
                f'{token_1}_bid_price', f'{token_1}_ask_price', f'{token_1}_bid_size', f'{token_1}_ask_size',
                f'{token_2}_bid_price', f'{token_2}_ask_price', f'{token_2}_bid_size', f'{token_2}_ask_size',
                f'z_score_{wind}_{tf}')
        df = df.rename({f'z_score_{wind}_{tf}': 'z_score'})
        df = df.drop_nulls()

        if df.is_empty():
            return (0, 0, '', 0, 0, 0, 0, 0)

        best_params = find_best_params(df, token_1, token_2,
                dp_1, dp_2, ps_1, ps_2, n_best_params=3,
                in_params=in_params, out_params=out_params,
                dist_in_params=dist_in_params, dist_out_params=dist_out_params,
                stop_loss_std=5.0, sl_method='leave', max_order=50, leverage=leverage,
                min_trades=min_trades, verbose=verbose)
        for profit, n_trades, thresh_in, thresh_out, dist_in, dist_out in best_params:
            if verbose > 0:
                print(f'profit: {profit:.2f}. {n_trades=}; params: {thresh_in, thresh_out, dist_in, dist_out}')

            if len(top_params) < n_top_params:
                heapq.heappush(top_params, (profit, n_trades, tf, wind, thresh_in, thresh_out, dist_in, dist_out))
            else:
                if profit > top_params[0][0]:
                    heapq.heapreplace(top_params, (profit, n_trades, tf, wind, thresh_in, thresh_out, dist_in, dist_out))

        if verbose > 0:
            print()
    for p in top_params:
        tqdm.write(f'({p[0]:.2f}, "{token_1}", "{token_2}", "{p[2]}", {p[3]}, {p[4]}, {p[5]}, {p[6]}, {p[7]})')

    return top_params

def main(method, in_params, out_params, dist_in_params, dist_out_params,
         min_trades, n_top_params,
         verbose=0,
         token_pairs_filename=None, save_to_file=None):

    config = load_config('./bot/config/config.yaml')

    end_time = config['end_time']
    valid_time = config['valid_time']
    start_time = config['start_time']
    leverage = config['leverage']
    min_order = config['min_order']
    max_order = config['max_order']
    search_space = config['search_space']

    # Зададим пространство поиска наилучших параметров входа
    search_space = [('4h', np.int64(w)) for w in search_space['4h']] + \
                        [('1h', np.int64(w)) for w in search_space['1h']]

    token_pairs = []
    with open(token_pairs_filename, 'r') as file:
        for line in file:
            a, b = line.strip().split()
            token_pairs.append((a, b))

    params_arr = []

    # --- Бектест по всем коинтегрированным парам токенов ---
    for token_1, token_2 in tqdm(token_pairs):
        filepath = f'./data/pair_backtest/{token_1}_{token_2}_{method}_full.parquet'
        try:
            df = pl.read_parquet(filepath, low_memory=True, rechunk=True, use_pyarrow=True)
        except FileNotFoundError:
            continue

        # print(df)
        top_params = grid_search(df, token_1, token_2, method, valid_time, end_time, min_trades, n_top_params,
            search_space, in_params, out_params, dist_in_params, dist_out_params,
            leverage, verbose=verbose)
        params_arr.extend(top_params)

    if save_to_file:
        with open(save_to_file, 'w') as file:
            for t, p in zip(token_pairs, params_arr):
                file.write(f'({p[0]:.2f}, "{t[0]}", "{t[1]}", "{p[2]}", {p[3]}, {p[4]}, {p[5]})\n')


if __name__ == '__main__':
    spr_method = 'dist'

    in_params = (1.6, 1.8, 2.0, 2.25, 2.5)
    out_params = (0.0, 0.25, 0.5)
    dist_in_params = (0, )
    dist_out_params = (0, )

    min_trades = 2
    n_top_params = 1 # Сколько лучших параметров печатать на экране

    token_pairs_filename = './data/token_pairs.txt'

    main(spr_method, in_params, out_params, dist_in_params, dist_out_params,
        min_trades, n_top_params, verbose=0,
        token_pairs_filename=token_pairs_filename,
        save_to_file='./data/pair_selection/ind_thresholds_dist.txt'
        )

    db_manager.close()
