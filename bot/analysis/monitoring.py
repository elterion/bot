import polars as pl
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
from bot.utils.pair_trading import get_lr_zscore, get_dist_zscore, get_tls_zscore
from bot.core.db.postgres_manager import DBManager
from bot.config.credentials import host, user, password, db_name

db_params = {'host': host, 'user': user, 'password': password, 'dbname': db_name}
db_manager = DBManager(db_params)


def plot_current_pairs(current_pairs, thr_in, thr_out):
    for row in current_pairs.iter_rows(named=True):
        token_1 = row['token_1']
        token_2 = row['token_2']
        side = row['side_1']
        t1_name = token_1[:-5]
        t2_name = token_2[:-5]
        open_time = row['created_at']

        start_ts = int(datetime.timestamp(open_time))
        end_ts = int(datetime.timestamp(datetime.now()))
        hist = db_manager.get_zscore_history(token_1, token_2, start_ts, end_ts)

        curr_zscore = round(hist[-1]['z_score'][0], 2)
        curr_profit = round(hist[-1]['profit'][0], 2)

        fig, ax1 = plt.subplots(figsize=(14, 3))

        # График z_score
        ax1.plot(hist.select('time'), hist.select(f'z_score'), color='blue', label='z_score', ls='-', lw=1)
        ax1.plot(hist.select('time'), hist.select(f'fixed_z_score'), color='gray', label='fixed z_score', ls='-', lw=1)
        ax1.set_title(f'{token_1[:-5]} - {token_2[:-5]} ({side}). z_score: {curr_zscore}; profit: {curr_profit}')

        ax1.set_ylabel('z_score')

        if side == 'long':
            ax1.axhline(-thr_in, c='g', linestyle='dotted')
            ax1.axhline(thr_out, c='r', linestyle='dotted')
        else:
            ax1.axhline(thr_in, c='g', linestyle='dotted')
            ax1.axhline(-thr_out, c='r', linestyle='dotted')

        # График профита
        ax2 = ax1.twinx()
        ax2.plot(hist.select('time'), hist.select('profit'), color='green', label='profit', lw=2.0)
        ax2.set_ylabel('profit')
        ax2.grid()
        plt.tight_layout()
        fig.legend(loc='upper right', bbox_to_anchor=(0.135, 0.9))
        plt.show()

def print_current_pairs(current_pairs):
    current_profit = 0
    for row in current_pairs.iter_rows(named=True):
        token_1 = row['token_1']
        token_2 = row['token_2']
        side = row['side_1']
        t1_name = token_1[:-5]
        t2_name = token_2[:-5]
        open_time = row['created_at']

        end_ts = int(datetime.timestamp(datetime.now()))
        start_ts = end_ts - 30

        try:
            hist = db_manager.get_zscore_history(token_1, token_2, start_ts, end_ts)
            curr_zscore = round(hist[-1]['z_score'][0], 2)
            fix_zscore = round(hist[-1]['fixed_z_score'][0], 2)
            curr_profit = round(hist[-1]['profit'][0], 2)
            current_profit += curr_profit
            print(f'{t1_name:>6} - {t2_name:6} ({side:>5}): z_score: {curr_zscore:5}; z_score fix: {fix_zscore:5}; profit: {curr_profit:5}')
        except IndexError:
            print(f'{t1_name:>6} - {t2_name:6} ({side:>5}): z_score: NaN; fix_zscore: NaN; profit: NaN')

        # if token_1 == 'ATOM_USDT':
        #     break

    print(f'{" "*52} current_profit: {current_profit:.2f}')

def all_pairs_monitoring(tf, wind, top_tokens):
    token_pairs = []
    with open('./bot/config/token_pairs.txt', 'r') as file:
        for line in file:
            a, b = line.strip().split()
            token_pairs.append((a, b))

    winds = np.array([wind,])
    td = int(tf[0]) * wind * 2 + 1

    end_time = datetime.now().replace(tzinfo=ZoneInfo("Europe/Moscow"))
    start_time = end_time - timedelta(hours = td)
    hist_df = db_manager.get_orderbooks(interval=tf, start_date=start_time)
    hist_df = hist_df.with_columns(pl.col('price').alias('avg_price'))

    current_data = db_manager.get_table('current_ob', df_type='polars')
    current_data = current_data.with_columns(
                        ((pl.col('bid_price_0') + pl.col('ask_price_0')) / 2.0).alias('avg_price')
                    )
    end_t = datetime.now().replace(tzinfo=ZoneInfo("Europe/Moscow"))
    st_t = end_t - timedelta(seconds = 20)

    tick_df = db_manager.get_tick_ob(start_time=st_t).with_columns(
        ((pl.col('bid_price') + pl.col('ask_price')) / 2.0).alias('avg_price')
    )

    all_pairs = {}

    for t1_name, t2_name in token_pairs:
        token_1 = t1_name + '_USDT'
        token_2 = t2_name + '_USDT'
        z_score = 0

        t1_tick_df = tick_df.filter(pl.col('token') == token_1)
        t2_tick_df = tick_df.filter(pl.col('token') == token_2)

        token_1_hist_price = hist_df.filter(pl.col('token') == token_1).tail(td)['avg_price'].to_numpy()
        token_2_hist_price = hist_df.filter(pl.col('token') == token_2).tail(td)['avg_price'].to_numpy()
        t1_med = np.append(token_1_hist_price, t1_tick_df['avg_price'].median())
        t2_med = np.append(token_2_hist_price, t2_tick_df['avg_price'].median())

        _, _, _, zscore = get_dist_zscore(t1_med, t2_med, np.array([wind]))
        spr, spr_mean, spr_std, alpha, lr_beta, lr_zscore = get_lr_zscore(t1_med, t2_med, np.array([wind]))
        spr, spr_mean, spr_std, alpha, tls_beta, tls_zscore = get_tls_zscore(t1_med, t2_med, np.array([wind]))
        dist_z_score = zscore[0]
        lr_zscore = lr_zscore[0]
        tls_zscore = tls_zscore[0]

        all_pairs[(t1_name, t2_name)] = (dist_z_score, lr_zscore, tls_zscore)

    sorted_pairs = sorted(
            all_pairs.items(),
            key=lambda x: abs(x[1][1]),
            reverse=True
        )

    for tokens, z_scores in sorted_pairs[:top_tokens]:
        t1, t2 = tokens
        dist_z_score, lr_zscore, tls_zscore = z_scores

        print(f'{t1:>7} - {t2:7} lr: {lr_zscore:5.2f}; dist: {dist_z_score:5.2f}; tls: {tls_zscore:5.2f}')
