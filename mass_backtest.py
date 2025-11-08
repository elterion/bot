from bot.utils.pair_trading import get_qty
from bot.analysis.strategy_analysis import analyze_strategy

from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import polars as pl
import numpy as np
import random
import pickle
import json
from tqdm import tqdm

def check_pos(name, pairs):
    token_1, token_2, *_ = name.split('_')
    return any(a == token_1 and b == token_2 for a, b, _ in pairs)

def calculate_profit(open_price, close_price, n_coins, side, fee_rate=0.00055):
    usdt_open = n_coins * open_price
    open_fee = usdt_open * fee_rate

    usdt_close = n_coins * close_price
    close_fee = usdt_close * fee_rate

    if side == 'long':
        profit = usdt_close - usdt_open - open_fee - close_fee
    elif side == 'short':
        profit = usdt_open - usdt_close - open_fee - close_fee
    return profit

def place_order(tokens_in_position, pairs, current_orders, trades,
                time, token_1, token_2, action, pos_side, qty_1, qty_2,
                t1_price, t2_price, t1_vol, t2_vol, beta, z_score,
                tf, wind, thresh_in, thresh_out, fee_rate,
                min_order_size, max_order_size, leverage, reason=None, verbose=False):
    if action == 'open':
        t1_avail_usdt = t1_vol / t1_price
        t2_avail_usdt = t2_vol / t2_price
        avail_usdt = min(t1_avail_usdt, t2_avail_usdt, max_order_size) * leverage

        if avail_usdt > min_order_size:
            pairs.append((token_1, token_2, pos_side))
            tokens_in_position.append(token_1)
            tokens_in_position.append(token_2)
            current_orders[(token_1, token_2)] = {'time': time, 'pos_side': pos_side, 'qty_1': qty_1, 'qty_2': qty_2,
                                                 't1_price': t1_price, 't2_price': t2_price, 'z_score': z_score}
            if verbose:
                act_1 = 'buy' if pos_side == 'long' else 'sell'
                act_2 = 'sell' if pos_side == 'long' else 'buy'
                print(f'{time} [{pos_side} open] {act_1} {qty_1} {token_1} for {t1_price}; {act_2} {qty_2} {token_2} for {t2_price}; z_score: {z_score:.2f}')

    elif action == 'close':
        open_data = current_orders[(token_1, token_2)]
        open_qty_1 = open_data['qty_1']
        open_qty_2 = open_data['qty_2']

        if (t1_vol > open_qty_1 and t2_vol > open_qty_2 and reason == 1) or (reason == 2):
            pairs.remove((token_1, token_2, pos_side))
            tokens_in_position.remove(token_1)
            tokens_in_position.remove(token_2)

            fees = fee_rate * (qty_1 * open_data['t1_price'] + qty_2 * open_data['t2_price']) * leverage
            pos_side_2 = 'short' if pos_side == 'long' else 'long'
            profit_1 = calculate_profit(open_data['t1_price'], t1_price, n_coins=qty_1, side=pos_side, fee_rate=fee_rate)
            profit_2 = calculate_profit(open_data['t2_price'], t2_price, n_coins=qty_2, side=pos_side_2, fee_rate=fee_rate)

            trades.append({
                'open_time': open_data['time'],
                'close_time': time,
                'token_1': token_1,
                'token_2': token_2,
                'side': pos_side,
                'tf': tf,
                'wind': wind,
                'thresh_in': thresh_in,
                'thresh_out': thresh_out,
                'beta': beta,
                'open_z_score': open_data['z_score'],
                'close_z_score': z_score,
                'qty_1': qty_1,
                'qty_2': qty_2,
                'open_price_1': open_data['t1_price'],
                'close_price_1': t1_price,
                'open_price_2': open_data['t2_price'],
                'close_price_2': t2_price,
                'fees': fees,
                'profit_1': profit_1,
                'profit_2': profit_2,
                'total_profit': profit_1 + profit_2,
                'reason': reason,
            })

            current_orders.pop((token_1, token_2))

            if verbose:
                act_1 = 'buy' if pos_side == 'long' else 'sell'
                act_2 = 'sell' if pos_side == 'long' else 'buy'
                print(f'{time} [{pos_side} close] {act_1} {qty_1} {token_1} for {t1_price}; {act_2} {qty_2} {token_2} for {t2_price}; z_score: {z_score:.2f}')

def run_single_tf_backtest(main_df, tf, wind, in_, out_, leverage, max_pairs, max_order_size,
                           fee_rate, start_time, end_time, coin_information):
    tokens_in_position = []
    pairs = []
    current_orders = {}
    trades = []

    all_pairs = [(col.split('_')[0], col.split('_')[1]) for col in main_df.columns if col.endswith('z_score')]

    # Добавим перемешивание порядка токенов, потому что он влияет на те позиции, которые будут открыты в моменте
    random.shuffle(all_pairs)

    for row in main_df.iter_rows(named=True):
        time = row['time']

        for token_1, token_2 in all_pairs:
            if not row[token_1] or not row[token_2] or not row[f'{token_1}_{token_2}_z_score']:
                continue

            low_in = -in_
            low_out = -out_
            high_in = in_
            high_out = out_

            z_score = row[f'{token_1}_{token_2}_z_score']

            # ----- Проверяем условия для входа в позицию -----
            if (len(pairs) < max_pairs and token_1 not in tokens_in_position and token_2 not in tokens_in_position):

                # --- Входим в лонг ---
                if z_score < low_in:
                    t1_price = row[f'{token_1}_ask_price']
                    t2_price = row[f'{token_2}_bid_price']
                    t1_vol = row[f'{token_1}_ask_size']
                    t2_vol = row[f'{token_2}_bid_size']
                    qty_1, qty_2 = get_qty(token_1, token_2, t1_price, t2_price, None, coin_information, 2 * max_order_size * leverage,
                              method='usdt_neutral')
                    place_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                                'open', 'long', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, z_score,
                                tf, wind, in_, out_, fee_rate, min_order_size, max_order_size, leverage, verbose=False)

                # --- Открываем шорт ---
                if z_score > high_in:
                    t1_price = row[f'{token_1}_bid_price']
                    t2_price = row[f'{token_2}_ask_price']
                    t1_vol = row[f'{token_1}_bid_size']
                    t2_vol = row[f'{token_2}_ask_size']
                    qty_1, qty_2 = get_qty(token_1, token_2, t1_price, t2_price, None, coin_information, 2 * max_order_size * leverage,
                              method='usdt_neutral')
                    place_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                                'open', 'short', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, z_score,
                                tf, wind, in_, out_, fee_rate, min_order_size, max_order_size, leverage, verbose=False)

            # ----- Проверяем условия для выхода из позиции -----
            # --- Закрываем лонг ---
            if z_score > high_out and (token_1, token_2, 'long') in pairs:
                t1_price = row[f'{token_1}_bid_price']
                t2_price = row[f'{token_2}_ask_price']
                t1_vol = row[f'{token_1}_bid_size']
                t2_vol = row[f'{token_2}_ask_size']
                qty_1 = current_orders[(token_1, token_2)]['qty_1']
                qty_2 = current_orders[(token_1, token_2)]['qty_2']
                place_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                            'close', 'long', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, z_score,
                                tf, wind, in_, out_, fee_rate, min_order_size, max_order_size, leverage, reason=1, verbose=False)

            # --- Закрываем шорт ---
            if z_score < low_out and (token_1, token_2, 'short') in pairs:
                t1_price = row[f'{token_1}_ask_price']
                t2_price = row[f'{token_2}_bid_price']
                t1_vol = row[f'{token_1}_ask_size']
                t2_vol = row[f'{token_2}_bid_size']
                qty_1 = current_orders[(token_1, token_2)]['qty_1']
                qty_2 = current_orders[(token_1, token_2)]['qty_2']
                place_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                            'close', 'short', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, z_score,
                                tf, wind, in_, out_, fee_rate, min_order_size, max_order_size, leverage, reason=1, verbose=False)

            # --- Проверка стоп-лосса ---
            if (token_1, token_2, 'long') in pairs:
                op_1 = current_orders[(token_1, token_2)]['t1_price']
                op_2 = current_orders[(token_1, token_2)]['t2_price']
                t1_price = row[f'{token_1}_bid_price']
                t2_price = row[f'{token_2}_ask_price']

                sl_price_1 = op_1 - 0.85 * op_1 / leverage
                sl_price_2 = op_2 + 0.85 * op_2 / leverage

                if t1_price < sl_price_1 or t2_price > sl_price_2:
                    qty_1 = current_orders[(token_1, token_2)]['qty_1']
                    qty_2 = current_orders[(token_1, token_2)]['qty_2']
                    t1_vol = row[f'{token_1}_bid_size']
                    t2_vol = row[f'{token_2}_ask_size']

                    place_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                                'close', 'long', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, z_score,
                                tf, wind, in_, out_, fee_rate, min_order_size, max_order_size, leverage, reason=2, verbose=False)

            if (token_1, token_2, 'short') in pairs:
                op_1 = current_orders[(token_1, token_2)]['t1_price']
                op_2 = current_orders[(token_1, token_2)]['t2_price']
                t1_price = row[f'{token_1}_ask_price']
                t2_price = row[f'{token_2}_bid_price']

                sl_price_1 = op_1 + 0.85 * op_1 / leverage
                sl_price_2 = op_2 - 0.85 * op_2 / leverage

                if t1_price > sl_price_1 or t2_price < sl_price_2:
                    qty_1 = current_orders[(token_1, token_2)]['qty_1']
                    qty_2 = current_orders[(token_1, token_2)]['qty_2']
                    t1_vol = row[f'{token_1}_ask_size']
                    t2_vol = row[f'{token_2}_bid_size']

                    place_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                                'close', 'short', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, z_score,
                                tf, wind, in_, out_, fee_rate, min_order_size, max_order_size, leverage, reason=2, verbose=False)

    trades_df = pl.DataFrame(trades)
    if trades_df.is_empty():
        return pl.DataFrame(), dict()

    try:
        trades_df = trades_df.with_columns(
            (pl.col('open_time').dt.timestamp() // 1_000_000).alias('open_ts'),
            (pl.col('close_time').dt.timestamp() // 1_000_000).alias('close_ts'),
            (pl.col('close_time') - pl.col('open_time')).alias('duration'),
        )

        metrics = analyze_strategy(trades_df, start_date=start_time, end_date=end_time, initial_balance=200.0)
        return trades_df, metrics
    except pl.ColumnNotFoundError:
        return None

def run_single_tf_backtest_reverse(main_df, tf, wind, in_, out_, dist_in, dist_out, max_pairs, leverage,
                                   max_order_size, fee_rate, start_time, end_time, coin_information,
                                   reverse_in=True, reverse_out=False):
    tokens_in_position = []
    pairs = []
    current_orders = {}
    trades = []
    curr_tracking_in = dict()
    curr_tracking_out = dict()

    all_pairs = [(col.split('_')[0], col.split('_')[1]) for col in main_df.columns if col.endswith('z_score')]

    # Добавим перемешивание порядка токенов, потому что он влияет на те позиции, которые будут открыты в моменте
    random.shuffle(all_pairs)

    for row in main_df.iter_rows(named=True):
        time = row['time']

        for token_1, token_2 in all_pairs:
            if not row[token_1] or not row[token_2] or not row[f'{token_1}_{token_2}_z_score']:
                continue

            low_in = -in_
            low_out = -out_
            high_in = in_
            high_out = out_

            flag_in = False
            flag_out = False

            z_score = row[f'{token_1}_{token_2}_z_score']

            # Если пара токенов входит в диапазон открытия позиции, и она ещё не отслеживается, добавляем в треккинг
            if abs(z_score) > in_ and not (token_1, token_2) in curr_tracking_in:
                curr_tracking_in[(token_1, token_2)] = z_score
            # Если пара токенов уже отслеживается, обновляем максимумы
            elif (token_1, token_2) in curr_tracking_in and abs(z_score) > abs(curr_tracking_in[(token_1, token_2)]):
                curr_tracking_in[(token_1, token_2)] = z_score
            # Если открыто максимальное кол-во позиций, а текущая пара выходит из диапазона входа, убираем из отслеживаемых
            elif (token_1, token_2) in curr_tracking_in and abs(z_score) < in_ and len(pairs) >= max_pairs:
                curr_tracking_in.pop((token_1, token_2))
            # Если z_score откатывается на dist от максимума, разрешаем открытие позиции
            elif (token_1, token_2) in curr_tracking_in and abs(z_score) < abs(curr_tracking_in[(token_1, token_2)]) - dist_in:
                flag_in = True

            # --- Дальше проверяем те пары, которые уже в позиции ---

            if (token_1, token_2, 'long') in pairs and z_score > high_out and not (token_1, token_2) in curr_tracking_out:
                curr_tracking_out[(token_1, token_2)] = z_score
            # Если пара токенов уже отслеживается, обновляем максимумы
            elif ((token_1, token_2, 'long') in pairs and
                  (token_1, token_2) in curr_tracking_out and
                  z_score > curr_tracking_out[(token_1, token_2)]):
                curr_tracking_out[(token_1, token_2)] = z_score
            # Если пара покидает диапазон выхода, убираем из трека
            elif ((token_1, token_2, 'long') in pairs and
                  (token_1, token_2) in curr_tracking_out and
                  z_score < high_out):
                curr_tracking_out.pop((token_1, token_2))
            # Если z_score откатывается на dist от максимума, разрешаем закрытие позиции
            elif ((token_1, token_2, 'long') in pairs and
                  (token_1, token_2) in curr_tracking_out and
                  z_score < curr_tracking_out[(token_1, token_2)] - dist_out):
                flag_out = True

            if (token_1, token_2, 'short') in pairs and z_score < low_out and not (token_1, token_2) in curr_tracking_out:
                curr_tracking_out[(token_1, token_2)] = z_score
            # Если пара токенов уже отслеживается, обновляем максимумы
            elif ((token_1, token_2, 'short') in pairs and
                  (token_1, token_2) in curr_tracking_out and
                  z_score < curr_tracking_out[(token_1, token_2)]):
                curr_tracking_out[(token_1, token_2)] = z_score
            # Если пара покидает диапазон выхода, убираем из трека
            elif ((token_1, token_2, 'short') in pairs and
                  (token_1, token_2) in curr_tracking_out and
                  z_score > low_out):
                curr_tracking_out.pop((token_1, token_2))
            # Если z_score откатывается на dist от максимума, разрешаем закрытие позиции
            elif ((token_1, token_2, 'short') in pairs and
                  (token_1, token_2) in curr_tracking_out and
                  z_score > curr_tracking_out[(token_1, token_2)] + dist_out):
                flag_out = True

            # ----- Проверяем условия для входа в позицию -----
            if (len(pairs) < max_pairs and token_1 not in tokens_in_position and token_2 not in tokens_in_position) and flag_in:

                # --- Входим в лонг ---
                if z_score < low_in:
                    t1_price = row[f'{token_1}_ask_price']
                    t2_price = row[f'{token_2}_bid_price']
                    t1_vol = row[f'{token_1}_ask_size']
                    t2_vol = row[f'{token_2}_bid_size']
                    qty_1, qty_2 = get_qty(token_1, token_2, t1_price, t2_price, None, coin_information, 2 * max_order_size * leverage,
                              method='usdt_neutral')
                    place_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                                'open', 'long', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, z_score,
                                tf, wind, in_, out_, fee_rate, min_order_size, max_order_size, leverage, verbose=False)
                    flag_in = False

                # --- Открываем шорт ---
                if z_score > high_in:
                    t1_price = row[f'{token_1}_bid_price']
                    t2_price = row[f'{token_2}_ask_price']
                    t1_vol = row[f'{token_1}_bid_size']
                    t2_vol = row[f'{token_2}_ask_size']
                    qty_1, qty_2 = get_qty(token_1, token_2, t1_price, t2_price, None, coin_information, 2 * max_order_size * leverage,
                              method='usdt_neutral')
                    place_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                                'open', 'short', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, z_score,
                                tf, wind, in_, out_, fee_rate, min_order_size, max_order_size, leverage, verbose=False)
                    flag_in = False

            # ----- Проверяем условия для выхода из позиции -----
            # --- Закрываем лонг ---
            if z_score > high_out and (token_1, token_2, 'long') in pairs and flag_out:
                t1_price = row[f'{token_1}_bid_price']
                t2_price = row[f'{token_2}_ask_price']
                t1_vol = row[f'{token_1}_bid_size']
                t2_vol = row[f'{token_2}_ask_size']
                qty_1 = current_orders[(token_1, token_2)]['qty_1']
                qty_2 = current_orders[(token_1, token_2)]['qty_2']
                place_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                            'close', 'long', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, z_score,
                                tf, wind, in_, out_, fee_rate, min_order_size, max_order_size, leverage, reason=1, verbose=False)
                flag_out = False

            # --- Закрываем шорт ---
            if z_score < low_out and (token_1, token_2, 'short') in pairs and flag_out:
                t1_price = row[f'{token_1}_ask_price']
                t2_price = row[f'{token_2}_bid_price']
                t1_vol = row[f'{token_1}_ask_size']
                t2_vol = row[f'{token_2}_bid_size']
                qty_1 = current_orders[(token_1, token_2)]['qty_1']
                qty_2 = current_orders[(token_1, token_2)]['qty_2']
                place_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                            'close', 'short', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, z_score,
                                tf, wind, in_, out_, fee_rate, min_order_size, max_order_size, leverage, reason=1, verbose=False)
                flag_out = False

            # --- Проверка стоп-лосса ---
            if (token_1, token_2, 'long') in pairs:
                op_1 = current_orders[(token_1, token_2)]['t1_price']
                op_2 = current_orders[(token_1, token_2)]['t2_price']
                t1_price = row[f'{token_1}_bid_price']
                t2_price = row[f'{token_2}_ask_price']

                sl_price_1 = op_1 - 0.85 * op_1 / leverage
                sl_price_2 = op_2 + 0.85 * op_2 / leverage

                if t1_price < sl_price_1 or t2_price > sl_price_2:
                    qty_1 = current_orders[(token_1, token_2)]['qty_1']
                    qty_2 = current_orders[(token_1, token_2)]['qty_2']
                    t1_vol = row[f'{token_1}_bid_size']
                    t2_vol = row[f'{token_2}_ask_size']

                    place_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                                'close', 'long', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, z_score,
                                tf, wind, in_, out_, fee_rate, min_order_size, max_order_size, leverage, reason=2, verbose=False)

            if (token_1, token_2, 'short') in pairs:
                op_1 = current_orders[(token_1, token_2)]['t1_price']
                op_2 = current_orders[(token_1, token_2)]['t2_price']
                t1_price = row[f'{token_1}_ask_price']
                t2_price = row[f'{token_2}_bid_price']

                sl_price_1 = op_1 + 0.85 * op_1 / leverage
                sl_price_2 = op_2 - 0.85 * op_2 / leverage

                if t1_price > sl_price_1 or t2_price < sl_price_2:
                    qty_1 = current_orders[(token_1, token_2)]['qty_1']
                    qty_2 = current_orders[(token_1, token_2)]['qty_2']
                    t1_vol = row[f'{token_1}_ask_size']
                    t2_vol = row[f'{token_2}_bid_size']

                    place_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                                'close', 'short', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, z_score,
                                tf, wind, in_, out_, fee_rate, min_order_size, max_order_size, leverage, reason=2, verbose=False)

    trades_df = pl.DataFrame(trades)
    if trades_df.is_empty():
        return pl.DataFrame(), dict()

    trades_df = trades_df.with_columns(
        (pl.col('open_time').dt.timestamp() // 1_000_000).alias('open_ts'),
        (pl.col('close_time').dt.timestamp() // 1_000_000).alias('close_ts'),
        (pl.col('close_time') - pl.col('open_time')).alias('duration'),
    )

    metrics = analyze_strategy(trades_df, start_date=start_time, end_date=end_time, initial_balance=200.0)
    return trades_df, metrics

def run_double_tf_backtest(main_df, tf_1, wind_1, tf_2, wind_2, in_1, out_1, in_2, out_2, leverage,
                           max_pairs, max_order_size, fee_rate, start_time, end_time, coin_information):
    tokens_in_position = []
    pairs = []
    current_orders = {}
    trades = []

    all_pairs = [(col.split('_')[0], col.split('_')[1]) for col in main_df.columns
                 if (col.endswith('z_score_1') or col.endswith('z_score_2'))]
    random.shuffle(all_pairs)

    for row in main_df.iter_rows(named=True):
        time = row['time']

        for token_1, token_2 in all_pairs:
            if not row[token_1] or not row[token_2] or not row[f'{token_1}_{token_2}_z_score_1'] or not row[f'{token_1}_{token_2}_z_score_2']:
                continue

            low_in_1 = -in_1
            low_out_1 = -out_1
            high_in_1 = in_1
            high_out_1 = out_1

            low_in_2 = -in_2
            low_out_2 = -out_2
            high_in_2 = in_2
            high_out_2 = out_2

            z_score_1 = row[f'{token_1}_{token_2}_z_score_1']
            z_score_2 = row[f'{token_1}_{token_2}_z_score_2']

            # ----- Проверяем условия для входа в позицию -----
            if (len(pairs) < max_pairs and token_1 not in tokens_in_position and token_2 not in tokens_in_position):

                # --- Входим в лонг ---
                if z_score_1 < low_in_1 and z_score_2 < low_in_2:
                    t1_price = row[f'{token_1}_ask_price']
                    t2_price = row[f'{token_2}_bid_price']
                    t1_vol = row[f'{token_1}_ask_size']
                    t2_vol = row[f'{token_2}_bid_size']
                    qty_1, qty_2 = get_qty(token_1, token_2, t1_price, t2_price, None, coin_information, 2 * max_order_size * leverage,
                              method='usdt_neutral')
                    place_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                        'open', 'long', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, (z_score_1, z_score_2),
                        (tf_1, tf_2), (wind_1, wind_2), (in_1, in_2), (out_1, out_2), fee_rate,
                        min_order_size, max_order_size, leverage, verbose=False)

                # --- Открываем шорт ---
                if z_score_1 > high_in_1 and z_score_2 > high_in_2:
                    t1_price = row[f'{token_1}_bid_price']
                    t2_price = row[f'{token_2}_ask_price']
                    t1_vol = row[f'{token_1}_bid_size']
                    t2_vol = row[f'{token_2}_ask_size']
                    qty_1, qty_2 = get_qty(token_1, token_2, t1_price, t2_price, None, coin_information, 2 * max_order_size * leverage,
                              method='usdt_neutral')
                    place_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                        'open', 'short', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, (z_score_1, z_score_2),
                        (tf_1, tf_2), (wind_1, wind_2), (in_1, in_2), (out_1, out_2), fee_rate,
                        min_order_size, max_order_size, leverage, verbose=False)

            # ----- Проверяем условия для выхода из позиции -----
            # --- Закрываем лонг ---
            if z_score_1 > high_out_1 and z_score_2 > high_out_2 and (token_1, token_2, 'long') in pairs:
                t1_price = row[f'{token_1}_bid_price']
                t2_price = row[f'{token_2}_ask_price']
                t1_vol = row[f'{token_1}_bid_size']
                t2_vol = row[f'{token_2}_ask_size']
                qty_1 = current_orders[(token_1, token_2)]['qty_1']
                qty_2 = current_orders[(token_1, token_2)]['qty_2']
                place_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                    'close', 'long', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, (z_score_1, z_score_2),
                    (tf_1, tf_2), (wind_1, wind_2), (in_1, in_2), (out_1, out_2), fee_rate,
                    min_order_size, max_order_size, leverage, reason=1, verbose=False)

            # --- Закрываем шорт ---
            if z_score_1 < low_out_1 and z_score_2 < low_out_2 and (token_1, token_2, 'short') in pairs:
                t1_price = row[f'{token_1}_ask_price']
                t2_price = row[f'{token_2}_bid_price']
                t1_vol = row[f'{token_1}_ask_size']
                t2_vol = row[f'{token_2}_bid_size']
                qty_1 = current_orders[(token_1, token_2)]['qty_1']
                qty_2 = current_orders[(token_1, token_2)]['qty_2']
                place_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                    'close', 'short', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, (z_score_1, z_score_2),
                    (tf_1, tf_2), (wind_1, wind_2), (in_1, in_2), (out_1, out_2), fee_rate,
                    min_order_size, max_order_size, leverage, reason=1, verbose=False)

            # --- Проверка стоп-лосса ---
            if (token_1, token_2, 'long') in pairs:
                op_1 = current_orders[(token_1, token_2)]['t1_price']
                op_2 = current_orders[(token_1, token_2)]['t2_price']
                t1_price = row[f'{token_1}_bid_price']
                t2_price = row[f'{token_2}_ask_price']

                sl_price_1 = op_1 - 0.85 * op_1 / leverage
                sl_price_2 = op_2 + 0.85 * op_2 / leverage

                if t1_price < sl_price_1 or t2_price > sl_price_2:
                    qty_1 = current_orders[(token_1, token_2)]['qty_1']
                    qty_2 = current_orders[(token_1, token_2)]['qty_2']
                    t1_vol = row[f'{token_1}_bid_size']
                    t2_vol = row[f'{token_2}_ask_size']

                    place_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                        'close', 'long', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, (z_score_1, z_score_2),
                        (tf_1, tf_2), (wind_1, wind_2), (in_1, in_2), (out_1, out_2), fee_rate,
                        min_order_size, max_order_size, leverage, reason=2, verbose=False)

            if (token_1, token_2, 'short') in pairs:
                op_1 = current_orders[(token_1, token_2)]['t1_price']
                op_2 = current_orders[(token_1, token_2)]['t2_price']
                t1_price = row[f'{token_1}_ask_price']
                t2_price = row[f'{token_2}_bid_price']

                sl_price_1 = op_1 + 0.85 * op_1 / leverage
                sl_price_2 = op_2 - 0.85 * op_2 / leverage

                if t1_price > sl_price_1 or t2_price < sl_price_2:
                    qty_1 = current_orders[(token_1, token_2)]['qty_1']
                    qty_2 = current_orders[(token_1, token_2)]['qty_2']
                    t1_vol = row[f'{token_1}_ask_size']
                    t2_vol = row[f'{token_2}_bid_size']

                    place_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                        'close', 'short', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, (z_score_1, z_score_2),
                        (tf_1, tf_2), (wind_1, wind_2), (in_1, in_2), (out_1, out_2), fee_rate,
                        min_order_size, max_order_size, leverage, reason=2, verbose=False)

    trades_df = pl.DataFrame(trades)
    if trades_df.is_empty():
        return pl.DataFrame(), dict()

    trades_df = trades_df.with_columns(
            (pl.col('open_time').dt.timestamp() // 1_000_000).alias('open_ts'),
            (pl.col('close_time').dt.timestamp() // 1_000_000).alias('close_ts'),
            (pl.col('close_time') - pl.col('open_time')).alias('duration'),
            pl.col("tf").list.get(0).alias("tf_1"),
            pl.col("tf").list.get(1).alias("tf_2"),
            pl.col("wind").list.get(0).alias("wind_1"),
            pl.col("wind").list.get(1).alias("wind_2"),
            pl.col("thresh_in").list.get(0).alias("in_1"),
            pl.col("thresh_in").list.get(1).alias("in_2"),
            pl.col("thresh_out").list.get(0).alias("out_1"),
            pl.col("thresh_out").list.get(1).alias("out_2"),
            pl.col("open_z_score").list.get(0).alias("open_z_score_1"),
            pl.col("open_z_score").list.get(1).alias("open_z_score_2"),
            pl.col("close_z_score").list.get(0).alias("close_z_score_1"),
            pl.col("close_z_score").list.get(1).alias("close_z_score_2"),
        ).drop('tf', 'wind', 'thresh_in', 'thresh_out', 'beta', 'open_z_score', 'close_z_score')

    metrics = analyze_strategy(trades_df, start_date=start_time, end_date=end_time, initial_balance=200.0)
    return trades_df, metrics

def select_cols_1tf(df, cointegrated_tokens, tf, wind):
    cols = []

    # Отбираем только нужные строки, чтобы сэкономить память
    for token_1, token_2 in cointegrated_tokens:
        if token_1 not in df.columns or token_2 not in df.columns:
            continue

        cols.extend(['time', token_1, f'{token_1}_size', f'{token_1}_bid_price', f'{token_1}_ask_price',
                        f'{token_1}_bid_size', f'{token_1}_ask_size', token_2, f'{token_2}_size',
                        f'{token_2}_bid_price', f'{token_2}_ask_price', f'{token_2}_bid_size', f'{token_2}_ask_size',
                        f'{token_1}_{token_2}_z_score_{wind}_{tf}'])
    cols = list(set(cols))
    cols = [col for col in cols if col in df.columns]
    cols_to_rename = [col for col in df.columns if (col.endswith(f'_{wind}_{tf}') and col in cols)]
    tail = len(f'_{wind}_{tf}')
    mapping = {c: c[:-tail] for c in cols_to_rename}

    return df.select(cols).rename(mapping)

def select_cols_2tf(df, cointegrated_tokens, tf_1, wind_1, tf_2, wind_2):
    cols = []

    # Отбираем только нужные строки, чтобы сэкономить память
    for token_1, token_2 in cointegrated_tokens:
        if token_1 not in df.columns or token_2 not in df.columns:
            continue

        cols.extend(['time', token_1, f'{token_1}_size', f'{token_1}_bid_price', f'{token_1}_ask_price',
                        f'{token_1}_bid_size', f'{token_1}_ask_size', token_2, f'{token_2}_size',
                        f'{token_2}_bid_price', f'{token_2}_ask_price', f'{token_2}_bid_size', f'{token_2}_ask_size',
                        f'{token_1}_{token_2}_z_score_{wind_1}_{tf_1}', f'{token_1}_{token_2}_z_score_{wind_2}_{tf_2}'])
    cols = list(set(cols))
    cols = [col for col in cols if col in df.columns]
    cols_to_rename_1 = [col for col in df.columns if (col.endswith(f'_{wind_1}_{tf_1}') and col in cols)]
    cols_to_rename_2 = [col for col in df.columns if (col.endswith(f'_{wind_2}_{tf_2}') and col in cols)]
    tail_1 = len(f'_{wind_1}_{tf_1}')
    mapping_1 = {c: c[:-tail_1] + '_1' for c in cols_to_rename_1}
    tail_2 = len(f'_{wind_2}_{tf_2}')
    mapping_2 = {c: c[:-tail_2] + '_2' for c in cols_to_rename_2}

    return df.select(cols).rename(mapping_1).rename(mapping_2)

if __name__ == '__main__':
    with open("./data/coin_information.pkl", "rb") as f:
        coin_information = pickle.load(f)

    cointegrated_tokens = []
    with open('./bot/config/cointegrated_tokens_test.txt', 'r') as file:
        for line in file:
            a, b = line.strip().split()
            cointegrated_tokens.append((a, b))

    # Загружаем полный датасет
    df = pl.scan_parquet('./data/full.parquet')

    # Выбрасываем столбцы с информацией о спреде, он нам сейчас не нужен
    all_cols = df.collect_schema().names()
    cols_to_drop = [col for col in all_cols if '_spread_' in col]
    df = df.drop(cols_to_drop).collect()

    method = 'lr'
    start_time = datetime(2025, 9, 16, 0, tzinfo=ZoneInfo("Europe/Moscow"))
    end_time = datetime(2025, 9, 26, 0, tzinfo=ZoneInfo("Europe/Moscow"))
    min_order_size = 40
    max_order_size = 50
    max_pairs = 5
    leverage = 2
    fee_rate = 0.00055

    tfs = ('4h', '1h', '5m')
    winds = {'4h': [10, 14, 18, 24],
            '1h': [36, 48, 60, 72, 96],
            '5m': [60, 90, 120, 180, 240, 300]}

    n_tf_params = (1, 1, 2)
    in_params = (1.75, 2.0, 2.25, 2.5)
    out_params = (0.25, 0.5, 0.75, 1.5, 1.75, 2.0, 2.25)
    dist_in_params = (0.1, 0.3, 0.5, 0.75)
    dist_out_params = (0.1, 0.3, 0.5, 0.75)

    metrics_arr = []

    n_iter = 5_000

    with tqdm(total=n_iter, desc="Обработка", unit="iter") as progress_bar:
        for _ in range(n_iter):
            try:
                n_tf = random.choice(n_tf_params)
                trades_df, metrics = None, None

                if n_tf == 1:
                    mode = random.choice(['1_tf_direct', '1_tf_rev_in', '1_tf_rev_out', '1_tf_rev_both'])
                    tf_1 = random.choice(tfs[:2]) # На одиночном таймфрейме игнорируем '5m'
                    wind_1 = random.choice(winds[tf_1])
                    tf_2, wind_2 = 0, 0
                    in_1 = random.choice(in_params)
                    out_1 = random.choice(out_params)
                    in_2, out_2 = 0, 0

                    tdf = select_cols_1tf(df, cointegrated_tokens, tf_1, wind_1)

                    if mode == '1_tf_direct':
                        dist_in, dist_out = 0, 0
                        trades_df, metrics = run_single_tf_backtest(tdf, tf_1, wind_1, in_1, out_1, leverage, max_pairs, max_order_size,
                            fee_rate, start_time, end_time, coin_information)
                    elif mode == '1_tf_rev_in':
                        dist_in = random.choice(dist_in_params)
                        dist_out = 0
                        trades_df, metrics = run_single_tf_backtest_reverse(tdf, tf_1, wind_1, in_1, out_1, dist_in, dist_out, max_pairs, leverage,
                                    max_order_size, fee_rate, start_time, end_time, coin_information,
                                    reverse_in=True, reverse_out=False)
                    elif mode == '1_tf_rev_out':
                        dist_in = 0
                        dist_out = random.choice(dist_out_params)
                        trades_df, metrics = run_single_tf_backtest_reverse(tdf, tf_1, wind_1, in_1, out_1, dist_in, dist_out, max_pairs, leverage,
                                    max_order_size, fee_rate, start_time, end_time, coin_information,
                                    reverse_in=False, reverse_out=True)
                    elif mode == '1_tf_rev_both':
                        dist_in = random.choice(dist_in_params)
                        dist_out = random.choice(dist_out_params)
                        trades_df, metrics = run_single_tf_backtest_reverse(tdf, tf_1, wind_1, in_1, out_1, dist_in, dist_out, max_pairs, leverage,
                                    max_order_size, fee_rate, start_time, end_time, coin_information,
                                    reverse_in=True, reverse_out=True)

                elif n_tf == 2:
                    mode = '2_tf_direct'
                    tf_1, tf_2 = random.choices(tfs, k=2)
                    wind_1 = random.choice(winds[tf_1])
                    wind_2 = random.choice(winds[tf_2])
                    dist_in, dist_out = 0, 0
                    in_1 = random.choice(in_params)
                    out_1 = random.choice(out_params)
                    in_2 = random.choice(in_params)
                    out_2 = random.choice(out_params)

                    if tf_1 == tf_2 and wind_1 == wind_2:
                        continue
                    if tf_1 == '5m' and tf_2 == '5m':
                        continue

                    tdf = select_cols_2tf(df, cointegrated_tokens, tf_1=tf_1, wind_1=wind_1, tf_2=tf_2, wind_2=wind_2)
                    trades_df, metrics = run_double_tf_backtest(tdf, tf_1, wind_1,
                                                    tf_2, wind_2, in_1, out_1, in_2, out_2, leverage,
                                                    max_pairs, max_order_size, fee_rate, start_time, end_time,
                                                    coin_information)
                else:
                    continue

                if not metrics:
                    continue

                log = {'n_tf': n_tf, 'tf_1': tf_1, 'tf_2': tf_2, 'wind_1': wind_1, 'wind_2': wind_2,
                        'in_1': in_1, 'in_2': in_2, 'out_1': out_1, 'out_2': out_2,
                        'dist_in': dist_in, 'dist_out': dist_out,
                        'n_trades': metrics['n_trades'],
                        'duration_min': metrics['duration_min'].total_seconds(), 'duration_max': metrics['duration_max'].total_seconds(),
                        'duration_avg': metrics['duration_avg'].total_seconds(), 'stop_losses': metrics['stop_losses'],
                        'liquidations': metrics['liquidations'],
                        'profit': metrics['profit'], 'max_drawdown': metrics['max_drawdown'], 'max_profit': metrics['max_profit'],
                        'max_loss': metrics['max_loss'], 'avg_profit': metrics['avg_profit'], 'profit_std': metrics['profit_std'],
                        'profit_ratio': metrics['profit_ratio']}
                json_log = json.dumps(log, default=float, ensure_ascii=False)
                with open('./logs/backtest_res.jsonl', 'a', encoding='utf-8') as f:
                    f.write(json_log + '\n')

                # with open('./logs/trades_bt.jsonl', 'a', encoding='utf-8') as f:
                #     for trade in trades_df.to_dicts():
                #         trade.pop('open_time')
                #         trade.pop('close_time')
                #         trade.pop('duration')

                #         trade_log = json.dumps(trade, default=float, ensure_ascii=False)
                #         f.write(trade_log + '\n')

                if n_tf == 1 and mode[5:] == 'direct':
                    tqdm.write(f'n_tf: {n_tf} ({mode[5:]:>8}); tf: {tf_1}; wind: {wind_1:>3}; in: {in_1:>4}; \
out: {out_1:>4}; profit: {metrics['profit']:.1f}')

                elif n_tf == 1 and mode[5:] == 'rev_in':
                    tqdm.write(f'n_tf: {n_tf} ({mode[5:]:>8}); tf: {tf_1}; wind: {wind_1:>3}; in: {in_1:>4}; \
out: {out_1:>4}; dist_in: {dist_in:>4}, profit: {metrics['profit']:.1f}')

                elif n_tf == 1 and mode[5:] == 'rev_out':
                    tqdm.write(f'n_tf: {n_tf} ({mode[5:]:>8}); tf: {tf_1}; wind: {wind_1:>3}; in: {in_1:>4}; \
out: {out_1:>4}; dist_out: {dist_out:>4}, profit: {metrics['profit']:.1f}')

                elif n_tf == 1 and mode[5:] == 'rev_out':
                    tqdm.write(f'n_tf: {n_tf} ({mode[5:]:>8}); tf: {tf_1}; wind: {wind_1:>3}; in: {in_1:>4}; \
out: {out_1:>4}; dist_in: {dist_in:>4}, dist_out: {dist_out:>4}, profit: {metrics['profit']:.1f}')

                elif n_tf == 2:
                    tqdm.write(f'n_tf: {n_tf} ({mode[5:]:>8}); tf_1: {tf_1}; wind_1: {wind_1:>3}; tf_2: {tf_2}; \
wind_2: {wind_2:>3}; in_1: {in_1:>4}; out_1: {out_1:>4}; in_2: {in_2:>4}; out_2: {out_2:>4}; \
profit: {metrics['profit']:.1f}')


                metrics_arr.append(log)
                progress_bar.update(1)
            except Exception as err:
                continue
