from bot.core.exchange.http_api import ExchangeManager, BybitRestAPI
from bot.utils.coins import get_step_info, get_price_scale
from bot.core.exchange.trade_api import set_leverage
from bot.utils.files import load_config
from requests.exceptions import Timeout, ConnectionError

from bot.core.db.postgres_manager import DBManager
from bot.config.credentials import host, user, password, db_name

import polars as pl
import numpy as np
from datetime import datetime, timezone
from time import sleep
import pickle
import json

from zoneinfo import ZoneInfo
from datetime import timedelta
from bot.utils.pair_trading import get_lr_zscore, get_dist_zscore
from uuid import uuid4
import math
from functools import lru_cache

def round_down(value: float, dp: float):
    return round(math.floor(value / dp) * dp, 6)

def open_position(token_1, token_2, mode, t1_data, t2_data, side_1, side_2, leverage,
                  min_order, max_order, fee_rate, spread_mean, spread_std, coin_information, db_manager):
    ct = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    t1_qty_step = get_step_info(coin_information, token_1, 'bybit_linear', 'bybit_linear')
    t2_qty_step = get_step_info(coin_information, token_2, 'bybit_linear', 'bybit_linear')
    ps_1 = coin_information['bybit_linear'][token_1]['price_scale']
    ps_2 = coin_information['bybit_linear'][token_2]['price_scale']

    Moscow_TZ = timezone(timedelta(hours=3))
    created_at = datetime.now(Moscow_TZ).strftime('%Y-%m-%d %H:%M:%S')

    price_1 = t1_data['ask_price_0'][0] if side_1 == 'long' else t1_data['bid_price_0'][0]
    price_2 = t2_data['bid_price_0'][0] if side_1 == 'long' else t2_data['ask_price_0'][0]
    t1_vol = t1_data['ask_volume_0'][0] if side_1 == 'long' else t1_data['bid_volume_0'][0]
    t2_vol = t2_data['bid_volume_0'][0] if side_1 == 'long' else t2_data['ask_volume_0'][0]

    t1_avail_usdt = t1_vol / price_1
    t2_avail_usdt = t2_vol / price_2

    avail_usdt = min(t1_avail_usdt, t2_avail_usdt, max_order) * leverage

    if avail_usdt > min_order * leverage:
        if t1_qty_step > t2_qty_step:
            qty_1 = round_down(avail_usdt / (1.0 + 2.0 * fee_rate) / price_1, t1_qty_step)
            qty_2 = round_down(qty_1 * price_1 / (1.0 + 2.0 * fee_rate) / price_2, t2_qty_step)
        else:
            qty_2 = round_down(avail_usdt / (1.0 + 2.0 * fee_rate) / price_2, t2_qty_step)
            qty_1 = round_down(qty_2 * price_2 / (1.0 + 2.0 * fee_rate) / price_1, t1_qty_step)

        usdt_1 = qty_1 * price_1
        usdt_2 = qty_2 * price_2

        db_manager.add_pair_order(token_1, token_2, created_at, mode, side_1, side_2, qty_1, qty_2,
                   price_1, price_2, usdt_1 / leverage, usdt_2 / leverage,
                   spread_mean, spread_std, leverage=leverage, status='opening')

        act_1 = 'buy' if side_1 == 'long' else 'sell'
        act_2 = 'buy' if side_2 == 'long' else 'sell'

        print(f'{ct} [{side_1} open] {act_1} {qty_1} {token_1[:-5]} for {price_1}; {act_2} {qty_2} {token_2[:-5]} for {price_2}')

def check_tokens(token_1: str, token_2: str, current_pairs: pl.DataFrame, stop_list: pl.DataFrame) -> bool:
    active_tokens = current_pairs['token_1'].to_list() + current_pairs['token_2'].to_list()
    in_pos = token_1 in active_tokens or token_2 in active_tokens
    in_stop_list = stop_list.filter(
                        ((pl.col('token_1') == token_1) & (pl.col('token_2') == token_2)) |
                        ((pl.col('token_1') == token_2) & (pl.col('token_2') == token_1))
                    ).height > 0

    if in_pos or in_stop_list:
        return False
    else:
        return True

def get_hist_df(db_manager, start_time):
    hist_df = db_manager.get_orderbooks(interval='1h', start_date=start_time)
    hist_df = hist_df.with_columns(pl.col('price').alias('avg_price'))

    return hist_df

def calculate_profit(open_price, close_price, n_coins, side, fee_rate=0.001):
    usdt_open = n_coins * open_price
    open_fee = usdt_open * fee_rate
    usdt_close = n_coins * close_price
    close_fee = usdt_close * fee_rate

    if side == 'long':
        profit = usdt_close - usdt_open - open_fee - close_fee
    elif side == 'short':
        profit = usdt_open - usdt_close - open_fee - close_fee
    return profit

@lru_cache
def set_leverage_cached(demo, token, leverage):
    try:
        set_leverage(demo=demo, symbol=token + '_USDT', leverage=leverage)
    except Timeout:
        print(f'Проблема с изменением leverage для токена {token}.')


def main(update_leverage):
    config = load_config('./bot/config/config.yaml')
    mode = config['mode']

    if mode == 'demo':
        print('DEMO mode.')
        demo = True
    elif mode == 'real':
        print('========= REAL MONEY mode! =========')
        demo = False
    elif mode == 'test':
        print('TEST mode.')
        demo = True
    else:
        raise NotImplementedError('Неизвестный режим работы бота!')

    open_new_orders = config['open_new_orders']
    min_order = config['min_order']
    max_order = config['max_order']
    max_pairs = config['max_pairs']
    leverage = config['leverage']
    fee_rate = config['fee_rate']

    spr_method = config['spr_method']
    tf = config['tf']
    wind = config['wind']
    open_method = config['open_method']
    thresh_in = config['thresh_in']
    thresh_out = config['thresh_out']
    dist_in = config['dist_in']
    min_alt_zscore = config['min_alt_zscore']

    sl_profit_ratio = config['sl_profit_ratio']
    sl_spread_std = config['sl_spread_std']

    # За сколько последних часов брать историю
    if spr_method == 'dist':
        td = int(tf[0]) * wind + 1
    elif spr_method == 'lr' or spr_method == 'tsl':
        td = int(tf[0]) * wind * 2 + 1

    update_positions_flag = False

    print(f'{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Начинаем работу...')

    # --- Загружаем все коинтегрированные токены ---
    token_pairs = []
    with open('./bot/config/token_pairs.txt', 'r') as file:
        for line in file:
            a, b = line.strip().split()
            token_pairs.append((a, b))

    # --- Загружаем техническую информацию по монетам с биржи ---
    exc_manager = ExchangeManager()
    exc_manager.add_market("bybit_linear", BybitRestAPI('linear'))
    coin_information = exc_manager.get_instrument_data()

    with open("./data/coin_information.pkl", "wb") as f:
        pickle.dump(coin_information, f)

    # --- Инициируем менеджеры, работающие с БД ---
    db_params = {'host': host, 'user': user, 'password': password, 'dbname': db_name}
    db_manager = DBManager(db_params)

    if update_leverage:
        print(f'{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Обновление плечей на бирже ByBit')
        for t1_name, t2_name in token_pairs:
            set_leverage_cached(demo, token=t1_name, leverage=leverage)
            sleep(0.5)
            set_leverage_cached(demo, token=t2_name, leverage=leverage)
            sleep(0.5)

    low_in = -thresh_in
    low_out = -thresh_out
    high_in = thresh_in
    high_out = thresh_out

    curr_tracking_in = dict() # Словарь для отслеживания позиций на вход

    time_now = datetime.now()
    last_zscore_update_time = int(datetime.timestamp(time_now))
    print(f'{time_now.strftime('%Y-%m-%d %H:%M:%S')} Старт основного цикла.')

    while True:
        try:
            zscore_arr = []

            time_now = datetime.now()
            ts = int(datetime.timestamp(time_now))
            ct = time_now.strftime('%Y-%m-%d %H:%M:%S')
            print(f'Контрольное время: {ct}', end='\r')

            # --- Устанавливаем heartbeat отметку ---
            db_manager.set_system_state('market_analyzer')

            # --- Проверяем работу модуля trades_executor ---

            if ts - db_manager.get_system_state('trades_executor') > 20:
                print(f'{ct} Потеряна связь с trades_executor!')
                break

            # --- Подгружаем исторические датафреймы 1 раз в час ---
            end_time = datetime.now().replace(tzinfo=ZoneInfo("Europe/Moscow"))
            start_time = end_time - timedelta(hours = td)

            try:
                last_updates_1h = (datetime.now(ZoneInfo("Europe/Moscow")) - hist_df[-1]['time'][0]).seconds
            except NameError:
                hist_df = get_hist_df(db_manager, start_time)
                last_updates_1h = (datetime.now(ZoneInfo("Europe/Moscow")) - hist_df[-1]['time'][0]).seconds

            if last_updates_1h > 3665: # 1 час 1 минута 5 секунд
                hist_df = get_hist_df(db_manager, start_time)

            # --- Текущие данные ---
            current_data = db_manager.get_table('current_ob', df_type='polars')
            if current_data.is_empty():
                print(f'{ct} current data is empty!')
                sleep(5)
                continue

            current_data = current_data.with_columns(
                    ((pl.col('bid_price_0') + pl.col('ask_price_0')) / 2.0).alias('avg_price')
                )

            if mode == 'real':
                pairs = db_manager.get_table('pairs', df_type='polars')
            elif mode == 'demo':
                pairs = db_manager.get_table('pairs_test', df_type='polars')
            stop_list = db_manager.get_table('stop_list', df_type='polars')
            active_orders = pairs.filter(pl.col('status') == 'active')

            # --- Секундный датафрейм для подсчёта среднего значения ---
            end_t = datetime.now().replace(tzinfo=ZoneInfo("Europe/Moscow"))
            st_t = end_t - timedelta(seconds = 30)

            tick_df = db_manager.get_tick_ob(start_time=st_t).with_columns(
                ((pl.col('bid_price') + pl.col('ask_price')) / 2.0).alias('avg_price')
            ).filter(
                (pl.col('bid_size') * pl.col('bid_price') > min_order) &
                (pl.col('ask_size') * pl.col('ask_price') > min_order)
            )

            # --- Обрабатываем каждую пару токенов ---
            for t1_name, t2_name in token_pairs:
                token_1 = t1_name + '_USDT'
                token_2 = t2_name + '_USDT'

                # --- Проверяем пару на наличие в стоп-листе ---
                if stop_list.filter(
                        ((pl.col('token_1') == token_1) & (pl.col('token_2') == token_2)) |
                        ((pl.col('token_1') == token_2) & (pl.col('token_2') == token_1))
                    ).height > 0:
                    continue

                # --- Обновляем открытые пары и текущие ордеры ---
                if update_positions_flag:
                    if mode == 'real':
                        pairs = db_manager.get_table('pairs', df_type='polars')
                    elif mode == 'demo':
                        pairs = db_manager.get_table('pairs_test', df_type='polars')
                    active_orders = pairs.filter(pl.col('status') == 'active')
                    update_positions_flag = False

                # --- Выбираем из общего датафрейма нужные токены ---
                t1_tick_df = tick_df.filter(pl.col('token') == token_1)
                t2_tick_df = tick_df.filter(pl.col('token') == token_2)

                # --- Проверяем, что датафрейм не пустой ---
                if t1_tick_df.height < 3 or t2_tick_df.height < 3:
                    continue

                # --- Проверяем, что этой пары нет в очереди на закрытие ---
                if pairs.filter((pl.col('token_1') == token_1) &
                                (pl.col('token_2') == token_2) &
                                (pl.col('status') == 'closing')).height > 0:
                    continue

                # Пропускаем пару, если уже открыто максимальное количество позиций,
                # а для этой пары позиция не открыта
                if (pairs.height >= max_pairs and pairs.filter(
                            (pl.col('token_1') == token_1) & (pl.col('token_2') == token_2)
                            ).is_empty()
                    ):
                    continue

                # --- Получаем средние цены за исторический период ---
                token_1_hist_price = hist_df.filter(pl.col('token') == token_1).tail(2 * wind + 1)['avg_price'].to_numpy()
                token_2_hist_price = hist_df.filter(pl.col('token') == token_2).tail(2 * wind + 1)['avg_price'].to_numpy()

                # --- Получаем текущие цены ---
                t1_data = current_data.filter(pl.col('token') == token_1)
                t2_data = current_data.filter(pl.col('token') == token_2)

                # --- Проверка на актуальность текущих цен ---
                try:
                    t1_ts = t1_data['update_ts'].item()
                    t2_ts = t2_data['update_ts'].item()
                except ValueError: # Ситуация, когда после восстановления соединения не все токены успевают обновиться
                    sleep(1)
                    break

                if abs(t1_ts - t2_ts) > 10: # Если разница во времени между двумя ценами больше 10 секунд, пропускаем эту пару
                    continue

                t1_med = np.append(token_1_hist_price, t1_tick_df['avg_price'].median())
                t2_med = np.append(token_2_hist_price, t2_tick_df['avg_price'].median())
                t1_curr = np.append(token_1_hist_price, t1_data['avg_price'][0])
                t2_curr = np.append(token_2_hist_price, t2_data['avg_price'][0])

                _, _, _, dist_zscore = get_dist_zscore(t1_med, t2_med, np.array([wind]))
                dist_zscore = dist_zscore[0]

                lr_spread, lr_spr_mean, lr_spr_std, _, _, lr_zscore = get_lr_zscore(t1_med, t2_med, np.array([wind]))
                _, _, _, _, _, zscore_curr = get_lr_zscore(t1_curr, t2_curr, np.array([wind]))
                z_score_curr = zscore_curr[0]
                lr_spread, lr_spr_mean, lr_spr_std, lr_zscore = lr_spread[0], lr_spr_mean[0], lr_spr_std[0], lr_zscore[0]


                # curr_spread = np.log(t1_data['avg_price'][0]) - np.log(t2_data['avg_price'][0])
                curr_pair = pairs.filter(
                        (pl.col('token_1') == token_1) & (pl.col('token_2') == token_2)
                    )
                if curr_pair.height > 0:
                    fixed_mean = curr_pair['fixed_mean'][0]
                    fixed_std = curr_pair['fixed_std'][0]
                    fixed_z_score = (lr_spread - fixed_mean) / fixed_std


                # ----- Проверяем условия для входа в позицию -----
                if open_new_orders and pairs.height < max_pairs and check_tokens(token_1, token_2, pairs, stop_list):

                    # ----- Вход в позицию на возврате спреда к среднему значению -----
                    if open_method == 'reverse_static':
                        # Если пара токенов входит в диапазон открытия позиции, и она ещё не отслеживается, добавляем в треккинг
                        if abs(lr_zscore) > abs(thresh_in) + dist_in and not (token_1, token_2) in curr_tracking_in:
                            curr_tracking_in[(token_1, token_2)] = 1
                            print(f'{ct} Add to tracking: {token_1} - {token_2}; z_score: {lr_zscore:.2f}')
                        # Если открыто максимальное кол-во позиций, а текущая пара выходит из диапазона входа, убираем из отслеживаемых
                        elif (token_1, token_2) in curr_tracking_in and abs(lr_zscore) < thresh_in and len(pairs) >= max_pairs:
                            curr_tracking_in.pop((token_1, token_2))
                            print(f'{ct} Delete from tracking: {token_1} - {token_2}; z_score: {lr_zscore:.2f}')
                        # Если z_score возвращается ниже отметки in_, входим в позицию
                        elif (token_1, token_2) in curr_tracking_in and abs(lr_zscore) < thresh_in:
                            if lr_zscore > low_in and dist_zscore < -min_alt_zscore:
                                open_position(token_1, token_2, mode, t1_data, t2_data,
                                    'long', 'short', leverage, min_order, max_order, fee_rate,
                                    lr_spr_mean, lr_spr_std, coin_information, db_manager)
                                update_positions_flag = True
                            elif lr_zscore < high_in and dist_zscore > min_alt_zscore:
                                open_position(token_1, token_2, mode, t1_data, t2_data,
                                    'short', 'long', leverage, min_order, max_order, fee_rate,
                                    lr_spr_mean, lr_spr_std, coin_information, db_manager)
                                update_positions_flag = True
                        # Если z_score возвращается ниже отметки in_, но z_score, посчитанный вторым методом, слишком плохой,
                        #   удаляем токен из треккинга
                        elif (token_1, token_2) in curr_tracking_in and abs(lr_zscore) < thresh_in and abs(dist_zscore) < min_alt_zscore:
                            curr_tracking_in.pop((token_1, token_2))
                            print(f'{ct} Delete from tracking: {token_1} - {token_2}; z_score: {lr_zscore:.2f}, z_score_2: {dist_zscore:.2f}')


                    # ----- Прямой вход в позицию при пересечении уровня входа -----
                    elif open_method == 'direct':
                        # Открытие long-позиции по token_1 и short-позиции по token_2
                        if lr_zscore < low_in and z_score_curr < low_in:
                            open_position(token_1, token_2, mode, t1_data, t2_data,
                                    'long', 'short', leverage, min_order, max_order, fee_rate,
                                    lr_spr_mean, lr_spr_std, coin_information, db_manager)
                            update_positions_flag = True
                            break

                        # Открытие short-позиции по token_1 и long-позиции по token_2
                        if lr_zscore > high_in and z_score_curr > high_in:
                            open_position(token_1, token_2, mode, t1_data, t2_data,
                                    'short', 'long', leverage, min_order, max_order, fee_rate,
                                    lr_spr_mean, lr_spr_std, coin_information, db_manager)
                            update_positions_flag = True
                            break


                # ----- Проверяем условия выхода из позиции -----
                opened = active_orders.filter(
                    (pl.col('token_1') == token_1) & (pl.col('token_2') == token_2)
                )

                # --- Добавляем текущий z_score и profit в таблицу БД ---
                if opened.height:
                    t1_op = opened['open_price_1'][0]
                    t2_op = opened['open_price_2'][0]
                    q1 = opened['qty_1'][0]
                    q2 = opened['qty_2'][0]

                    side_1 = opened['side_1'][0]
                    side_2 = opened['side_2'][0]
                    t1_vol = t1_data['bid_volume_0'][0] if side_1 == 'long' else t1_data['ask_volume_0'][0]
                    t2_vol = t2_data['ask_volume_0'][0] if side_1 == 'long' else t2_data['bid_volume_0'][0]

                    curr_profit_1 = calculate_profit(t1_op, t1_tick_df['avg_price'].median(), q1, side_1)
                    curr_profit_2 = calculate_profit(t2_op, t2_tick_df['avg_price'].median(), q2, side_2)

                    curr_profit = curr_profit_1 + curr_profit_2
                    zscore_arr.append((ts, 'bybit', token_1, token_2, curr_profit, lr_zscore, fixed_z_score, lr_spread))

                    # --- Стоп-лосс по профиту ---
                    if curr_profit < -sl_profit_ratio * 2 * max_order:
                        print(f'{ct} {token_1} - {token_2} STOP-LOSS by profit!')
                        db_manager.close_pair_order(mode, token_1, token_2, side_1, 'sl_profit')
                        db_manager.add_pair_to_stop_list(token_1, token_2)
                        update_positions_flag = True
                        break
                    # --- Стоп-лосс по z_score ---
                    if abs(fixed_z_score) > sl_spread_std:
                        print(f'{ct} {token_1} - {token_2} STOP-LOSS by z_score!')
                        db_manager.close_pair_order(mode, token_1, token_2, side_1, 'sl_zscore')
                        db_manager.add_pair_to_stop_list(token_1, token_2)
                        update_positions_flag = True
                        break

                    # --- Выходим из позиции, если позволяют условия ---
                    if t1_vol > q1 and t2_vol > q2:
                        if side_1 == 'long' and lr_zscore > high_out and z_score_curr > high_out:
                            db_manager.close_pair_order(mode, token_1, token_2, side_1, 'target')
                            print(f'{ct} [long close] sell {q1} {token_1}; buy {q2} {token_2}')
                            update_positions_flag = True
                            break
                        elif side_1 == 'short' and lr_zscore < low_out and z_score_curr < low_out:
                            db_manager.close_pair_order(mode, token_1, token_2, side_1, 'target')
                            print(f'{ct} [short close] buy {q1} {token_1}; sell {q2} {token_2}')
                            update_positions_flag = True
                            break

            zscore_upd_time = int(datetime.timestamp(datetime.now()))
            if zscore_upd_time - last_zscore_update_time >= 10:
                db_manager.add_data_to_zscore_history(zscore_arr)
                last_zscore_update_time = zscore_upd_time

            sleep(0.5)

        except KeyboardInterrupt:
            print('\nЗавершение работы.')
            break


if __name__ == '__main__':
    main(update_leverage=True)
