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
from bot.utils.pair_trading import get_lr_zscore
from uuid import uuid4
import math
from functools import lru_cache

def round_down(value: float, dp: float):
    return round(math.floor(value / dp) * dp, 6)

def open_position(token_1, token_2, t1_data, t2_data, side_1, side_2, leverage,
                  min_order, max_order, fee_rate, coin_information, db_manager):
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

        db_manager.add_pair_order(token_1, token_2, created_at, side_1, side_2, qty_1, qty_2,
                   price_1, price_2, usdt_1 / leverage, usdt_2 / leverage, leverage=leverage, status='opening')

        act_1 = 'buy' if side_1 == 'long' else 'sell'
        act_2 = 'buy' if side_2 == 'long' else 'sell'

        print(f'{ct} [{side_1} open] {act_1} {qty_1} {token_1[:-5]} for {price_1}; {act_2} {qty_2} {token_2[:-5]} for {price_2}')

def close_position(token_1, token_2, t1_data, t2_data, side_1, side_2, db_manager):
    t1_vol = t1_data['bid_volume_0'][0] if side_1 == 'long' else t1_data['ask_volume_0'][0]
    t2_vol = t2_data['ask_volume_0'][0] if side_1 == 'long' else t2_data['bid_volume_0'][0]

    current_pairs = db_manager.get_table('pairs', df_type='polars')
    pair = current_pairs.filter((pl.col('token_1') == token_1) & (pl.col('token_2') == token_2))

    open_qty_1 = pair['qty_1'][0]
    open_qty_2 = pair['qty_2'][0]

    if t1_vol > open_qty_1 and t2_vol > open_qty_2:
        db_manager.close_pair_order(token_1, token_2, side_1)

        ct = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        act_1 = 'buy' if side_1 == 'short' else 'sell'
        act_2 = 'buy' if side_2 == 'short' else 'sell'
        print(f'{ct} [{side_1} close] {act_1} {open_qty_1} {token_1[:-5]}; {act_2} {open_qty_2} {token_2[:-5]}')

def check_tokens(token_1: str, token_2: str, current_pairs: pl.DataFrame) -> bool:
    active_tokens = current_pairs['token_1'].to_list() + current_pairs['token_2'].to_list()
    in_pos = token_1 in active_tokens or token_2 in active_tokens

    if in_pos:
        return False
    else:
        return True

def get_hist_df(postgre_manager, start_time):
    hour_1_df = postgre_manager.get_orderbooks(interval='1h', start_date=start_time)
    hour_1_df = hour_1_df.with_columns(pl.col('price').alias('avg_price'))

    hour_4_df = postgre_manager.get_orderbooks(interval='4h', start_date=start_time)
    hour_4_df = hour_4_df.with_columns(pl.col('price').alias('avg_price'))

    return hour_4_df, hour_1_df

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

@lru_cache
def set_leverage_cached(demo, token, leverage):
    try:
        set_leverage(demo=demo, symbol=token + '_USDT', leverage=leverage)
    except Timeout:
        print(f'Проблема с изменением leverage для токена {token}.')

def write_order_log(ts, ct, token_1, token_2, tf, wind, thresh_in, thresh_out, side, action,
                    t1, t2, t1_bid_price, t1_ask_price, t2_bid_price, t2_ask_price,
                    t1_bid_size, t1_ask_size, t2_bid_size, t2_ask_size, qty_1, qty_2,
                    z_score, beta=None):
    """
    Запись сделки в лог файл.
    ts - unix timestamp
    ct - текущее время в формате datetime
    token_1 - название токена_1
    token_2 - название токена_2
    tf - таймфрейм
    wind - размер скользящего окна
    t1 - исторические данные для токена_1
    t2 - исторические данные для токена_2
    t1_df_sec - датафрейм с последними записями токена_1
    t2_df_sec - датафрейм с последними записями токена_1
    z_score - z_score

    На выходе ключи словаря t1_last и t2_last являются последними ценами
    токенов, взятыми из t1_df_sec и t2_df_sec.

    """


    t1 = [round(x, 6) for x in t1.tolist()]
    t2 = [round(x, 6) for x in t2.tolist()]
    z_score = round(float(z_score), 2)
    beta = round(float(beta), 2)

    log = {'ts': ts,
            'ct': ct,
            'token_1': token_1[:-5],
            'token_2': token_2[:-5],
            'tf': tf,
            'wind': wind,
            'thresh_in': thresh_in,
            'thresh_out': thresh_out,
            'side': side,
            'action': action,
            't1': t1,
            't2': t2,
            't1_bid_price': t1_bid_price,
            't1_ask_price': t1_ask_price,
            't2_bid_price': t2_bid_price,
            't2_ask_price': t2_ask_price,
            't1_bid_size': t1_bid_size,
            't1_ask_size': t1_ask_size,
            't2_bid_size': t2_bid_size,
            't2_ask_size': t2_ask_size,
            'qty_1': qty_1,
            'qty_2': qty_2,
            'z_score': z_score,
            'beta': beta
           }
    json_log = json.dumps(log, default=float, ensure_ascii=False)

    with open('./logs/trades.jsonl', 'a', encoding='utf-8') as f:
        f.write(json_log + '\n')

def main():
    exchange = 'bybit'
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

    tf = config['tf']
    wind = config['wind']
    thresh_in = config['thresh_in']
    thresh_out = config['thresh_out']
    td = int(tf[0]) * wind * 2 # За сколько последних часов брать историю

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
    postgre_manager = DBManager(db_params)

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
            postgre_manager.set_system_state('market_analyzer')

            # --- Проверяем работу модуля trades_executor ---

            if ts - postgre_manager.get_system_state('trades_executor') > 20:
                print(f'{ct} Потеряна связь с trades_executor!')
                break

            # --- Подгружаем исторические датафреймы 1 раз в час ---
            end_time = datetime.now().replace(tzinfo=ZoneInfo("Europe/Moscow"))
            start_time = end_time - timedelta(hours = td)

            try:
                last_updates_1h = (datetime.now(ZoneInfo("Europe/Moscow")) - hour_1_df[-1]['time'][0]).seconds
            except NameError:
                hour_4_df, hour_1_df = get_hist_df(postgre_manager, start_time)
                last_updates_1h = (datetime.now(ZoneInfo("Europe/Moscow")) - hour_1_df[-1]['time'][0]).seconds

            if last_updates_1h > 3665: # 1 час 1 минута 5 секунд
                hour_4_df, hour_1_df = get_hist_df(postgre_manager, start_time)

            # --- Текущие данные ---
            current_data = postgre_manager.get_table('current_ob', df_type='polars')
            if current_data.is_empty():
                print(f'{ct} current data is empty!')
                sleep(5)
                continue

            current_data = current_data.with_columns(
                    ((pl.col('bid_price_0') + pl.col('ask_price_0')) / 2.0).alias('avg_price')
                )

            pairs = postgre_manager.get_table('pairs', df_type='polars')
            active_orders = pairs.filter(pl.col('status') == 'active')

            # --- Секундный датафрейм для подсчёта среднего значения ---
            end_t = datetime.now().replace(tzinfo=ZoneInfo("Europe/Moscow"))
            st_t = end_t - timedelta(seconds = 30)

            tick_df = postgre_manager.get_tick_ob(start_time=st_t).with_columns(
                ((pl.col('bid_price') + pl.col('ask_price')) / 2.0).alias('avg_price')
            ).filter(
                (pl.col('bid_size') * pl.col('bid_price') > min_order) &
                (pl.col('ask_size') * pl.col('ask_price') > min_order)
            )

            # --- Обрабатываем каждую пару токенов ---
            for t1_name, t2_name in token_pairs:
                token_1 = t1_name + '_USDT'
                token_2 = t2_name + '_USDT'
                z_score = 0

                # --- Обновляем открытые пары и текущие ордеры ---
                if update_positions_flag:
                    pairs = postgre_manager.get_table('pairs', df_type='polars')
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
                hist_df = hour_1_df if tf == '1h' else hour_4_df

                token_1_hist_price = hist_df.filter(pl.col('token') == token_1).tail(2 * wind + 1)['avg_price'].to_numpy()
                token_2_hist_price = hist_df.filter(pl.col('token') == token_2).tail(2 * wind + 1)['avg_price'].to_numpy()

                # --- Получаем текущие цены ---
                t1_curr_data = current_data.filter(pl.col('token') == token_1)
                t2_curr_data = current_data.filter(pl.col('token') == token_2)

                # --- Проверка на актуальность текущих цен ---
                try:
                    t1_ts = t1_curr_data['update_ts'].item()
                    t2_ts = t2_curr_data['update_ts'].item()
                except ValueError: # Ситуация, когда после восстановления соединения не все токены успевают обновиться
                    sleep(1)
                    break

                if abs(t1_ts - t2_ts) > 10: # Если разница во времени между двумя ценами больше 10 секунд, пропускаем эту пару
                    continue

                t1_med = np.append(token_1_hist_price, t1_tick_df['avg_price'].median())
                t2_med = np.append(token_2_hist_price, t2_tick_df['avg_price'].median())
                t1_curr = np.append(token_1_hist_price, t1_curr_data['avg_price'][0])
                t2_curr = np.append(token_2_hist_price, t2_curr_data['avg_price'][0])

                _, _, _, _, beta, zscore = get_lr_zscore(t1_med, t2_med, np.array([wind]))
                _, _, _, _, beta_curr, zscore_curr = get_lr_zscore(t1_curr, t2_curr, np.array([wind]))
                z_score = zscore[0]
                z_score_curr = zscore_curr[0]
                beta = beta[0]

                # ----- Проверяем условия для входа в позицию -----
                if open_new_orders and pairs.height < max_pairs and check_tokens(token_1, token_2, pairs):
                    # Проверяем открытие long-позиции по token_1 и short-позиции по token_2
                    if zscore < low_in and z_score_curr < low_in:
                        open_position(token_1, token_2, t1_curr_data, t2_curr_data,
                                'long', 'short', leverage, min_order, max_order, fee_rate,
                                coin_information, postgre_manager)
                        update_positions_flag = True
                        break

                    # Проверяем открытие short-позиции по token_1 и long-позиции по token_2
                    if zscore > high_in and z_score_curr > high_in:
                        open_position(token_1, token_2, t1_curr_data, t2_curr_data,
                                'short', 'long', leverage, min_order, max_order, fee_rate,
                                coin_information, postgre_manager)
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

                    curr_profit_1 = calculate_profit(t1_op, t1_tick_df['avg_price'].median(), q1, side_1)
                    curr_profit_2 = calculate_profit(t2_op, t2_tick_df['avg_price'].median(), q2, side_2)

                    curr_profit = curr_profit_1 + curr_profit_2
                    zscore_arr.append((ts, 'bybit', token_1, token_2, curr_profit, z_score))

                # --- Выходим из лонг позиции, если позволяют условия ---
                if opened.height and side_1 == 'long' and zscore > high_out and z_score_curr > high_out:
                    close_position(token_1, token_2, t1_curr_data, t2_curr_data, side_1, side_2, postgre_manager)
                    update_positions_flag = True
                    break

                # --- Выходим из шорт позиции, если позволяют условия ---
                if opened.height and side_1 == 'short' and zscore < low_out and z_score_curr < low_out:
                    close_position(token_1, token_2, t1_curr_data, t2_curr_data, side_1, side_2, postgre_manager)
                    update_positions_flag = True
                    break

            zscore_upd_time = int(datetime.timestamp(datetime.now()))
            if zscore_upd_time - last_zscore_update_time >= 10:
                postgre_manager.add_data_to_zscore_history(zscore_arr)
                last_zscore_update_time = zscore_upd_time

            sleep(0.5)

        except KeyboardInterrupt:
            print('\nЗавершение работы.')
            break


if __name__ == '__main__':
    main()
