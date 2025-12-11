import polars as pl
import polars_ols as pls
from datetime import datetime
import numpy as np
from numba import njit
from numba.typed import List as NumbaList
import random

from bot.analysis.strategy_analysis import analyze_strategy
from bot.utils.pair_trading import calculate_profit, get_qty, get_dist_zscore

from bot.core.db.postgres_manager import DBManager
from bot.config.credentials import host, user, password, db_name
db_params = {'host': host, 'user': user, 'password': password, 'dbname': db_name}
db_manager = DBManager(db_params)

SIG_NONE, SIG_LONG_OPEN, SIG_SHORT_OPEN, SIG_LONG_CLOSE, SIG_SHORT_CLOSE = 0, 1, 2, 3, 4
POS_NONE, POS_LONG, POS_SHORT = 0, 1, 2
REASON_NONE, REASON_THRESHOLD, REASON_STOPLOSS, REASON_LIQ, REASON_FORCE = 0, 1, 2, 3, 4
LIQ_NONE, LIQ_LONG, LIQ_SHORT = 0, 1, 2
EV_TYPE_OPEN, EV_TYPE_CLOSE, EV_TYPE_SL, EV_TYPE_LIQ, EV_TYPE_FORCE = 1, 2, 3, 4, 5
USDT_NEUT, VOL_NEUT = 0, 1

try:
    # Проверяем, есть ли IPython и работаем ли мы в ноутбуке
    from IPython import get_ipython
    shell = get_ipython().__class__.__name__
    if shell == "ZMQInteractiveShell":  # Jupyter Notebook или JupyterLab
        from tqdm.notebook import tqdm
    else:  # IPython в терминале или обычный Python
        from tqdm import tqdm
except Exception:
    # Если IPython не установлен — значит точно CLI
    from tqdm import tqdm



@njit("float64(float64, float64)", fastmath=True, cache=True)
def _round(value, dp):
    return np.floor(value / dp) * dp

@njit(fastmath=True, cache=True)
def backtest_fast(time_arr, z_score, currspr_arr, spread_arr, std_arr, bid_1, ask_1, bid_2, ask_2,
            dp_1, dp_2, thresh_low_in, thresh_low_out,
            thresh_high_in, thresh_high_out, long_possible, short_possible,
            dist_in, dist_out, balance, order_size, qty_method, std_1, std_2,
            fee_rate,  sl_std, sl_dist, sl_method, sl_seconds=0,
            open_method=0, close_method=0, leverage=1, force_close=0):

    n = z_score.shape[0]
    total_balance = balance

    signal = SIG_NONE # Текущее состояние ордера
    reason = REASON_NONE # Причина закрытия сделки
    pos_side = POS_NONE

    open_time = 0
    open_price_1 = 0.0
    open_price_2 = 0.0
    qty_1 = 0.0
    qty_2 = 0.0

    liq_status = LIQ_NONE
    long_in_min_value = 0
    short_in_max_value = 0
    long_out_max_value = 0
    short_out_min_value = 0
    sl_counter = 0
    sl_block_long = 0
    sl_block_short = 0
    fixed_mean = 0.0
    fixed_std = 0.0
    fixed_z_score = 0.0

    out = NumbaList()
    events = NumbaList()

    last_chance_time = time_arr[-2] # Timestamp команды на принудительное закрытие позы

    for i in range(n):
        z = z_score[i]
        if total_balance < 0:
            return out, events

        # --- Проверка ликвидации позиции ---
        if pos_side != POS_NONE:
            if pos_side == POS_LONG:
                long_liq_price = open_price_1 * (1 - 1/leverage)
                short_liq_price = open_price_2 * (1 + 1/leverage)
                long_price = bid_1[i]    # цена закрытия long — бид по token_1
                short_price = ask_2[i]   # цена закрытия short — аск по token_2
            elif pos_side == POS_SHORT:
                long_liq_price = open_price_2 * (1 - 1/leverage)
                short_liq_price = open_price_1 * (1 + 1/leverage)
                long_price = bid_2[i]    # цена закрытия long — бид по token_2
                short_price = ask_1[i]   # цена закрытия short — аск по token_1

            if short_price > short_liq_price:
                liq_status = LIQ_SHORT
            elif long_price < long_liq_price:
                liq_status = LIQ_LONG

            if liq_status != LIQ_NONE:
                # Рассчитываем цены закрытия по стандартной логике стороны:
                if pos_side == POS_LONG:
                    price_1 = bid_1[i]
                    price_2 = ask_2[i]
                else:  # POS_SHORT
                    price_1 = ask_1[i]
                    price_2 = bid_2[i]

                usdt_open_1 = qty_1 * open_price_1
                usdt_open_2 = qty_2 * open_price_2
                open_fee_1 = usdt_open_1 * fee_rate
                open_fee_2 = usdt_open_2 * fee_rate

                usdt_close_1 = qty_1 * price_1
                usdt_close_2 = qty_2 * price_2
                close_fee_1 = usdt_close_1 * fee_rate
                close_fee_2 = usdt_close_2 * fee_rate

                if pos_side == POS_LONG:
                    profit_1 = usdt_close_1 - usdt_open_1 - open_fee_1 - close_fee_1
                    profit_2 = usdt_open_2 - usdt_close_2 - open_fee_2 - close_fee_2
                else:
                    profit_1 = usdt_open_1 - usdt_close_1 - open_fee_1 - close_fee_1
                    profit_2 = usdt_close_2 - usdt_open_2 - open_fee_2 - close_fee_2

                if liq_status == LIQ_LONG:
                    profit_1 = -usdt_open_1  # потеря всей long-ноги
                    close_fee_1 = 0.0
                elif liq_status == LIQ_SHORT:
                    profit_2 = -usdt_open_2  # потеря всей short-ноги
                    close_fee_2 = 0.0

                fees = open_fee_1 + open_fee_2 + close_fee_1 + close_fee_2
                total_profit = profit_1 + profit_2
                total_balance += total_profit

                reason = REASON_LIQ

                events.append((EV_TYPE_LIQ, open_time, time_arr[i], qty_1, qty_2,
                           open_price_1, price_1, open_price_2, price_2, pos_side,
                           fees, profit_1, profit_2, total_profit, reason))

                out.append((open_time, time_arr[i], qty_1, qty_2,
                          open_price_1, price_1, open_price_2, price_2,
                          pos_side, fees, profit_1, profit_2, total_profit,
                          reason))

                # Сброс состояния
                signal = SIG_NONE
                reason = REASON_NONE
                pos_side = POS_NONE
                open_time = 0
                open_price_1 = 0.0
                open_price_2 = 0.0
                qty_1 = 0.0
                qty_2 = 0.0
                liq_status = LIQ_NONE

        # --- Открываем ордер, если есть команда на открытие ---
        if signal == SIG_LONG_OPEN:
            open_time = time_arr[i]
            open_price_1 = ask_1[i]
            open_price_2 = bid_2[i]
            if qty_method == USDT_NEUT:
                qty_1 = _round(leverage * order_size / (1.0 + 2.0 * fee_rate) / open_price_1, dp_1)
                qty_2  = _round(leverage * order_size / (1.0 + 2.0 * fee_rate) / open_price_2, dp_2)
            elif qty_method == VOL_NEUT:
                c_eff = 2 * order_size * leverage / (1.0 + 4.0 * fee_rate)
                d1 = c_eff * std_2 / (std_1 + std_2)
                d2 = c_eff * std_1 / (std_1 + std_2)

                qty_1 = _round(d1 / open_price_1, dp_1)
                qty_2 = _round(d2 / open_price_2, dp_2)
            pos_side = POS_LONG
            signal = SIG_NONE
            fixed_mean = spread_arr[i]
            fixed_std = std_arr[i]

            events.append((EV_TYPE_OPEN, open_time, 0, qty_1, qty_2,
                           open_price_1, 0.0, open_price_2, 0.0, pos_side,
                           0.0, 0.0, 0.0, 0.0, 0))

        elif signal == SIG_SHORT_OPEN:
            open_time = time_arr[i]
            open_price_1 = bid_1[i]
            open_price_2 = ask_2[i]
            if qty_method == USDT_NEUT:
                qty_1 = _round(leverage * order_size / (1.0 + 2.0 * fee_rate) / open_price_1, dp_1)
                qty_2 = _round(leverage * order_size / (1.0 + 2.0 * fee_rate) / open_price_2, dp_2)
            elif qty_method == VOL_NEUT:
                c_eff = 2 * order_size * leverage / (1.0 + 4.0 * fee_rate)
                d1 = c_eff * std_2 / (std_1 + std_2)
                d2 = c_eff * std_1 / (std_1 + std_2)

                qty_1 = _round(d1 / open_price_1, dp_1)
                qty_2 = _round(d2 / open_price_2, dp_2)
            pos_side = POS_SHORT
            signal = SIG_NONE
            fixed_mean = spread_arr[i]
            fixed_std = std_arr[i]

            events.append((EV_TYPE_OPEN, open_time, 0, qty_1, qty_2,
                           open_price_1, 0.0, open_price_2, 0.0, pos_side,
                           0.0, 0.0, 0.0, 0.0, 0))

        # --- Закрываем ордер, если есть команда на закрытие ---
        elif signal == SIG_LONG_CLOSE or signal == SIG_SHORT_CLOSE:
            if signal == SIG_LONG_CLOSE:
                price_1 = bid_1[i]
                price_2 = ask_2[i]
            else:  # SIG_SHORT_CLOSE
                price_1 = ask_1[i]
                price_2 = bid_2[i]

            usdt_open_1 = qty_1 * open_price_1
            usdt_open_2 = qty_2 * open_price_2
            open_fee_1 = usdt_open_1 * fee_rate
            open_fee_2 = usdt_open_2 * fee_rate

            usdt_close_1 = qty_1 * price_1
            usdt_close_2 = qty_2 * price_2
            close_fee_1 = usdt_close_1 * fee_rate
            close_fee_2 = usdt_close_2 * fee_rate

            fees = open_fee_1 + open_fee_2 + close_fee_1 + close_fee_2

            if signal == SIG_LONG_CLOSE:
                profit_1 = usdt_close_1 - usdt_open_1 - open_fee_1 - close_fee_1
                profit_2 = usdt_open_2 - usdt_close_2 - open_fee_2 - close_fee_2
            else:
                profit_1 = usdt_open_1 - usdt_close_1 - open_fee_1 - close_fee_1
                profit_2 = usdt_close_2 - usdt_open_2 - open_fee_2 - close_fee_2

            total_profit = profit_1 + profit_2
            total_balance += total_profit

            if reason == REASON_THRESHOLD:
                reas = EV_TYPE_CLOSE
            elif reason == REASON_FORCE:
                reas = EV_TYPE_FORCE
            else:
                reas = EV_TYPE_SL

            events.append((reas, open_time, time_arr[i], qty_1, qty_2,
                           open_price_1, price_1, open_price_2, price_2, pos_side,
                           fees, profit_1, profit_2, total_profit, reason))
            out.append((
                open_time, time_arr[i],
                qty_1, qty_2,
                open_price_1, price_1,
                open_price_2, price_2,
                pos_side,
                fees,
                profit_1, profit_2, total_profit,
                reason
            ))


            # Сброс
            signal = SIG_NONE
            reason = REASON_NONE
            pos_side = POS_NONE
            open_time = 0
            open_price_1 = 0.0
            open_price_2 = 0.0
            qty_1 = 0.0
            qty_2 = 0.0
            fixed_mean = 0.0
            fixed_std = 0.0

        # --- Проверяем действующий стоп-лосс счётчик ---
        if sl_counter > 0:
            sl_counter -= 1
            continue


        # --- Проверяем условие входа в сделку ---
        if pos_side == POS_NONE:
            # Проверяем блокировку после стоп-лосса
            if sl_method == 2:
                # Разблокируем стоп-лосс при выходе из зоны лонг-ставки
                if sl_block_long:
                    if z > thresh_low_in + sl_dist:
                        sl_block_long = 0
                # Разблокируем стоп-лосс при выходе из зоны шорт-ставки
                if sl_block_short:
                    if z < thresh_high_in - sl_dist:
                        sl_block_short = 0

            # Прямой способ входа (когда z_score входит в диапазон входа)
            if open_method == 0:
                if z < thresh_low_in and long_possible and not sl_block_long:
                    signal = SIG_LONG_OPEN
                elif z > thresh_high_in and short_possible and not sl_block_short:
                    signal = SIG_SHORT_OPEN
            # Обратный способ входа со слежением (порог входа z_score движется за z_score)
            elif open_method == 1:
                # --- Long ---
                if long_possible and not sl_block_long:
                    # Спред только входит в диапазон открытия позиции
                    if not long_in_min_value and z < thresh_low_in:
                        long_in_min_value = z
                        # print('Инициируем флаг входа в лонг', time_arr[i], z)
                    # Если пара токенов уже отслеживается, обновляем максимумы
                    elif long_in_min_value and z < long_in_min_value:
                        long_in_min_value = z
                    # Если z_score откатывается на dist от максимума, разрешаем открытие позиции
                    elif long_in_min_value and z > long_in_min_value + dist_in and z < thresh_low_in:
                        signal = SIG_LONG_OPEN
                        long_in_min_value = 0
                        # print('Входим в лонг', time_arr[i], z)
                # --- Short ---
                if short_possible and not sl_block_long:
                    # Спред только входит в диапазон открытия позиции
                    if not short_in_max_value and z > thresh_high_in:
                        short_in_max_value = z
                        # print('Инициируем флаг входа в шорт', time_arr[i], z)
                    # Если пара токенов уже отслеживается, обновляем максимумы
                    elif short_in_max_value and z > short_in_max_value:
                        short_in_max_value = z
                    # Если z_score откатывается на dist от максимума, разрешаем открытие позиции
                    elif short_in_max_value and z < short_in_max_value - dist_in and z > thresh_high_in:
                        short_in_max_value = 0
                        signal = SIG_SHORT_OPEN
                        # print('Входим в шорт', time_arr[i], z)
            elif open_method == 2:
                # --- Long ---
                if long_possible and not sl_block_long:
                    # Спред только входит в диапазон открытия позиции
                    if not long_in_min_value and z < thresh_low_in - dist_in:
                        long_in_min_value = 1
                    # Если z_score откатывается на dist от максимума, разрешаем открытие позиции
                    elif long_in_min_value and z > thresh_low_in:
                        signal = SIG_LONG_OPEN
                        long_in_min_value = 0
                # --- Short ---
                if short_possible and not sl_block_long:
                    # Спред только входит в диапазон открытия позиции
                    if not short_in_max_value and z > thresh_high_in + dist_in:
                        short_in_max_value = 1
                    # Если z_score откатывается на dist от максимума, разрешаем открытие позиции
                    elif short_in_max_value and z < thresh_high_in:
                        short_in_max_value = 0
                        signal = SIG_SHORT_OPEN


        # --- Обрабатываем открытую позицию ---
        if pos_side == POS_LONG or pos_side == POS_SHORT:
            avg_1 = (bid_1[i] + ask_1[i]) / 2.0
            avg_2 = (bid_2[i] + ask_2[i]) / 2.0
            fixed_z_score = (currspr_arr[i] - fixed_mean) / fixed_std

            # --- Проверяем стоп-лосс ---
            if abs(fixed_z_score) > sl_std and pos_side == POS_SHORT:
                signal = SIG_SHORT_CLOSE
                reason = REASON_STOPLOSS
                if sl_method == 2:
                    sl_block_short = 1
            elif abs(fixed_z_score) > sl_std and pos_side == POS_LONG:
                signal = SIG_LONG_CLOSE
                reason = REASON_STOPLOSS
                if sl_method == 2:
                    sl_block_long = 1

            if reason == REASON_STOPLOSS and sl_method == 1 and sl_seconds > 0:
                sl_counter = sl_seconds

            # --- Проверяем условие выхода из сделки ---
            if pos_side == POS_SHORT:
                # Принудительное закрытие позы по завершению бектеста
                if force_close and time_arr[i] == last_chance_time:
                    signal = SIG_SHORT_CLOSE
                    reason = REASON_FORCE

                # Прямой способ (когда z_score входит в диапазон выхода)
                if close_method == 0:
                    if z < thresh_low_out:
                        signal = SIG_SHORT_CLOSE
                        reason = REASON_THRESHOLD
                # Выходим из сделки, когда z_score покидает диапазон выхода
                else:
                    # Начинаем отслеживать, когда спред опускается ниже порога выхода
                    if not short_out_min_value and z < thresh_low_out:
                        short_out_min_value = z
                    # Если спред обновляет минимум
                    elif short_out_min_value and z < short_out_min_value:
                        short_out_min_value = z
                    # Если спред откатывается на dist_out от минимума
                    elif short_out_min_value and z > short_out_min_value + dist_out and z < thresh_low_out:
                        signal = SIG_SHORT_CLOSE
                        reason = REASON_THRESHOLD
                        short_out_min_value = 0

            elif pos_side == POS_LONG:
                if force_close and time_arr[i] == last_chance_time:
                    signal = SIG_LONG_CLOSE
                    reason = REASON_FORCE

                # Прямой способ (когда z_score входит в диапазон выхода)
                if close_method == 0:
                    if z > thresh_high_out:
                        signal = SIG_LONG_CLOSE
                        reason = REASON_THRESHOLD
                # Выходим из сделки, когда z_score покидает диапазон выхода
                else:
                    # Начинаем отслеживать, когда спред опускается ниже порога выхода
                    if not long_out_max_value and z > thresh_high_out:
                        long_out_max_value = z
                    # Если спред обновляет максимум
                    elif long_out_max_value and z > long_out_max_value:
                        long_out_max_value = z
                    # Если спред откатывается на dist_out от максимума
                    elif long_out_max_value and z < long_out_max_value - dist_out and z > thresh_high_out:
                        signal = SIG_LONG_CLOSE
                        reason = REASON_THRESHOLD
                        long_out_max_value = 0

    return out, events

def backtest(df, token_1, token_2, dp_1, dp_2, thresh_low_in, thresh_low_out,
            thresh_high_in, thresh_high_out, long_possible, short_possible,
            balance, order_size, qty_method, std_1, std_2,
            fee_rate,  sl_std, sl_dist, sl_method=None, sl_seconds=0,
            open_method='direct', close_method='direct', leverage=1, dist_in=0, dist_out=0,
            force_close=False, verbose=False):
    """


    close_method: Как закрывать позицию. direct - по обычному z_score, fix - по фиксированному
    """

    time_arr = df['ts'].to_numpy()
    z = df["z_score"].to_numpy()
    bid_1 = df[f"{token_1}_bid_price"].to_numpy()
    ask_1 = df[f"{token_1}_ask_price"].to_numpy()
    bid_2 = df[f"{token_2}_bid_price"].to_numpy()
    ask_2 = df[f"{token_2}_ask_price"].to_numpy()
    currspr_arr =df["spread"].to_numpy()
    spread_arr = df["spread_mean"].to_numpy()
    std_arr = df["spread_std"].to_numpy()

    spread_arr = spread_arr[~np.isnan(spread_arr)]
    std_arr = std_arr[~np.isnan(std_arr)]

    sl_map = {None: 0, 'counter': 1, 'leave': 2}
    qty_map = {'usdt_neutral': USDT_NEUT, 'vol_neutral': VOL_NEUT}
    close_map = {'direct': 0, 'reverse': 1}
    open_map = {'direct': 0, 'reverse_dynamic': 1, 'reverse_static': 2}

    if qty_method == 'vol_neutral' and (std_1 is None or std_2 is None):
        raise Exception('При использовании vol_neutral необходимо задать std_1 и std_2')

    force_close = 1 if force_close else 0

    res, events = backtest_fast(time_arr, z, currspr_arr, spread_arr, std_arr, bid_1, ask_1, bid_2, ask_2,
            dp_1, dp_2,
            thresh_low_in=thresh_low_in, thresh_high_in=thresh_high_in,
            thresh_low_out=thresh_low_out, thresh_high_out=thresh_high_out,
            long_possible=long_possible, short_possible=short_possible,
            dist_in=dist_in, dist_out=dist_out,
            balance=balance, order_size=order_size, qty_method=qty_map[qty_method],
            std_1=std_1, std_2=std_2, fee_rate=fee_rate,
            sl_std=sl_std, sl_dist=sl_dist, sl_method=sl_map[sl_method], sl_seconds=sl_seconds,
            open_method=open_map[open_method], close_method=close_map[close_method],
            leverage=leverage, force_close=force_close)

    trades_df = pl.DataFrame(res, schema=[
            "open_ts", "close_ts", "qty_1", "qty_2", "open_price_1", "close_price_1",
            "open_price_2", "close_price_2", "pos_side", "fees", "profit_1", "profit_2",
            "total_profit", "reason"], orient="row")

    if trades_df.height > 0:
        profit = trades_df['total_profit'].sum()

        trades_df = trades_df.with_columns(
            pl.from_epoch(pl.col("open_ts"), time_unit="s"
                        ).dt.convert_time_zone("Europe/Moscow"
                        ).alias('open_time'),
            pl.from_epoch(pl.col("close_ts"), time_unit="s"
                        ).dt.convert_time_zone("Europe/Moscow"
                        ).alias('close_time'),
        ).select('open_time', 'open_ts', 'close_time', 'close_ts', 'qty_1', 'qty_2',
                 'open_price_1', 'close_price_1', 'open_price_2', 'close_price_2',
                 'pos_side', 'fees', 'profit_1', 'profit_2', 'total_profit', 'reason')
    else:
        profit = 0

    if verbose >= 1:
        print(f'low_in: {thresh_low_in}; high_in: {thresh_high_in}; \
low_out: {thresh_low_out}; high_out: {thresh_high_out}. n_trades: {trades_df.height}.\
Profit: {profit:.2f}.')
    if verbose >= 2:
        for ev in events:
            etype, open_time, close_time, qty_1, qty_2, open_price_1, price_1, open_price_2, price_2, pos_side, fees, profit_1, profit_2, total_profit, reason = ev

            open_date = datetime.fromtimestamp(open_time).strftime('%Y-%m-%d %H:%M:%S')
            close_date = datetime.fromtimestamp(close_time).strftime('%Y-%m-%d %H:%M:%S')

            side_1 = 'Buy' if pos_side == 1 else 'Sell'
            side_2 = 'Buy' if pos_side == 2 else 'Sell'

            if dp_1 >= 1:
                qty_1 = int(qty_1)
            else:
                qty_1 = _round(qty_1, dp_1)

            if dp_2 >= 1:
                qty_2 = int(qty_2)
            else:
                qty_2 = _round(qty_2, dp_2)

            if etype == 1:
                print(f"[ Open] {open_date}. {side_1} {round(qty_1, 6)} {token_1}, {side_2} {round(qty_2, 6)} {token_2}")
            elif etype == 2:
                print(f"[Close] {close_date}. Profit: {total_profit:.2f}")
            elif etype == 3:
                print(f"[STOP LOSS!] {close_date}. Profit: {total_profit:.2f}")
            elif etype == 4:
                print(f"[LIQUIDATION!] {close_date}. Profit: {total_profit:.2f}")
            elif etype == 5:
                print(f"[FORCE] {close_date}. Profit: {total_profit:.2f}")

    return trades_df


def place_demo_order(tokens_in_position, pairs, current_orders, trades,
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

def run_single_tf_backtest(main_df, tf, wind, in_, out_, leverage, max_pairs, min_order_size, max_order_size,
                           qty_method, fee_rate, start_time, end_time, sl_ratio,
                           coin_information, force_close=False, verbose=False):
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
            if (len(pairs) < max_pairs and
                token_1 not in tokens_in_position and
                token_2 not in tokens_in_position):

                # --- Входим в лонг ---
                if z_score < low_in:
                    t1_price = row[f'{token_1}_ask_price']
                    t2_price = row[f'{token_2}_bid_price']
                    t1_vol = row[f'{token_1}_ask_size']
                    t2_vol = row[f'{token_2}_bid_size']
                    qty_1, qty_2 = get_qty(token_1, token_2, t1_price, t2_price, None, coin_information, 2 * max_order_size * leverage,
                              method=qty_method)
                    place_demo_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                                'open', 'long', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, z_score,
                                tf, wind, in_, out_, fee_rate, min_order_size, max_order_size, leverage, verbose=verbose)

                # --- Открываем шорт ---
                if z_score > high_in:
                    t1_price = row[f'{token_1}_bid_price']
                    t2_price = row[f'{token_2}_ask_price']
                    t1_vol = row[f'{token_1}_bid_size']
                    t2_vol = row[f'{token_2}_ask_size']
                    qty_1, qty_2 = get_qty(token_1, token_2, t1_price, t2_price, None, coin_information, 2 * max_order_size * leverage,
                              method=qty_method)
                    place_demo_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                                'open', 'short', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, z_score,
                                tf, wind, in_, out_, fee_rate, min_order_size, max_order_size, leverage, verbose=verbose)

            # ----- Проверяем условия для выхода из позиции -----
            # --- Закрываем лонг ---
            if z_score > high_out and (token_1, token_2, 'long') in pairs:
                t1_price = row[f'{token_1}_bid_price']
                t2_price = row[f'{token_2}_ask_price']
                t1_vol = row[f'{token_1}_bid_size']
                t2_vol = row[f'{token_2}_ask_size']
                qty_1 = current_orders[(token_1, token_2)]['qty_1']
                qty_2 = current_orders[(token_1, token_2)]['qty_2']
                place_demo_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                            'close', 'long', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, z_score,
                                tf, wind, in_, out_, fee_rate, min_order_size, max_order_size, leverage, reason=1, verbose=verbose)

            # --- Закрываем шорт ---
            if z_score < low_out and (token_1, token_2, 'short') in pairs:
                t1_price = row[f'{token_1}_ask_price']
                t2_price = row[f'{token_2}_bid_price']
                t1_vol = row[f'{token_1}_ask_size']
                t2_vol = row[f'{token_2}_bid_size']
                qty_1 = current_orders[(token_1, token_2)]['qty_1']
                qty_2 = current_orders[(token_1, token_2)]['qty_2']
                place_demo_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                            'close', 'short', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, z_score,
                                tf, wind, in_, out_, fee_rate, min_order_size, max_order_size, leverage, reason=1,
                                verbose=verbose)

            # --- Проверка стоп-лосса ---
            if (token_1, token_2, 'long') in pairs:
                qty_1 = current_orders[(token_1, token_2)]['qty_1']
                qty_2 = current_orders[(token_1, token_2)]['qty_2']
                op_1 = current_orders[(token_1, token_2)]['t1_price']
                op_2 = current_orders[(token_1, token_2)]['t2_price']
                t1_price = row[f'{token_1}_bid_price']
                t2_price = row[f'{token_2}_ask_price']

                sl_price_1 = op_1 - 0.85 * op_1 / leverage
                sl_price_2 = op_2 + 0.85 * op_2 / leverage

                pos_size = (qty_1 * op_1 + qty_2 * op_2) / leverage

                pr_1 = calculate_profit(open_price=op_1, close_price=t1_price, n_coins=qty_1, side='long')
                pr_2 = calculate_profit(open_price=op_2, close_price=t2_price, n_coins=qty_2, side='short')
                total_pr = pr_1 + pr_2

                if t1_price < sl_price_1 or t2_price > sl_price_2 or total_pr < -sl_ratio * pos_size:
                    t1_vol = row[f'{token_1}_bid_size']
                    t2_vol = row[f'{token_2}_ask_size']

                    place_demo_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                                'close', 'long', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, z_score,
                                tf, wind, in_, out_, fee_rate, min_order_size, max_order_size, leverage, reason=2, verbose=verbose)

            if (token_1, token_2, 'short') in pairs:
                qty_1 = current_orders[(token_1, token_2)]['qty_1']
                qty_2 = current_orders[(token_1, token_2)]['qty_2']
                op_1 = current_orders[(token_1, token_2)]['t1_price']
                op_2 = current_orders[(token_1, token_2)]['t2_price']
                t1_price = row[f'{token_1}_ask_price']
                t2_price = row[f'{token_2}_bid_price']

                sl_price_1 = op_1 + 0.85 * op_1 / leverage
                sl_price_2 = op_2 - 0.85 * op_2 / leverage

                pos_size = (qty_1 * op_1 + qty_2 * op_2) / leverage

                pr_1 = calculate_profit(open_price=op_1, close_price=t1_price, n_coins=qty_1, side='short')
                pr_2 = calculate_profit(open_price=op_2, close_price=t2_price, n_coins=qty_2, side='long')
                total_pr = pr_1 + pr_2

                if t1_price > sl_price_1 or t2_price < sl_price_2 or total_pr < -sl_ratio * pos_size:
                    t1_vol = row[f'{token_1}_ask_size']
                    t2_vol = row[f'{token_2}_bid_size']

                    place_demo_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                                'close', 'short', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, z_score,
                                tf, wind, in_, out_, fee_rate, min_order_size, max_order_size, leverage, reason=2, verbose=verbose)

    if verbose:
        for pair, info in current_orders.items():
            print(pair, info['time'])

    if force_close:
        row = main_df[-1]
        time = row['time'][0]
        orders_to_close = current_orders.copy()

        for pair, info in orders_to_close.items():
            z_score = row[f'{token_1}_{token_2}_z_score'][0]
            token_1 = pair[0]
            token_2 = pair[1]
            pos_side = info['pos_side']
            qty_1 = info['qty_1']
            qty_2 = info['qty_2']
            t1_vol = 1_000_000
            t2_vol = 1_000_000
            t1_price = row[f'{token_1}_bid_price'][0] if pos_side == 'long' else row[f'{token_1}_ask_price'][0]
            t2_price = row[f'{token_2}_ask_price'][0] if pos_side == 'long' else row[f'{token_2}_bid_price'][0]

            place_demo_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                            'close', pos_side, qty_1, qty_2, t1_price, t2_price,
                            t1_vol, t2_vol, None, z_score, tf, wind, in_, out_, fee_rate,
                            min_order_size, max_order_size, leverage, reason=1,
                            verbose=verbose)


    trades_df = pl.DataFrame(trades)
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
                                   min_order_size, max_order_size, qty_method, fee_rate, start_time, end_time,
                                   sl_ratio, coin_information, reverse_in=True, reverse_out=False,
                                   verbose=False):
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
                              method=qty_method)
                    place_demo_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                                'open', 'long', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, z_score,
                                tf, wind, in_, out_, fee_rate, min_order_size, max_order_size, leverage, verbose=verbose)
                    flag_in = False

                # --- Открываем шорт ---
                if z_score > high_in:
                    t1_price = row[f'{token_1}_bid_price']
                    t2_price = row[f'{token_2}_ask_price']
                    t1_vol = row[f'{token_1}_bid_size']
                    t2_vol = row[f'{token_2}_ask_size']
                    qty_1, qty_2 = get_qty(token_1, token_2, t1_price, t2_price, None, coin_information, 2 * max_order_size * leverage,
                              method=qty_method)
                    place_demo_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                                'open', 'short', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, z_score,
                                tf, wind, in_, out_, fee_rate, min_order_size, max_order_size, leverage, verbose=verbose)
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
                place_demo_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                            'close', 'long', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, z_score,
                                tf, wind, in_, out_, fee_rate, min_order_size, max_order_size, leverage, reason=1, verbose=verbose)
                flag_out = False

            # --- Закрываем шорт ---
            if z_score < low_out and (token_1, token_2, 'short') in pairs and flag_out:
                t1_price = row[f'{token_1}_ask_price']
                t2_price = row[f'{token_2}_bid_price']
                t1_vol = row[f'{token_1}_ask_size']
                t2_vol = row[f'{token_2}_bid_size']
                qty_1 = current_orders[(token_1, token_2)]['qty_1']
                qty_2 = current_orders[(token_1, token_2)]['qty_2']
                place_demo_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                            'close', 'short', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, z_score,
                                tf, wind, in_, out_, fee_rate, min_order_size, max_order_size, leverage, reason=1, verbose=verbose)
                flag_out = False

            # --- Проверка стоп-лосса ---
            if (token_1, token_2, 'long') in pairs:
                qty_1 = current_orders[(token_1, token_2)]['qty_1']
                qty_2 = current_orders[(token_1, token_2)]['qty_2']
                op_1 = current_orders[(token_1, token_2)]['t1_price']
                op_2 = current_orders[(token_1, token_2)]['t2_price']
                t1_price = row[f'{token_1}_bid_price']
                t2_price = row[f'{token_2}_ask_price']

                sl_price_1 = op_1 - 0.85 * op_1 / leverage
                sl_price_2 = op_2 + 0.85 * op_2 / leverage

                pos_size = (qty_1 * op_1 + qty_2 * op_2) / leverage

                pr_1 = calculate_profit(open_price=op_1, close_price=t1_price, n_coins=qty_1, side='long')
                pr_2 = calculate_profit(open_price=op_2, close_price=t2_price, n_coins=qty_2, side='short')
                total_pr = pr_1 + pr_2

                if t1_price < sl_price_1 or t2_price > sl_price_2 or total_pr < -sl_ratio * pos_size:

                    t1_vol = row[f'{token_1}_bid_size']
                    t2_vol = row[f'{token_2}_ask_size']

                    place_demo_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                                'close', 'long', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, z_score,
                                tf, wind, in_, out_, fee_rate, min_order_size, max_order_size, leverage, reason=2, verbose=verbose)

            if (token_1, token_2, 'short') in pairs:
                qty_1 = current_orders[(token_1, token_2)]['qty_1']
                qty_2 = current_orders[(token_1, token_2)]['qty_2']
                op_1 = current_orders[(token_1, token_2)]['t1_price']
                op_2 = current_orders[(token_1, token_2)]['t2_price']
                t1_price = row[f'{token_1}_ask_price']
                t2_price = row[f'{token_2}_bid_price']

                sl_price_1 = op_1 + 0.85 * op_1 / leverage
                sl_price_2 = op_2 - 0.85 * op_2 / leverage

                pos_size = (qty_1 * op_1 + qty_2 * op_2) / leverage

                pr_1 = calculate_profit(open_price=op_1, close_price=t1_price, n_coins=qty_1, side='short')
                pr_2 = calculate_profit(open_price=op_2, close_price=t2_price, n_coins=qty_2, side='long')
                total_pr = pr_1 + pr_2

                if t1_price > sl_price_1 or t2_price < sl_price_2 or total_pr < -sl_ratio * pos_size:
                    t1_vol = row[f'{token_1}_ask_size']
                    t2_vol = row[f'{token_2}_bid_size']

                    place_demo_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                                'close', 'short', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, z_score,
                                tf, wind, in_, out_, fee_rate, min_order_size, max_order_size, leverage, reason=2, verbose=verbose)

    if verbose:
        for pair, info in current_orders.items():
            print(pair, info['time'])

    trades_df = pl.DataFrame(trades)
    trades_df = trades_df.with_columns(
        (pl.col('open_time').dt.timestamp() // 1_000_000).alias('open_ts'),
        (pl.col('close_time').dt.timestamp() // 1_000_000).alias('close_ts'),
        (pl.col('close_time') - pl.col('open_time')).alias('duration'),
    )

    metrics = analyze_strategy(trades_df, start_date=start_time, end_date=end_time, initial_balance=200.0)
    return trades_df, metrics

def run_double_tf_backtest(main_df, tf_1, wind_1, tf_2, wind_2, in_1, out_1, in_2, out_2, leverage,
                           max_pairs, min_order_size, max_order_size, fee_rate, start_time, end_time,
                           sl_ratio, coin_information, verbose=False):
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
                    place_demo_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                        'open', 'long', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, (z_score_1, z_score_2),
                        (tf_1, tf_2), (wind_1, wind_2), (in_1, in_2), (out_1, out_2), fee_rate,
                        min_order_size, max_order_size, leverage, verbose=verbose)

                # --- Открываем шорт ---
                if z_score_1 > high_in_1 and z_score_2 > high_in_2:
                    t1_price = row[f'{token_1}_bid_price']
                    t2_price = row[f'{token_2}_ask_price']
                    t1_vol = row[f'{token_1}_bid_size']
                    t2_vol = row[f'{token_2}_ask_size']
                    qty_1, qty_2 = get_qty(token_1, token_2, t1_price, t2_price, None, coin_information, 2 * max_order_size * leverage,
                              method='usdt_neutral')
                    place_demo_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                        'open', 'short', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, (z_score_1, z_score_2),
                        (tf_1, tf_2), (wind_1, wind_2), (in_1, in_2), (out_1, out_2), fee_rate,
                        min_order_size, max_order_size, leverage, verbose=verbose)

            # ----- Проверяем условия для выхода из позиции -----
            # --- Закрываем лонг ---
            if z_score_1 > high_out_1 and z_score_2 > high_out_2 and (token_1, token_2, 'long') in pairs:
                t1_price = row[f'{token_1}_bid_price']
                t2_price = row[f'{token_2}_ask_price']
                t1_vol = row[f'{token_1}_bid_size']
                t2_vol = row[f'{token_2}_ask_size']
                qty_1 = current_orders[(token_1, token_2)]['qty_1']
                qty_2 = current_orders[(token_1, token_2)]['qty_2']
                place_demo_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                    'close', 'long', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, (z_score_1, z_score_2),
                    (tf_1, tf_2), (wind_1, wind_2), (in_1, in_2), (out_1, out_2), fee_rate,
                    min_order_size, max_order_size, leverage, reason=1, verbose=verbose)

            # --- Закрываем шорт ---
            if z_score_1 < low_out_1 and z_score_2 < low_out_2 and (token_1, token_2, 'short') in pairs:
                t1_price = row[f'{token_1}_ask_price']
                t2_price = row[f'{token_2}_bid_price']
                t1_vol = row[f'{token_1}_ask_size']
                t2_vol = row[f'{token_2}_bid_size']
                qty_1 = current_orders[(token_1, token_2)]['qty_1']
                qty_2 = current_orders[(token_1, token_2)]['qty_2']
                place_demo_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                    'close', 'short', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, (z_score_1, z_score_2),
                    (tf_1, tf_2), (wind_1, wind_2), (in_1, in_2), (out_1, out_2), fee_rate,
                    min_order_size, max_order_size, leverage, reason=1, verbose=verbose)

            # --- Проверка стоп-лосса ---
            if (token_1, token_2, 'long') in pairs:
                qty_1 = current_orders[(token_1, token_2)]['qty_1']
                qty_2 = current_orders[(token_1, token_2)]['qty_2']
                op_1 = current_orders[(token_1, token_2)]['t1_price']
                op_2 = current_orders[(token_1, token_2)]['t2_price']
                t1_price = row[f'{token_1}_bid_price']
                t2_price = row[f'{token_2}_ask_price']

                sl_price_1 = op_1 - 0.85 * op_1 / leverage
                sl_price_2 = op_2 + 0.85 * op_2 / leverage

                pos_size = (qty_1 * op_1 + qty_2 * op_2) / leverage

                pr_1 = calculate_profit(open_price=op_1, close_price=t1_price, n_coins=qty_1, side='long')
                pr_2 = calculate_profit(open_price=op_2, close_price=t2_price, n_coins=qty_2, side='short')
                total_pr = pr_1 + pr_2

                if t1_price < sl_price_1 or t2_price > sl_price_2 or total_pr < -sl_ratio * pos_size:
                    t1_vol = row[f'{token_1}_bid_size']
                    t2_vol = row[f'{token_2}_ask_size']

                    place_demo_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                        'close', 'long', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, (z_score_1, z_score_2),
                        (tf_1, tf_2), (wind_1, wind_2), (in_1, in_2), (out_1, out_2), fee_rate,
                        min_order_size, max_order_size, leverage, reason=2, verbose=verbose)

            if (token_1, token_2, 'short') in pairs:
                qty_1 = current_orders[(token_1, token_2)]['qty_1']
                qty_2 = current_orders[(token_1, token_2)]['qty_2']
                op_1 = current_orders[(token_1, token_2)]['t1_price']
                op_2 = current_orders[(token_1, token_2)]['t2_price']
                t1_price = row[f'{token_1}_ask_price']
                t2_price = row[f'{token_2}_bid_price']

                sl_price_1 = op_1 + 0.85 * op_1 / leverage
                sl_price_2 = op_2 - 0.85 * op_2 / leverage

                pos_size = (qty_1 * op_1 + qty_2 * op_2) / leverage

                pr_1 = calculate_profit(open_price=op_1, close_price=t1_price, n_coins=qty_1, side='long')
                pr_2 = calculate_profit(open_price=op_2, close_price=t2_price, n_coins=qty_2, side='short')
                total_pr = pr_1 + pr_2

                if t1_price > sl_price_1 or t2_price < sl_price_2 or total_pr < -sl_ratio * pos_size:
                    t1_vol = row[f'{token_1}_ask_size']
                    t2_vol = row[f'{token_2}_bid_size']

                    place_demo_order(tokens_in_position, pairs, current_orders, trades, time, token_1, token_2,
                        'close', 'short', qty_1, qty_2, t1_price, t2_price, t1_vol, t2_vol, None, (z_score_1, z_score_2),
                        (tf_1, tf_2), (wind_1, wind_2), (in_1, in_2), (out_1, out_2), fee_rate,
                        min_order_size, max_order_size, leverage, reason=2, verbose=verbose)

    trades_df = pl.DataFrame(trades)
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
    cols_to_rename = [col for col in cols if col.endswith(f'_{wind}_{tf}')]
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
    cols_to_rename_1 = [col for col in cols if col.endswith(f'_{wind_1}_{tf_1}')]
    cols_to_rename_2 = [col for col in cols if col.endswith(f'_{wind_2}_{tf_2}')]
    tail_1 = len(f'_{wind_1}_{tf_1}')
    mapping_1 = {c: c[:-tail_1] + '_1' for c in cols_to_rename_1}
    tail_2 = len(f'_{wind_2}_{tf_2}')
    mapping_2 = {c: c[:-tail_2] + '_2' for c in cols_to_rename_2}

    return df.select(cols).rename(mapping_1).rename(mapping_2)
