import polars as pl
import polars_ols as pls
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import numpy as np
from numba import njit
from numba.typed import List as NumbaList

from bot.core.exchange.http_api import ExchangeManager, BybitRestAPI

from bot.core.db.postgres_manager import DBManager
from bot.config.credentials import host, user, password, db_name
db_params = {'host': host, 'user': user, 'password': password, 'dbname': db_name}
db_manager = DBManager(db_params)

SIG_NONE, SIG_LONG_OPEN, SIG_SHORT_OPEN, SIG_LONG_CLOSE, SIG_SHORT_CLOSE = 0, 1, 2, 3, 4
POS_NONE, POS_LONG, POS_SHORT = 0, 1, 2
REASON_NONE, REASON_THRESHOLD, REASON_STOPLOSS, REASON_LIQ = 0, 1, 2, 3
LIQ_NONE, LIQ_LONG, LIQ_SHORT = 0, 1, 2  # для внутренней логики
EV_TYPE_OPEN, EV_TYPE_CLOSE, EV_TYPE_SL, EV_TYPE_LIQ = 1, 2, 3, 4
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
def backtest_fast(time_arr, z_score, spread_arr, std_arr, bid_1, ask_1, bid_2, ask_2,
            dp_1, dp_2, thresh_low_in, thresh_low_out,
            thresh_high_in, thresh_high_out, long_possible, short_possible,
            dist_in, dist_out, balance, order_size, qty_method, std_1, std_2,
            fee_rate,  sl_std, sl_dist, sl_method, sl_seconds=0,
            close_method=0, leverage=1):

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

            reas = EV_TYPE_CLOSE if reason == REASON_THRESHOLD else EV_TYPE_SL

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
            if dist_in == 0:
                if z < thresh_low_in and long_possible and not sl_block_long:
                    signal = SIG_LONG_OPEN
                elif z > thresh_high_in and short_possible and not sl_block_short:
                    signal = SIG_SHORT_OPEN
            # Обратный способ входа (когда z_score выходит из диапазона входа)
            else:
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

        # --- Обрабатываем открытую позицию ---
        if pos_side == POS_LONG or pos_side == POS_SHORT:
            avg_1 = (bid_1[i] + ask_1[i]) / 2.0
            avg_2 = (bid_2[i] + ask_2[i]) / 2.0
            curr_spr = np.log(avg_1) - np.log(avg_2)
            fixed_z_score = (curr_spr - fixed_mean) / fixed_std

            out_condition = z if close_method == 0 else fixed_z_score

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
                # Прямой способ (когда z_score входит в диапазон выхода)
                if not dist_out:
                    if out_condition < thresh_low_out:
                        signal = SIG_SHORT_CLOSE
                        reason = REASON_THRESHOLD
                # Выходим из сделки, когда z_score покидает диапазон выхода
                else:
                    # Начинаем отслеживать, когда спред опускается ниже порога выхода
                    if not short_out_min_value and out_condition < thresh_low_out:
                        short_out_min_value = out_condition
                    # Если спред обновляет минимум
                    elif short_out_min_value and out_condition < short_out_min_value:
                        short_out_min_value = out_condition
                    # Если спред откатывается на dist_out от минимума
                    elif short_out_min_value and out_condition > short_out_min_value + dist_out and out_condition < thresh_low_out:
                        signal = SIG_SHORT_CLOSE
                        reason = REASON_THRESHOLD
                        short_out_min_value = 0

            elif pos_side == POS_LONG:
                # Прямой способ (когда z_score входит в диапазон выхода)
                if not dist_out:
                    if out_condition > thresh_high_out:
                        signal = SIG_LONG_CLOSE
                        reason = REASON_THRESHOLD
                # Выходим из сделки, когда z_score покидает диапазон выхода
                else:
                    # Начинаем отслеживать, когда спред опускается ниже порога выхода
                    if not long_out_max_value and out_condition > thresh_high_out:
                        long_out_max_value = out_condition
                    # Если спред обновляет максимум
                    elif long_out_max_value and out_condition > long_out_max_value:
                        long_out_max_value = out_condition
                    # Если спред откатывается на dist_out от максимума
                    elif long_out_max_value and out_condition < long_out_max_value - dist_out and out_condition > thresh_high_out:
                        signal = SIG_LONG_CLOSE
                        reason = REASON_THRESHOLD
                        long_out_max_value = 0

    return out, events

def backtest(df, token_1, token_2, dp_1, dp_2, thresh_low_in, thresh_low_out,
            thresh_high_in, thresh_high_out, long_possible, short_possible,
            balance, order_size, qty_method, std_1, std_2,
            fee_rate,  sl_std, sl_dist, sl_method=None, sl_seconds=0,
            close_method='regular', leverage=1, dist_in=0, dist_out=0,
            verbose=False):
    """


    close_method: Как закрывать позицию. regular - по обычному z_score, fix - по фиксированному
    """

    time_arr = df['ts'].to_numpy()
    z = df["z_score"].to_numpy()
    bid_1 = df[f"{token_1}_bid_price"].to_numpy()
    ask_1 = df[f"{token_1}_ask_price"].to_numpy()
    bid_2 = df[f"{token_2}_bid_price"].to_numpy()
    ask_2 = df[f"{token_2}_ask_price"].to_numpy()
    spread_arr = df["spread_mean"].to_numpy()
    std_arr = df["spread_std"].to_numpy()

    spread_arr = spread_arr[~np.isnan(spread_arr)]
    std_arr = std_arr[~np.isnan(std_arr)]

    sl_map = {None: 0, 'counter': 1, 'leave': 2}
    qty_map = {'usdt_neutral': USDT_NEUT, 'vol_neutral': VOL_NEUT}
    close_map = {'regular': 0, 'fix': 1}

    if qty_method == 'vol_neutral' and (std_1 is None or std_2 is None):
        raise Exception('При использовании vol_neutral необходимо задать std_1 и std_2')

    res, events = backtest_fast(time_arr, z, spread_arr, std_arr, bid_1, ask_1, bid_2, ask_2,
            dp_1, dp_2,
            thresh_low_in=thresh_low_in, thresh_high_in=thresh_high_in,
            thresh_low_out=thresh_low_out, thresh_high_out=thresh_high_out,
            long_possible=long_possible, short_possible=short_possible,
            dist_in=dist_in, dist_out=dist_out,
            balance=balance, order_size=order_size, qty_method=qty_map[qty_method],
            std_1=std_1, std_2=std_2, fee_rate=fee_rate,
            sl_std=sl_std, sl_dist=sl_dist, sl_method=sl_map[sl_method], sl_seconds=sl_seconds,
            close_method=close_map[close_method],
            leverage=leverage)

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

    return trades_df
