import polars as pl
import polars_ols as pls
import numpy as np
from numba import njit
from bot.utils.coins import get_step_info
import math
import ast
from datetime import datetime, timedelta
from hurst import compute_Hc
from bot.indicators.rsi import rsi

def round_down(value: float, dp: float):
    return round(math.floor(value / dp) * dp, 6)

def make_trunc_df(df, timeframe, token_1, token_2, start_date=None, end_date=None,
                  method="last", offset='0h', return_bid_ask=False):
    select_spread = True if 'spread' in df.columns else False

    df = df.with_columns(
            ((pl.col(f'{token_1}_bid_price') + pl.col(f'{token_1}_ask_price')) / 2).alias(token_1),
            ((pl.col(f'{token_2}_bid_price') + pl.col(f'{token_2}_ask_price')) / 2).alias(token_2),
        )

    # условия агрегации
    if method == "last":
        agg_exprs = [
            pl.col("ts").first(),
            pl.col(token_1).last().alias(token_1),
            pl.col(token_2).last().alias(token_2),
        ]

        if select_spread:
            agg_exprs.append(pl.col("spread").last().alias("spread"))

    elif method == "triple":
        agg_exprs = [
            pl.col("ts").first(),
            ((pl.col(token_1).last() + pl.col(token_1).max() + pl.col(token_1).min()) / 3).alias(token_1),
            ((pl.col(token_2).last() + pl.col(token_2).max() + pl.col(token_2).min()) / 3).alias(token_2),
        ]

        if select_spread:
            agg_exprs.append(((pl.col("spread").last() + pl.col("spread").max() +
                               pl.col("spread").min()) / 3).alias("spread"))
    else:
        raise ValueError(f"Unknown method: {method}")

    if return_bid_ask:
        agg_exprs.extend((
            pl.col(f'{token_1}_bid_price').last(),
            pl.col(f'{token_1}_ask_price').last(),
            pl.col(f'{token_2}_bid_price').last(),
            pl.col(f'{token_2}_ask_price').last(),
        ))

    df = df.group_by_dynamic(
                index_column="time",
                every=timeframe,
                offset=offset,
                label='left'
            ).agg(agg_exprs)
    if start_date:
        df = df.filter(pl.col('time') > start_date)
    if end_date:
        df = df.filter(pl.col('time') < end_date)

    return df

def make_df_from_orderbooks(df_1, df_2, token_1, token_2, start_time=None, end_time=None):
    """
    Функция на вход принимает 2 датафрейма с ордербуками, усредняет цены покупки/продажи
    и возвращает новый датафрейм с рассчитанным спредом.

    """

    date_col = 'bucket' if 'bucket' in df_1.columns else 'time'

    if start_time is None:
        start_time = df_1[date_col].min()
    if end_time is None:
        end_time = df_1[date_col].max()

    df = df_1.drop('token'
        ).rename(
            {'bid_price': f'{token_1}_bid_price', 'bid_size': f'{token_1}_bid_size',
             'ask_price': f'{token_1}_ask_price', 'ask_size': f'{token_1}_ask_size'}
        ).join(df_2.drop('token'),
            on=date_col, how='full', coalesce=True, suffix='_r'
        ).sort(by=date_col
        ).rename(
            {'bid_price': f'{token_2}_bid_price', 'bid_size': f'{token_2}_bid_size',
             'ask_price': f'{token_2}_ask_price', 'ask_size': f'{token_2}_ask_size'}
        ).fill_null(strategy='forward'
        ).filter(
            (pl.col(date_col) > start_time) & (pl.col(date_col) < end_time)
        ).with_columns(
            pl.col(date_col).dt.epoch('s').alias('ts'),
            ((pl.col(f'{token_1}_bid_price') + pl.col(f'{token_1}_ask_price')) / 2).alias(token_1),
            ((pl.col(f'{token_2}_bid_price') + pl.col(f'{token_2}_ask_price')) / 2).alias(token_2),
            ((pl.col(f'{token_1}_bid_size') + pl.col(f'{token_1}_ask_size')) / 2).alias(token_1 + '_size'),
            ((pl.col(f'{token_2}_bid_size') + pl.col(f'{token_2}_ask_size')) / 2).alias(token_2 + '_size'),
        )

    return df

def make_zscore_df(df, token_1, token_2, wind, method='dist'):
    if method == 'dist':
        return df.lazy().with_columns(
                (pl.col(token_1).log() - pl.col(token_2).log()).alias('spread')
            ).with_columns(
                pl.col('spread').rolling_mean(wind).alias(f'mean'),
                pl.col('spread').rolling_std(wind).alias(f'std')
            ).with_columns(
                ((pl.col('spread') - pl.col('mean')) / pl.col('std')).alias(f'z_score')
            ).collect()
    elif method == 'lr':
        return df.lazy().with_columns(
            pl.col(token_1)
            .least_squares.rolling_ols(pl.col(token_2),
                                    mode='coefficients',
                                    add_intercept=True,
                                    window_size=wind)
            .alias("predictions")
        ).rename({token_2: 'temp'}
        ).unnest('predictions'
        ).rename({token_2: 'beta'}
        ).rename({'temp': token_2}
        ).with_columns(
            (pl.col(token_1) - pl.col('const') - pl.col('beta') * pl.col(token_2)).alias('spread')
        ).with_columns(
            pl.col('spread').rolling_mean(wind).alias('mean'),
            pl.col('spread').rolling_std(wind).alias('std')
        ).with_columns(
            ((pl.col('spread') - pl.col('mean')) / pl.col('std')).alias('z_score')
        ).collect()

@njit(cache=True)
def get_tls_zscore(t1, t2, winds):
    """
    Рассчитывает симметричный Z-score на основе ортогональной регрессии (TLS).
    Порядок подачи t1 и t2 больше не влияет на модуль значения Z-score.
    """
    t1 = np.log(t1)
    t2 = np.log(t2)

    n = t1.shape[0]
    m = winds.shape[0]

    alpha_full = np.full((m, n), np.nan, dtype=np.float64)
    beta_full  = np.full((m, n), np.nan, dtype=np.float64)
    z_full     = np.full((m, n), np.nan, dtype=np.float64)

    spread_full = np.full((m, n), np.nan, dtype=np.float64)
    mean_s_full = np.full((m, n), np.nan, dtype=np.float64)
    std_s_full  = np.full((m, n), np.nan, dtype=np.float64)

    max_w = 0
    for j in range(m):
        wj = winds[j]
        if wj > max_w:
            max_w = wj
    if max_w <= 0:
        return spread_full[:, -1], mean_s_full[:, -1], std_s_full[:, -1], alpha_full[:, -1], beta_full[:, -1], z_full[:, -1]

    spread_bufs = np.zeros((m, max_w), dtype=np.float64)

    sum_x  = np.zeros(m, dtype=np.float64)
    sum_y  = np.zeros(m, dtype=np.float64)
    sum_xx = np.zeros(m, dtype=np.float64)
    sum_yy = np.zeros(m, dtype=np.float64) # <--- Добавлено для TLS
    sum_xy = np.zeros(m, dtype=np.float64)

    sum_s  = np.zeros(m, dtype=np.float64)
    sum_ss = np.zeros(m, dtype=np.float64)

    for i in range(n):
        x = t2[i]
        y = t1[i]

        for j in range(m):
            w = winds[j]
            if w <= 0:
                continue

            # Обновляем суммы
            sum_x[j]  += x
            sum_y[j]  += y
            sum_xx[j] += x * x
            sum_yy[j] += y * y # <--- Добавлено
            sum_xy[j] += x * y

            # Удаляем старые элементы
            if i >= w:
                x_old = t2[i - w]
                y_old = t1[i - w]
                sum_x[j]  -= x_old
                sum_y[j]  -= y_old
                sum_xx[j] -= x_old * x_old
                sum_yy[j] -= y_old * y_old # <--- Добавлено
                sum_xy[j] -= x_old * y_old

            if i >= w - 1:
                mean_x = sum_x[j] / w
                mean_y = sum_y[j] / w

                # Считаем дисперсии и ковариацию
                var_x = (sum_xx[j] / w) - mean_x * mean_x
                var_y = (sum_yy[j] / w) - mean_y * mean_y # <--- Добавлено
                cov_xy = (sum_xy[j] / w) - mean_x * mean_y

                # --- Расчет Beta по методу TLS (Orthogonal Regression) ---
                # Формула: решает квадратное уравнение для минимизации перпендикулярных отрезков

                # Обработка деления на ноль (если cov_xy = 0)
                if abs(cov_xy) < 1e-12:
                     # Если ковариация 0, наклон либо 0 (горизонт), либо inf (вертикаль)
                     # в зависимости от того, чья дисперсия больше
                     beta = 0.0
                else:
                    delta = var_y - var_x
                    # Основная формула TLS
                    beta = (delta + np.sqrt(delta**2 + 4 * cov_xy**2)) / (2 * cov_xy)

                alpha = mean_y - beta * mean_x

                beta_full[j, i] = beta
                alpha_full[j, i] = alpha

                # --- Расчет ортогонального спреда ---
                # Важно: делим на sqrt(1 + beta^2) для получения реального расстояния
                norm_factor = np.sqrt(1.0 + beta * beta)
                s = (y - (alpha + beta * x)) / norm_factor

                # Дальше логика Rolling Z-score не меняется, так как мы уже получили "правильный" s
                pos = i % w
                if i >= w:
                    s_old = spread_bufs[j, pos]
                    sum_s[j]  -= s_old
                    sum_ss[j] -= s_old * s_old

                spread_bufs[j, pos] = s
                sum_s[j]  += s
                sum_ss[j] += s * s

                mean_s = sum_s[j] / w
                num = sum_ss[j] - w * mean_s * mean_s
                denom = w - 1

                eps = 1e-12 * (1.0 + abs(sum_ss[j]))
                if num < 0.0 and num > -eps:
                    num = 0.0

                if num >= 0.0 and np.isfinite(num) and denom > 0:
                    var_s_sample = num / denom
                    if var_s_sample <= 0.0:
                        std_s = 0.0
                    else:
                        std_s = np.sqrt(var_s_sample)

                    if std_s > 0.0:
                        z = (s - mean_s) / std_s
                    else:
                        z = 0.0

                    mean_s_full[j, i] = mean_s
                    std_s_full[j, i]  = std_s
                    z_full[j, i]      = z
                else:
                    mean_s_full[j, i] = mean_s
                    std_s_full[j, i]  = np.nan
                    z_full[j, i]      = np.nan

                spread_full[j, i] = s
            else:
                pass

    return spread_full[:, -1], mean_s_full[:, -1], std_s_full[:, -1], alpha_full[:, -1], beta_full[:, -1], z_full[:, -1]

@njit(cache=True)
def get_lr_zscore(t1, t2, winds):
    t1 = np.log(t1)
    t2 = np.log(t2)

    n = t1.shape[0]
    m = winds.shape[0]

    alpha_full = np.full((m, n), np.nan, dtype=np.float64)
    beta_full  = np.full((m, n), np.nan, dtype=np.float64)
    z_full     = np.full((m, n), np.nan, dtype=np.float64)

    spread_full = np.full((m, n), np.nan, dtype=np.float64)
    mean_s_full = np.full((m, n), np.nan, dtype=np.float64)
    std_s_full  = np.full((m, n), np.nan, dtype=np.float64)

    # максимальный размер окна
    max_w = 0
    for j in range(m):
        wj = winds[j]
        if wj > max_w:
            max_w = wj
    if max_w <= 0:
        return spread_full[:, -1], mean_s_full[:, -1], std_s_full[:, -1], alpha_full[:, -1], beta_full[:, -1], z_full[:, -1]

    spread_bufs = np.zeros((m, max_w), dtype=np.float64)

    sum_x  = np.zeros(m, dtype=np.float64)
    sum_y  = np.zeros(m, dtype=np.float64)
    sum_xx = np.zeros(m, dtype=np.float64)
    sum_xy = np.zeros(m, dtype=np.float64)

    sum_s  = np.zeros(m, dtype=np.float64)
    sum_ss = np.zeros(m, dtype=np.float64)

    for i in range(n):
        x = t2[i]
        y = t1[i]

        for j in range(m):
            w = winds[j]
            if w <= 0:
                continue

            # обновляем суммы для регрессии
            sum_x[j] += x
            sum_y[j] += y
            sum_xx[j] += x * x
            sum_xy[j] += x * y

            # удаляем старые элементы если окно скользнуло
            if i >= w:
                x_old = t2[i - w]
                y_old = t1[i - w]
                sum_x[j]  -= x_old
                sum_y[j]  -= y_old
                sum_xx[j] -= x_old * x_old
                sum_xy[j] -= x_old * y_old

            if i >= w - 1:
                mean_x = sum_x[j] / w
                mean_y = sum_y[j] / w

                var_x = (sum_xx[j] / w) - mean_x * mean_x
                cov_xy = (sum_xy[j] / w) - mean_x * mean_y

                if var_x <= 0.0 or not np.isfinite(var_x):
                    beta = np.nan
                    alpha = np.nan
                else:
                    beta = cov_xy / var_x
                    alpha = mean_y - beta * mean_x

                beta_full[j, i] = beta
                alpha_full[j, i] = alpha

                s = y - (alpha + beta * x)

                pos = i % w
                if i >= w:
                    s_old = spread_bufs[j, pos]
                    sum_s[j]  -= s_old
                    sum_ss[j] -= s_old * s_old

                spread_bufs[j, pos] = s
                sum_s[j]  += s
                sum_ss[j] += s * s

                mean_s = sum_s[j] / w
                num = sum_ss[j] - w * mean_s * mean_s
                denom = w - 1

                # устойчивая обработка малых отрицательных num
                eps = 1e-12 * (1.0 + abs(sum_ss[j]))
                if num < 0.0 and num > -eps:
                    num = 0.0

                if num >= 0.0 and np.isfinite(num) and denom > 0:
                    var_s_sample = num / denom
                    if var_s_sample <= 0.0:
                        std_s = 0.0
                    else:
                        std_s = np.sqrt(var_s_sample)
                    if std_s > 0.0:
                        z = (s - mean_s) / std_s
                    else:
                        z = 0.0

                    mean_s_full[j, i] = mean_s
                    std_s_full[j, i]  = std_s
                    z_full[j, i]      = z
                else:
                    mean_s_full[j, i] = mean_s
                    std_s_full[j, i]  = np.nan
                    z_full[j, i]      = np.nan

                spread_full[j, i] = s
            else:
                # окно ещё не заполнено => оставляем nan
                pass

    return spread_full[:, -1], mean_s_full[:, -1], std_s_full[:, -1], alpha_full[:, -1], beta_full[:, -1], z_full[:, -1]

@njit(cache=True)
def get_dist_zscore(t1: np.ndarray, t2: np.ndarray, winds: np.ndarray):
    """
    spread: 1D float64 array (n,)
    winds: 1D int64 array (m,)
    returns: means (m,n), stds (m,n), zs (m,n)
    """
    spread = np.log(t1) - np.log(t2)

    n = spread.shape[0]
    m = winds.shape[0]

    # Prepare outputs
    spr = np.full((m, n), np.nan, dtype=np.float64)
    means = np.full((m, n), np.nan, dtype=np.float64)
    stds = np.full((m, n), np.nan, dtype=np.float64)
    zs = np.full((m, n), np.nan, dtype=np.float64)

    if n == 0 or m == 0:
        return spr[:, -1], means[:, -1], stds[:, -1], zs[:, -1]

    # Вычисляем кумулятивные суммы и квадраты (внутри компилированной функции)
    cumulative = np.empty(n, dtype=np.float64)
    cumulative_sq = np.empty(n, dtype=np.float64)
    s = 0.0
    sq = 0.0
    for i in range(n):
        s += spread[i]
        sq += spread[i] * spread[i]
        cumulative[i] = s
        cumulative_sq[i] = sq

    # Распараллеливание по окнам (каждое окно — независимая задача)
    for wi in range(m):
        wind = winds[wi]
        if wind <= 1 or n < wind:
            # оставляем строки заполненными NaN
            continue

        # для каждого окна двигаемся по позициям
        for i in range(wind - 1, n):
            start_index = i - wind + 1
            if start_index == 0:
                s_window = cumulative[i]
                sq_window = cumulative_sq[i]
            else:
                s_window = cumulative[i] - cumulative[start_index - 1]
                sq_window = cumulative_sq[i] - cumulative_sq[start_index - 1]

            mean_val = s_window / wind
            # ddof=1
            variance = (sq_window - wind * mean_val * mean_val) / (wind - 1)
            if variance < 0.0:
                # численные погрешности
                variance = 0.0
            std_val = np.sqrt(variance)

            means[wi, i] = mean_val
            spr[wi, i] = spread[i]
            stds[wi, i] = std_val
            if std_val != 0.0:
                zs[wi, i] = (spread[i] - mean_val) / std_val
            else:
                zs[wi, i] = np.nan

    return spr[:, -1], means[:, -1], stds[:, -1], zs[:, -1]

@njit(cache=True)
def calculate_z_score(start_ts: int,
                        median_length: int,
                        tss: np.ndarray,
                        price1: np.ndarray,
                        price2: np.ndarray,
                        hist_ts: np.ndarray,
                        hist_t1: np.ndarray,
                        hist_t2: np.ndarray,
                        winds: np.ndarray,
                        spr_method: int):
    """
    Функция на исторических данных рассчитывает z_score для каждой строки.
    Последний элемент в агрегированном датафрейме отбрасывается, так как
        уже содержит информацию о текущей цене.

    Args:
        start_ts: unix timestamp начала бектеста
        median_length: количество секунд для вычисления медианы
        tss: массив, состоящий из unix timestamp, для каждой строки секундного датафрейма
        price1: массив из цен для токена_1
        price2: массив из цен для токена_2
        size1: массив из доступных объёмов для токена_1
        size2: массив из доступных объёмов для токена_2
        hist_ts: массив из unix timestamp для каждой строки датафрейма с агрегированными данными
        hist_t1: массив из цен для токена_1 из датафрейма с агрегированными данными
        hist_t2: массив из цен для токена_2 из датафрейма с агрегированными данными
        winds: массив из размеров окон
        min_order: размер минимального ордера в usdt для фильтрации слишком малого объёма

    """
    nrows = tss.shape[0]
    n_winds = winds.shape[0]
    ts_arr = np.full(nrows, np.nan)
    spread_arr = np.full((nrows, n_winds), np.nan)
    spr_mean_arr = np.full((nrows, n_winds), np.nan)
    spr_std_arr = np.full((nrows, n_winds), np.nan)
    z_score_arr = np.full((nrows, n_winds), np.nan)

    for i in range(nrows):
        # Пропускаем начало датафрейма, нужное для вычисления медианы
        if tss[i] <= start_ts:
            continue

        t1_price = price1[i - median_length: i]
        t2_price = price2[i - median_length: i]
        t1_med = np.median(t1_price)
        t2_med = np.median(t2_price)

        # Выберем из агрегированных цен только те, которые были до текущего момента
        mask = hist_ts < tss[i]
        t1_hist = hist_t1[mask]
        t2_hist = hist_t2[mask]

        # Сформируем массивы, в которых к историческим данным в конец добавим текущую медианную цену, и посчитаем z_score
        t1_arr = np.append(t1_hist, t1_med)
        t2_arr = np.append(t2_hist, t2_med)

        if spr_method == 0:
            spread, mean_spread, spread_std, alpha, beta, zscore = get_lr_zscore(t1_arr, t2_arr, winds)
        elif spr_method == 1:
            spread, mean_spread, spread_std, zscore = get_dist_zscore(t1_arr, t2_arr, winds)
        elif spr_method == 2:
            spread, mean_spread, spread_std, alpha, beta, zscore = get_tls_zscore(t1_arr, t2_arr, winds)

        ts_arr[i] = tss[i]
        spread_arr[i] = spread
        spr_mean_arr[i] = mean_spread
        spr_std_arr[i] = spread_std
        z_score_arr[i] = zscore

    return ts_arr, spread_arr, spr_mean_arr, spr_std_arr, z_score_arr

def create_zscore_df(token_1, token_2, sec_df, agg_df, tf, winds, start_ts, median_length, spr_method):
    if spr_method == 'lr':
        spr_method = 0
    elif spr_method == 'dist':
        spr_method = 1
    elif spr_method == 'tls':
        spr_method = 2

    # --- Перевод polars в numpy ---
    tss = sec_df['ts'].to_numpy()
    price1 = sec_df[token_1].to_numpy()
    price2 = sec_df[token_2].to_numpy()

    hist_ts = agg_df['ts'].to_numpy()
    hist_t1 = agg_df[token_1].to_numpy()
    hist_t2 = agg_df[token_2].to_numpy()

    # --- Вычисляем z_score ---
    ts_arr, spr_arr, spr_mean_arr, spr_std_arr, z_score_arr = calculate_z_score(
        start_ts, median_length, tss, price1, price2,
                        hist_ts, hist_t1, hist_t2, winds, spr_method)

    # --- Собираем итоговый polars DataFrame из буферов (только заполненные строки) ---
    base_df = pl.DataFrame({'ts': tss})
    for i, wind in enumerate(winds):
        tdf = pl.DataFrame({'ts': tss,
                        f'spread_{wind}_{tf}': spr_arr[:, i],
                        f'spread_mean_{wind}_{tf}': spr_mean_arr[:, i],
                        f'spread_std_{wind}_{tf}': spr_std_arr[:, i],
                        f'z_score_{wind}_{tf}': z_score_arr[:, i]})
        base_df = base_df.join(tdf, on='ts')

    return sec_df.select('time', 'ts', token_1, token_2, f'{token_1}_size',
                         f'{token_2}_size', f'{token_1}_bid_price',
                         f'{token_1}_ask_price', f'{token_1}_bid_size',
                         f'{token_1}_ask_size', f'{token_2}_bid_price',
                         f'{token_2}_ask_price', f'{token_2}_bid_size',
                         f'{token_2}_ask_size').join(base_df, on='ts').drop_nans()

def get_qty(
        token_1: str,
        token_2: str,
        price_1: float,
        price_2: float,
        beta: float,
        coin_information: dict,
        total_usdt_amount: float = 100.0,
        fee_rate: float = 0.001,
        std_1: float | None = None,
        std_2: float | None = None,
        method: 'str' = 'beta',
    ):
    """
        Вычисляет размеры позиций для двух активов.
        Args:
            price_1: цена актива 1 (в долларах)
            price_2: цена актива 2 (в долларах)
            beta: хедж-коэффициент (количество B на 1 A)
            coin_information: словарь с технической информацией по монетам
            total_usdt_amount: общий размер позиции в долларах
            fee_rate: комиссия за сделки
            std_1, std_2 - волатильность активов 1 и 2 для расчёта vol-neutral позиции
            method: метод распределения денег между плечами сделки. 'beta' или 'usdt_neutral'
        Returns:
            qty_1, qty_2: количество токенов 1 и 2
    """

    if not token_1.endswith('_USDT'):
        token_1 += '_USDT'
    if not token_2.endswith('_USDT'):
        token_2 += '_USDT'

    dp_1 = get_step_info(coin_information, token_1, 'bybit_linear', 'bybit_linear')
    dp_2 = get_step_info(coin_information, token_2, 'bybit_linear', 'bybit_linear')

    if method == 'beta_neutral':
        qty_1 = total_usdt_amount * (1 - fee_rate) / (price_1 + beta * price_2)
        qty_1 = round_down(qty_1, dp_1)
        qty_2 = beta * qty_1
        qty_2 = round_down(qty_2, dp_2)
    elif method == 'usdt_neutral':
        qty_1 = round_down(0.5 * total_usdt_amount / (1.0 + 2.0 * fee_rate) / price_1, dp_1)
        qty_2 = round_down(0.5 * total_usdt_amount / (1.0 + 2.0 * fee_rate) / price_2, dp_2)
    elif method == 'vol_neutral':
        c_eff = total_usdt_amount / (1.0 + 4.0 * fee_rate)
        d1 = c_eff * std_2 / (std_1 + std_2)
        d2 = c_eff * std_1 / (std_1 + std_2)

        qty_1 = round_down(d1 / price_1, dp_1)
        qty_2 = round_down(d2 / price_2, dp_2)

    return qty_1, qty_2

def calculate_profit_curve(df, token_1, token_2, side, t1_op, t2_op, t1_qty, t2_qty, fee_rate):
    if side == 'long':
        tdf = df.select('time', f'{token_1}_bid_price', f'{token_1}_bid_size',
                        f'{token_2}_ask_price', f'{token_2}_ask_size', 'z_score').rename(
            {
                f'{token_1}_bid_price': f'{token_1}_price', f'{token_1}_bid_size': f'{token_1}_size',
                f'{token_2}_ask_price': f'{token_2}_price', f'{token_2}_ask_size': f'{token_2}_size',
            }
        )

        expr_t1_long = (
            pl.lit(t1_qty)
            * (pl.col(f"{token_1}_price") - pl.lit(t1_op) - pl.lit(fee_rate) * (pl.lit(t1_op) + pl.col(f"{token_1}_price")))
        )
        expr_t2_short = (
            pl.lit(t2_qty)
            * (pl.lit(t2_op) - pl.col(f"{token_2}_price") - pl.lit(fee_rate) * (pl.lit(t2_op) + pl.col(f"{token_2}_price")))
        )

        tdf = tdf.with_columns((expr_t1_long + expr_t2_short).alias("profit"))


    elif side == 'short':
        tdf = df.select('time', f'{token_1}_ask_price', f'{token_1}_ask_size',
                        f'{token_2}_bid_price', f'{token_2}_bid_size', 'z_score').rename(
            {
                f'{token_1}_ask_price': f'{token_1}_price', f'{token_1}_ask_size': f'{token_1}_size',
                f'{token_2}_bid_price': f'{token_2}_price', f'{token_2}_bid_size': f'{token_2}_size',
            }
        )

        expr_t1_short = (
            pl.lit(t1_qty)
            * (pl.lit(t1_op) - pl.col(f"{token_1}_price") - pl.lit(fee_rate) * (pl.lit(t1_op) + pl.col(f"{token_1}_price")))
        )

        expr_t2_long = (
            pl.lit(t2_qty)
            * (pl.col(f"{token_2}_price") - pl.lit(t2_op) - pl.lit(fee_rate) * (pl.lit(t2_op) + pl.col(f"{token_2}_price")))
        )

        tdf = tdf.with_columns((expr_t1_short + expr_t2_long).alias("profit"))

    return tdf

def get_thresholds():
    data = []
    with open('./bot/config/thresholds.txt', 'r') as file:
        for line in file:
            line = line.strip()  # Удаляем пробелы и переносы строк
            if line:  # Игнорируем пустые строки
                # Преобразуем строку в кортеж с помощью literal_eval
                tuple_data = ast.literal_eval(line)
                data.append(tuple_data)
    return data

def check_pos(name, pairs):
    token_1, token_2, *_ = name.split('_')
    return any(a == token_1 and b == token_2 for a, b, _ in pairs)

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

@njit(cache=True)
def calculate_half_life(spread):
    """
    Рассчитывает период полураспада (Half-Life) спреда через процесс Орнштейна-Уленбека.
    """
    n = len(spread)

    # Создаем лагированный спред и разности
    spread_lag = np.empty_like(spread)
    spread_lag[0] = 0
    spread_lag[1:] = spread[:-1]

    spread_ret = np.empty_like(spread)
    spread_ret[0] = 0
    spread_ret[1:] = spread[1:] - spread[:-1]

    y = spread_ret[1:]  # зависимая переменная
    x = spread_lag[1:]  # независимая переменная
    n_valid = len(y)

    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.sum(x**2)
    sum_xy = np.dot(x, y)

    denominator = n_valid * sum_x2 - sum_x**2
    if denominator == 0:
        return np.inf

    lambda_param = (n_valid * sum_xy - sum_x * sum_y) / denominator

    if lambda_param >= 0:
        return np.inf

    half_life = -np.log(2) / lambda_param
    return half_life

def lr_coefs(y):
    n = len(y)
    x = np.arange(n)

    sum_x = n * (n - 1) / 2
    sum_x2 = (n - 1) * n * (2 * n - 1) / 6
    sum_y = np.sum(y)
    sum_xy = np.dot(x, y)

    k = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    b = (sum_y - k * sum_x) / n
    return k, b

def load_data(token_1, token_2, valid_time, end_time, tf, wind, db_manager):
        """
        Args:
            token_1, token_2 - Названия токенов
            valid_time - время в формате datetime.datetime, с которого начинается непосредственно бектест
            end_time   - время окончания бектеста в формате datetime.datetime
            params - dict() с параметрами
        """
        train_length = int(tf[0]) * wind * 2 + 1
        start_time = valid_time - timedelta(hours=train_length)

        df_1 = db_manager.get_tick_ob(token=token_1 + '_USDT',
                                         start_time=start_time,
                                         end_time=end_time)
        df_2 = db_manager.get_tick_ob(token=token_2 + '_USDT',
                                         start_time=start_time,
                                         end_time=end_time)
        tick_df = make_df_from_orderbooks(df_1, df_2, token_1, token_2, start_time=start_time)
        agg_df = make_trunc_df(tick_df, timeframe=tf, token_1=token_1, token_2=token_2, method='triple')

        return tick_df, agg_df

def get_dpr_dz(profit_arr, z_score_arr):
    mask = (~np.isnan(profit_arr)) & (~np.isnan(z_score_arr))

    dz = np.diff(z_score_arr[mask])
    dpr = np.diff(profit_arr[mask])

    eps = 0.000001
    mask = (np.abs(dz) >= eps) & (np.abs(dpr) >= eps)
    mask_pos = (np.abs(dz) >= eps) & (np.abs(dpr) >= eps) & (dz > 0)
    mask_neg = (np.abs(dz) >= eps) & (np.abs(dpr) >= eps) & (dz < 0)

    dpr_dz = dpr[mask] / dz[mask]
    dpr_dz_pos = dpr[mask_pos] / dz[mask_pos]
    dpr_dz_neg = dpr[mask_neg] / dz[mask_neg]

    return dpr_dz.mean(), dpr_dz_pos.mean(), dpr_dz_neg.mean()

def calculate_tls_beta(price_x, price_y):
    """
    Считает Beta методом Total Least Squares (ODR).
    Подходит для Дистанционного метода и пар с шумом в обеих ногах.
    price_x: массив цен токена 2 (независимая переменная, 'IMX')
    price_y: массив цен токена 1 (зависимая переменная, 'DRIFT')
    """
    # Переходим к логарифмам
    x = np.log(price_x)
    y = np.log(price_y)

    # Центрируем данные (убираем среднее, чтобы найти наклон)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_centered = x - x_mean
    y_centered = y - y_mean

    # Формируем матрицу данных [N, 2]
    data = np.vstack([x_centered, y_centered]).T

    # Сингулярное разложение (SVD) - это сердце TLS
    # Мы ищем направление наименьшей дисперсии (перпендикуляр к линии тренда)
    _, _, vh = np.linalg.svd(data)

    # Последняя строка Vh - это вектор нормали к линии (a*x + b*y = c)
    # Нам нужен наклон beta = -a/b
    # vh имеет вид [[v_xx, v_xy], [v_yx, v_yy]]
    # Наш вектор нормали - это последняя строка (индекс -1)

    normal_vector = vh[-1]

    # Коэффициенты a и b уравнения ax + by = 0
    a = normal_vector[0]
    b = normal_vector[1]

    # Beta = -a / b
    beta_tls = -a / b

    return beta_tls

def get_profit_arr(tick_df, agg_df, token_1, token_2, side_1, side_2, wind, coin_information):
    """
        В tick_df передаётся тиковый датафрейм, отсечённый по моменту входа в позицию.
        agg_df - также отсечённый df.
    """
    open_time = tick_df['time'][-1]
    time_12h = open_time - timedelta(hours=12)
    time_wind = open_time - timedelta(hours=wind)
    ts_12h = int(datetime.timestamp(time_12h))
    ts_wind = int(datetime.timestamp(time_wind))

    open_row_12 = tick_df.filter(pl.col('ts') == ts_12h)
    open_row_wind = tick_df.filter(pl.col('ts') == ts_wind)

    t1_op_12 = open_row_12[token_1][0]
    t2_op_12 = open_row_12[token_2][0]
    t1_op_w = open_row_wind[token_1][0]
    t2_op_w = open_row_wind[token_2][0]

    q1_12, q2_12 = get_qty(token_1, token_2, t1_op_12, t2_op_12, None, coin_information, 100, method='usdt_neutral')
    q1_w, q2_w = get_qty(token_1, token_2, t1_op_w, t2_op_w, None, coin_information, 100, method='usdt_neutral')

    tss = tick_df['ts'].to_numpy()
    price1 = tick_df[token_1].to_numpy()
    price2 = tick_df[token_2].to_numpy()

    hist_ts = agg_df['ts'].to_numpy()
    hist_t1 = agg_df[token_1].to_numpy()
    hist_t2 = agg_df[token_2].to_numpy()

    nrows = tss.shape[0]
    t_arr = np.full(nrows, np.nan)
    ts_arr = np.full(nrows, np.nan)
    z_score_arr = np.full(nrows, np.nan)
    profit_12_arr = np.full(nrows, np.nan)
    profit_w_arr = np.full(nrows, np.nan)

    for i in range(nrows):
        t1_price = price1[i]
        t2_price = price2[i]

        # Выберем из агрегированных цен только те, которые были до текущего момента
        mask = hist_ts < tss[i]
        t1_hist = hist_t1[mask][-wind:]
        t2_hist = hist_t2[mask][-wind:]

        # Сформируем массивы, в которых к историческим данным в конец добавим текущую медианную цену, и посчитаем z_score
        t1_arr = np.append(t1_hist, t1_price)
        t2_arr = np.append(t2_hist, t2_price)

        _, _, _, zscore = get_dist_zscore(t1_arr, t2_arr, np.array([wind]))
        curr_profit_1w = calculate_profit(t1_op_w, t1_price, q1_w, side_1)
        curr_profit_2w = calculate_profit(t2_op_w, t2_price, q2_w, side_2)
        curr_profit_w = curr_profit_1w + curr_profit_2w

        if tss[i] > ts_12h:
            curr_profit_1_12 = calculate_profit(t1_op_12, t1_price, q1_12, side_1)
            curr_profit_2_12 = calculate_profit(t2_op_12, t2_price, q2_12, side_2)
            curr_profit_12 = curr_profit_1_12 + curr_profit_2_12
            profit_12_arr[i] = curr_profit_12

        ts_arr[i] = tss[i]
        z_score_arr[i] = zscore[0]
        profit_w_arr[i] = curr_profit_w

    return ts_arr, z_score_arr, profit_w_arr, profit_12_arr

def get_open_time_stats(token_1, token_2, open_time, tick_df, agg_df, lr_spread, params, coin_information):
    tf = params['tf']
    wind = params['wind']
    side_1 = params['side_1']
    side_2 = params['side_2']

    stats = pl.read_parquet('./data/pair_selection/all_pairs.parquet').filter(
                (pl.col('coin1') == token_1) & (pl.col('coin2') == token_2)
            )

    tick_hist = tick_df.filter(pl.col('time') < open_time).tail(wind * 12 * 60)
    agg_hist = agg_df.filter(pl.col('time') < open_time).tail(wind)

    agg_df_wind = agg_hist.with_columns([
        pl.col(token_1).pct_change().std().alias("vol_1"),
        pl.col(token_2).pct_change().std().alias("vol_2"),
        pl.col('log_spread').mean().alias('mean')
    ])

    agg_df_12 = agg_hist.tail(12).with_columns([
        pl.col(token_1).pct_change().std().alias("vol_1"),
        pl.col(token_2).pct_change().std().alias("vol_2"),
        pl.col('log_spread').mean().alias('mean')
    ])

    std_1 = stats['std_1'][0]
    std_2 = stats['std_2'][0]
    beta_1 = stats['beta_1'][0]
    beta_2 = stats['beta_2'][0]
    pv_1 = stats['pv_1'][0]
    pv_2 = stats['pv_2'][0]
    tls_beta = stats['tls_beta'][0]
    coint = stats['coint'][0]
    hedge_r = stats['hedge_r'][0]

    mean_wind = agg_df_wind["mean"][0]
    std_1_wind = agg_df_wind["vol_1"][0]
    std_2_wind = agg_df_wind["vol_2"][0]
    mean_12 = agg_df_12["mean"][0]
    std_1_12 = agg_df_12["vol_1"][0]
    std_2_12 = agg_df_12["vol_2"][0]
    mean_diff = (mean_12 / mean_wind - 1) * 100

    tls_beta_wind = calculate_tls_beta(agg_df_wind[token_1].to_numpy(), agg_df_wind[token_2].to_numpy())
    tls_beta_12 = calculate_tls_beta(agg_df_12[token_1].to_numpy(), agg_df_12[token_2].to_numpy())

    spread_20m = make_trunc_df(tick_hist, '20m', token_1, token_2, method="triple").with_columns(
            (pl.col(token_1).log() - pl.col(token_2).log()).alias('log_spread')
        ).filter(pl.col('time') < open_time).tail(3*wind)
    H_wind, _, _ = compute_Hc(spread_20m['log_spread'])

    spread_5m = make_trunc_df(tick_hist, '5m', token_1, token_2, method="triple").with_columns(
            (pl.col(token_1).log() - pl.col(token_2).log()).alias('log_spread')
        ).filter(pl.col('time') < open_time).tail(12 * 12)
    H_12, _, _ = compute_Hc(spread_5m['log_spread'])

    open_time_tick_df = tick_df.select('time', 'ts', token_1, token_2, 'log_spread').filter(pl.col('time') < open_time).tail(1)
    open_time_1h_df = agg_hist.vstack(open_time_tick_df)
    spread_rsi_1h = rsi(open_time_1h_df, window=14, col_name='log_spread')['rsi'][-1]
    spread_rsi_5m = rsi(spread_5m, window=24, col_name='log_spread')['rsi'][-1]

    rsi_t1_5m = rsi(spread_5m, window=24, col_name=token_1)['rsi'][-1]
    rsi_t2_5m = rsi(spread_5m, window=24, col_name=token_2)['rsi'][-1]
    rsi_t1_1h = rsi(open_time_1h_df, window=14, col_name=token_1)['rsi'][-1]
    rsi_t2_1h = rsi(open_time_1h_df, window=14, col_name=token_2)['rsi'][-1]

    ts_arr, z_score_arr, profit_w_arr, profit_12_arr = get_profit_arr(tick_hist,
                agg_hist, token_1, token_2, side_1, side_2, wind, coin_information)
    rel_mean_12, rel_pos_12, rel_neg_12 = get_dpr_dz(profit_12_arr, z_score_arr)
    rel_mean_w, rel_pos_w, rel_neg_w = get_dpr_dz(profit_w_arr, z_score_arr)

    tick_spread_wind = tick_hist['log_spread'].tail(wind * 12 * 60).to_numpy()
    tick_spread_12 = tick_hist['log_spread'].tail(12 * 12 * 60).to_numpy()
    k_wind, b_wind = lr_coefs(tick_spread_wind)
    k_12, b_12 = lr_coefs(tick_spread_12)
    y_end_wind = k_wind * (len(tick_spread_wind) - 1) + b_wind
    y_end_12 = k_12 * (len(tick_spread_12) - 1) + b_12
    trend_wind = (y_end_wind - b_wind) / b_wind * 100
    trend_12 = (y_end_12 - b_12) / b_12 * 100

    half_life_log_spread = calculate_half_life(tick_df['log_spread'].to_numpy()) / 12 / 60
    half_life_lr_spread = calculate_half_life(lr_spread) / 12 / 60

    return {
        'std_1_180d': std_1,
        'std_2_180d': std_2,
        'beta_1_180d': beta_1,
        'beta_2_180d': beta_2,
        'pv_1': pv_1,
        'pv_2': pv_2,
        'tls_beta_180d': tls_beta,
        'coint_180d': coint,
        'johansen_beta_180d': hedge_r,
        'mean_wind': mean_wind,          # Среднее значение спреда за последние wind часов
        'std_1_wind': std_1_wind,        # Стандартное отклонение спреда за последние wind часов
        'std_2_wind': std_2_wind,
        'mean_12h': mean_12,              # Среднее значение спреда за последние 12 часов
        'std_1_12h': std_1_12,
        'std_2_12h': std_2_12,
        'mean_diff': mean_diff,          # Изменение среднего значения между 12 часами и wind часами в %
        'tls_beta_wind': tls_beta_wind,
        'tls_beta_12h': tls_beta_12,
        'hurst_wind': H_wind,            # Показатель Хёрста за wind часов
        'hurst_12h': H_12,                # Показатель Хёрста за 12 часов
        'spread_rsi_5m': spread_rsi_5m,  # Индикатор RSI на агрегированном по 5 мин спреде
        'spread_rsi_1h': spread_rsi_1h,
        'rsi_t1_5m': rsi_t1_5m,          # Индикатор RSI на агрегированном по 5 мин цене токена_1
        'rsi_t2_5m': rsi_t2_5m,
        'rsi_t1_1h': rsi_t1_1h,          # Индикатор RSI на агрегированном по 1 часу цене токена_1
        'rsi_t2_1h': rsi_t2_1h,
        'profit_sensitivity_12h_mean': rel_mean_12, # Изменение профита на единицу изменения z_score за последние 12 часов (среднее)
        'profit_sensitivity_12h_pos': rel_pos_12,   # Только положительные изменения z_score
        'profit_sensitivity_12h_neg': rel_neg_12,   # Только отрицательные изменения z_score
        'profit_sensitivity_wind_mean': rel_mean_w,
        'profit_sensitivity_wind_pos': rel_pos_w,
        'profit_sensitivity_wind_neg': rel_neg_w,
        'trend_wind': trend_wind,         # Отношение конца регрессионной прямой к началу за wind часов
        'trend_12h': trend_12,
        'half_life_log_spread': half_life_log_spread,
        'half_life_lr_spread': half_life_lr_spread

    }
