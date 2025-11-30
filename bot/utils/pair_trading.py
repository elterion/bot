import polars as pl
import polars_ols as pls
import numpy as np
from numba import njit
from bot.utils.coins import get_step_info
import math
import ast
from datetime import datetime, timedelta

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

def get_zscore(df, token_1, token_2, winds, method):
    t1 = df[token_1].to_numpy()
    t2 = df[token_2].to_numpy()
    winds_np = np.array(winds)

    if method == 'lr':
        alpha, beta, zscore = get_lr_zscore(t1, t2, winds_np)
        return alpha, beta, zscore
    elif method == 'dist':
        means, stds, z_scores = get_dist_zscore(t1, t2, winds_np)
        return means, stds, z_scores
    elif method == 'tls':
        alpha, beta, zscore = get_lr_zscore(t1, t2, winds_np)
        return alpha, beta, zscore

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
    means = np.full((m, n), np.nan, dtype=np.float64)
    stds = np.full((m, n), np.nan, dtype=np.float64)
    zs = np.full((m, n), np.nan, dtype=np.float64)

    if n == 0 or m == 0:
        return means[:, -1], stds[:, -1], zs[:, -1]

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
            stds[wi, i] = std_val
            if std_val != 0.0:
                zs[wi, i] = (spread[i] - mean_val) / std_val
            else:
                zs[wi, i] = np.nan

    return means[:, -1], stds[:, -1], zs[:, -1]

@njit(cache=True)
def calculate_z_score(start_ts: int,
                        median_length: int,
                        tss: np.ndarray,
                        price1: np.ndarray,
                        price2: np.ndarray,
                        size1: np.ndarray,
                        size2: np.ndarray,
                        hist_ts: np.ndarray,
                        hist_t1: np.ndarray,
                        hist_t2: np.ndarray,
                        winds: np.ndarray,
                        min_order: float,
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
        t1_arr_med = np.append(t1_hist, t1_med)
        t2_arr_med = np.append(t2_hist, t2_med)

        if spr_method == 0:
            spread, mean_spread, spread_std, alpha, beta, zscore = get_lr_zscore(t1_arr_med, t2_arr_med, winds)
        elif spr_method == 1:
            mean_spread, spread_std, zscore = get_dist_zscore(t1_arr_med, t2_arr_med, winds)
        elif spr_method == 2:
            spread, mean_spread, spread_std, alpha, beta, zscore = get_tls_zscore(t1_arr_med, t2_arr_med, winds)

        ts_arr[i] = tss[i]
        spr_mean_arr[i] = mean_spread
        spr_std_arr[i] = spread_std
        z_score_arr[i] = zscore

    return ts_arr, spr_mean_arr, spr_std_arr, z_score_arr

def create_zscore_df(token_1, token_2, sec_df, agg_df, tf, winds, min_order, start_ts, median_length, spr_method):
    if spr_method == 'lr':
        spr_method = 0
    elif spr_method == 'dist':
        spr_method = 1
    elif spr_method == 'tls':
        spr_method = 2

    # --- Перевод polars в numpy ---
    tss = sec_df['ts'].to_numpy()
    size1 = sec_df[f'{token_1}_size'].to_numpy()   # np.ndarray, shape (n,)
    price1 = sec_df[token_1].to_numpy()
    size2 = sec_df[f'{token_2}_size'].to_numpy()
    price2 = sec_df[token_2].to_numpy()

    hist_ts = agg_df['ts'].to_numpy()
    hist_t1 = agg_df[token_1].to_numpy()
    hist_t2 = agg_df[token_2].to_numpy()

    # --- Вычисляем z_score ---
    ts_arr, spr_mean_arr, spr_std_arr, z_score_arr = calculate_z_score(
        start_ts, median_length, tss, price1, price2, size1, size2,
                                 hist_ts, hist_t1, hist_t2, winds, min_order, spr_method)

    # --- Собираем итоговый polars DataFrame из буферов (только заполненные строки) ---
    base_df = pl.DataFrame({'ts': tss})
    for i, wind in enumerate(winds):
        tdf = pl.DataFrame({'ts': tss, f'spread_mean_{wind}_{tf}': spr_mean_arr[:, i],
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
        start_time = valid_time - timedelta(days=train_length)
        start_ts   = int(datetime.timestamp(valid_time))

        df_1 = db_manager.get_tick_ob(token=token_1 + '_USDT',
                                         start_time=start_time,
                                         end_time=end_time)
        df_2 = db_manager.get_tick_ob(token=token_2 + '_USDT',
                                         start_time=start_time,
                                         end_time=end_time)
        df = make_df_from_orderbooks(df_1, df_2, token_1, token_2, start_time=start_time)
        tick_df = make_df_from_orderbooks(df_1, df_2, token_1, token_2, start_time=start_time)
        agg_df = make_trunc_df(df, timeframe=tf, token_1=token_1, token_2=token_2, method='triple')

        return tick_df, agg_df
