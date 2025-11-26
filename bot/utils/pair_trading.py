import polars as pl
import polars_ols as pls
import numpy as np
from numba import njit
from bot.utils.coins import get_step_info
import math
import ast
import random
from bot.analysis.strategy_analysis import analyze_strategy

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
