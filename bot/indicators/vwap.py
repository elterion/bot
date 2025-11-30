import polars as pl

def vwap(df, window=14):
    return df.with_columns(
        (pl.col("Volume") * (pl.col("High") + pl.col("Low") + pl.col("Close")) / 3.0
        ).rolling_sum(window_size=window).alias('cum_tpv')
    ).with_columns(
        pl.col('Volume').rolling_sum(window_size=window).alias('cum_vol')
    ).with_columns(
        (pl.col('cum_tpv') / pl.col('cum_vol')).alias('vwap')
    ).drop('cum_tpv', 'cum_vol')

def vwap_signal(df, col_name='vwap', threshold = 0.0):
    signal_col = f'{col_name}_signal'
    return df.with_columns(
            pl.when(pl.col('Close') - pl.col(col_name) > threshold).then(1)
                .when(pl.col('Close') - pl.col(col_name) < threshold).then(-1)
                .otherwise(0)  # Нейтральная зона
                .alias(signal_col))
