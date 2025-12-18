import os
import polars as pl
from requests.exceptions import Timeout, ConnectionError
from datetime import datetime
from time import sleep
from bot.core.exchange.http_api import BybitRestAPI
from bot.core.db.postgres_manager import DBManager
from bot.config.credentials import host, user, password, db_name

def get_funding():
    with open('./bot/config/tokens.txt', 'r') as file:
        token_list = [line.strip() for line in file.readlines()]

    exc_manager = BybitRestAPI('linear')
    prices = exc_manager.get_tickers()

    df = pl.DataFrame()
    for token, data in prices.items():
        if token in token_list:
            fr = data['funding_rate']
            ft = data.get('next_fund_time', '')

            df = df.vstack(pl.DataFrame({'symbol': token,
                            'funding': fr, 'time': ft}))
    df = df.with_columns(
            pl.col('time').str.to_datetime(format="%Y-%m-%d %H:%M"),
            pl.when(pl.col("symbol").str.ends_with("USDT"))
                .then(pl.col("symbol").str.replace(r"(_USDT)$", ""))
                .otherwise(pl.col("symbol"))
        ).with_columns(
            (pl.col('time').dt.timestamp(time_unit='ms') // 1000).alias('ts')
        )

    return df.select("symbol", "funding", "ts", "time")

def main():
    db_params = {'host': host, 'user': user, 'password': password, 'dbname': db_name}
    db_manager = DBManager(db_params)

    while True:
        try:
            os.system( 'cls' )
            df = get_funding()
            db_manager.update_funding(df)

            ct = datetime.now().strftime('%H:%M:%S')
            print(f'Последнее обновление: {ct}')

            sleep(5 * 60)
        except KeyboardInterrupt:
            print('Завершение работы.')
            break
        except (Timeout, ConnectionError) as err:
            print(f'{ct} Timeout error.')
            sleep(5)


if __name__ == '__main__':
    main()
