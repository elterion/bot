import os
import polars as pl
import requests
from requests.exceptions import ReadTimeout
from datetime import datetime
from time import sleep
from tqdm import tqdm
import asyncio
from urllib3.exceptions import ConnectTimeoutError, ProtocolError
from bot.data.http_api import ExchangeManager, BybitRestAPI, OKXRestAPI, GateIORestAPI
from bot.db.postgres_manager import DBManager
from bot.config.credentials import host, user, password, db_name

db_params = {'host': host, 'user': user, 'password': password, 'dbname': db_name}
db_manager = DBManager(db_params)


def get_okx_funding(symbol):
    base_url = "https://www.okx.com"
    endpoint = "/api/v5/public/funding-rate"

    syms = symbol.split('_')
    params = {"instId": f"{syms[0]}-{syms[1]}-SWAP"}
    while True:
        try:
            response = requests.get(base_url + endpoint, params=params, timeout=10)
            break
        except (ConnectTimeoutError, ProtocolError, ReadTimeout):
            sleep(2)


    fr, nft = None, None
    if response.status_code == 200:
        data = response.json()
        try:
            # print(data["data"])
            fr = round(float(data["data"][0]["fundingRate"]), 6) * 100
            next_fund_time = int(data["data"][0]["fundingTime"]) // 1000
            nft = datetime.fromtimestamp(next_fund_time).strftime('%Y-%m-%d %H:%M')

            next_fund_time_1 = int(data["data"][0]["nextFundingTime"]) // 1000
            fund_interval = int((next_fund_time_1 - next_fund_time) / 3600)
            # print(fund_interval)
        except IndexError:
            pass

    return pl.DataFrame({'token': symbol, 'exchange': 'okx', 'ask_price': 0.0, 'bid_price': 0.0,
                        'funding_rate': fr, 'fund_interval': fund_interval, 'next_fund_time': nft})

async def get_funding():
    print('Собираем информацию с ByBit и Gate.io')
    exc_manager = ExchangeManager()
    exc_manager.add_market("bybit_linear", BybitRestAPI('linear'))
    exc_manager.add_market("okx_linear", OKXRestAPI('linear'))
    exc_manager.add_market("gate_linear", GateIORestAPI('linear'))

    res = await exc_manager.get_prices()
    instr = exc_manager.get_instrument_data()

    okx_tokens = [token for token, data in res['okx_linear'].items() if data['vol24h_usdt'] > 200_000]

    df = pl.DataFrame()
    for exc, data in res.items():
        if exc == 'okx_linear':
            continue
        for token, token_data in data.items():
            # print(token_data)
            ask_price = token_data['ask_price']
            bid_price = token_data['bid_price']

            if token.startswith('10000000'):
                token = token[8:]
                ask_price *= 10_000_000
                bid_price *= 10_000_000
            elif token.startswith('1000000'):
                token = token[7:]
                ask_price *= 1_000_000
                bid_price *= 1_000_000
            elif token.startswith('100000'):
                token = token[6:]
                ask_price *= 100_000
                bid_price *= 100_000
            elif token.startswith('10000'):
                token = token[5:]
                ask_price *= 10_000
                bid_price *= 10_000
            elif token.startswith('1000'):
                token = token[4:]
                ask_price *= 1000
                bid_price *= 1000
            fr = token_data['funding_rate']
            ft = token_data.get('next_fund_time', '')

            df = df.vstack(pl.DataFrame({'token': token, 'exchange': exc.split('_')[0],
                                         'ask_price': ask_price,
                                         'bid_price': bid_price,
                                         'funding_rate': fr, 'next_fund_time': ft}))

    fund_int_df = pl.DataFrame()
    for exc, data in instr.items():
        if exc in ('bybit_linear', 'gate_linear'):
            for token, token_data in data.items():
                fund_interval = token_data.get('fund_interval', 0)
                next_fund_time = token_data.get('next_fund_time', "")
                if token.startswith('10000000'):
                    token = token[8:]
                elif token.startswith('1000000'):
                    token = token[7:]
                elif token.startswith('100000'):
                    token = token[6:]
                elif token.startswith('10000'):
                    token = token[5:]
                elif token.startswith('1000'):
                    token = token[4:]

                # print(token, fund_interval, next_fund_time)
                fund_int_df = fund_int_df.vstack(pl.DataFrame({'token': token,
                                                               'exchange': exc.split('_')[0],
                                                               'fund_interval': fund_interval,
                                                               'next_fund_time': next_fund_time}))
        else:
            continue

    df = df.join(fund_int_df, on=('token', 'exchange'), how='inner', suffix='_r').with_columns(
                pl.when(pl.col("next_fund_time") != "")
                  .then(pl.col("next_fund_time"))
                  .otherwise(pl.col("next_fund_time_r"))
                  .alias("combined_fund_time")
            ).drop('next_fund_time', 'next_fund_time_r'
            ).rename({'combined_fund_time': 'next_fund_time'}
            ).with_columns(pl.col('funding_rate') * 100
            )

    print('Скачиваем информацию с Okx')
    okx_df = pl.DataFrame()
    for token in tqdm(okx_tokens[:]):
        okx_df = okx_df.vstack(get_okx_funding(token))

    return df.vstack(okx_df)

async def main():
    while True:
        try:
            os.system( 'cls' )
            res = await get_funding()

            tokens = res['token'].value_counts().filter(
                pl.col("count") > 1
                ).select('token'
                         ).to_series().to_list()
            df = res.filter(pl.col("token").is_in(tokens))
            db_manager.clear_table('funding_data')
            db_manager.update_funding_data(df)

            ct = datetime.now().strftime('%H:%M:%S')
            print(f'Последнее обновление: {ct}')

            sleep(5 * 60)
        except KeyboardInterrupt:
            print('Завершение работы.')
            break


if __name__ == '__main__':
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except (asyncio.exceptions.CancelledError, KeyboardInterrupt):
        print('Завершение работы.')
