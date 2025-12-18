import requests
import pandas as pd
import polars as pl
from datetime import datetime
import pickle
from time import sleep

with open("./data/coin_information.pkl", "rb") as f:
    coin_information = pickle.load(f)

class BybitRestAPI():
    EXCHANGE_NAME = 'bybit'
    BASE_URL = "https://api.bybit.com"

    def __init__(self, category):
        self.category = category

    def _create_symbol_name(self, symbol, **kwargs):
        return ''.join(symbol.split('_'))

    def _parse_instr_data(self, data):
        instr_data = {}
        for ticker in data['result']['list']:
            if ticker['status'] != 'Trading':
                print(f'Bybit {ticker['symbol']} status is {ticker['status']}')

            if ticker['symbol'].endswith('USDT'):
                base = ticker['baseCoin']
                quote = ticker['quoteCoin']
                min_qty = float(ticker['lotSizeFilter']['minOrderQty'])
                price_scale = int(ticker['priceScale'])

                if self.category == 'linear':
                    qty_step = float(ticker['lotSizeFilter']['qtyStep'])
                    fund_interval = int(ticker['fundingInterval']) // 60
                    instr_data[base+'_'+quote] = {'min_qty': min_qty,
                                                  'qty_step': qty_step,
                                                  'ct_val': 1,
                                                  'fund_interval': fund_interval,
                                                  'price_scale': price_scale}
                elif self.category == 'spot':
                    qty_step = float(ticker['lotSizeFilter']['basePrecision'])
                    instr_data[base+'_'+quote] = {'min_qty': min_qty,
                                                  'qty_step': qty_step,
                                                  'price_scale': price_scale}
        return instr_data

    def get_instrument_data(self, symbol=''):
        if self.category == 'linear':
            endpoint = "/v5/market/instruments-info"
        elif self.category == 'spot':
            endpoint = "/v5/market/instruments-info"

        params = {'category': self.category, 'symbol': symbol, 'limit': 1000}

        r = None
        for _ in range(20):
            try:
                r = requests.request('GET', self.BASE_URL + endpoint, params=params).json()
                break
            except requests.exceptions.ConnectionError as err:
                print(f'get_instrument_data connection error.')
                sleep(5)

        if r is None:
            print('Не удалось скачать техническую информацию по монетам.')
            return None

        return self._parse_instr_data(r)

    def get_candles(self, symbol, interval, n_iters=1, end_date=None):
        symbol = self._create_symbol_name(symbol)
        endpoint = '/v5/market/kline'
        params = {'category': self.category, 'symbol': symbol, 'limit': 1000,
                  'interval': interval}
        if end_date is None:
            end_date = ''

        cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
        hist_df = pd.DataFrame(columns=cols)

        try:
            for _ in range(n_iters):
                data = requests.get(self.BASE_URL + endpoint, params=params).json()
                hist = data['result']['list']
                params['end'] = str(int(hist[-1][0]) - 1)

                tdf = pd.DataFrame(hist, columns=cols)
                hist_df = pd.concat([hist_df if not hist_df.empty else None, tdf],
                                    ignore_index=True)

        except KeyError as e:
            pass
        except Exception as e:
            pass

        hist_df[['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']] = hist_df[
            ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']].astype(float)

        hist_df['Date'] = pd.to_datetime(hist_df['Date'].astype(float), unit='ms')
        hist_df.index = hist_df['Date']
        hist_df.drop('Date', axis=1, inplace=True)
        hist_df.index = hist_df.index.tz_localize('UTC').tz_convert('Europe/Moscow')
        hist_df.sort_index(inplace=True)

        return hist_df[['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']]

    def get_tickers(self):
        endpoint = "/v5/market/tickers"
        params = {"category": self.category}
        exchange_rates = {}

        data = requests.get(url=self.BASE_URL+endpoint, params=params, timeout=(3, 4)).json()

        for ticker in data['result']['list']:
            vol24h = int(float(ticker['turnover24h']))

            if ticker['symbol'].endswith('USDT') and vol24h > 10_000:
                sym = ticker['symbol'][:-4] + '_' + ticker['symbol'][-4:]

                next_ft = datetime.fromtimestamp(int(ticker['nextFundingTime'][:-3])).strftime('%Y-%m-%d %H:%M')

                exchange_rates[sym] = {'bid_price': float(ticker['bid1Price']), 'ask_price': float(ticker['ask1Price']),
                                        'bid_size': float(ticker['bid1Size']), 'ask_size': float(ticker['ask1Size']),
                                        'last_price': float(ticker['lastPrice']), 'index_price': float(ticker['indexPrice']),
                                        'vol24h_usdt': float(ticker['turnover24h']),
                                        'funding_rate': float(ticker['fundingRate']), 'next_fund_time': next_ft}
        return exchange_rates

    def get_funding_history(self, symbol, start_date, end_date=None, limit=200):
        symbol = self._create_symbol_name(symbol)
        first_date = int(start_date.timestamp() * 1000)
        last_date = int(end_date.timestamp()*1000) if end_date else int(datetime.now().timestamp()*1000)

        main_df = pl.DataFrame()

        while last_date > first_date:
            params = {'category': 'linear',
                      'symbol': symbol,
                      'limit': limit,
                      'startTime': first_date,
                      'endTime': last_date
                  }
            endpoint = '/v5/market/funding/history'

            res = requests.get(self.BASE_URL + endpoint, params=params, timeout=(3, 4)).json()
            lst = res['result']['list']

            if lst:
                df = pl.DataFrame(lst).rename({'fundingRate': 'funding',
                                                'fundingRateTimestamp': 'ts'})
                df = df.with_columns(
                    pl.when(pl.col("symbol").str.ends_with("USDT"))
                        .then(pl.col("symbol").str.replace(r"(USDT)$", ""))
                        .otherwise(pl.col("symbol")),
                    pl.col('funding').cast(pl.Float64),
                    pl.col('ts').cast(pl.Int64),
                    pl.col('ts').cast(pl.Int64).cast(pl.Datetime("ms")
                        ).dt.convert_time_zone("Europe/Moscow").alias('time')
                )
                last_date = df['ts'][-1] - 1
                main_df = main_df.vstack(df)
            else:
                break


        main_df = main_df.with_columns(
                (pl.col('ts') // 1_000)
            )
        return main_df
