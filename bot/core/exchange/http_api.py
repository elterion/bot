from aiohttp import ClientSession, ClientError, ClientResponseError
import asyncio
import requests
import pandas as pd
import polars as pl
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
import logging
import pickle
from time import sleep

# Настройка логгирования
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logging.getLogger('aiohttp').setLevel('ERROR')
logging.getLogger('asyncio').setLevel('ERROR')
logger = logging.getLogger()

with open("./data/coin_information.pkl", "rb") as f:
    coin_information = pickle.load(f)

class ExchangeRestAPI(ABC):
    interval_dic = {'bybit': {'1m': 1, '5m': 5, '15m': 15, '30m': 30,
                            '1h': 60, '4h': 240, '1d': 'D'},
                    'okx': {'1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                            '1h': '1H', '4h': '4H', '1d': '1Dutc'},
                    'gate': {'1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                            '1h': '1h', '4h': '4h', '1d': '1d'}}

    def __init__(self, category):
        self.category = category
        assert self.category in ('spot', 'linear'), 'category should be "spot" or "linear"'

    async def _send_request(self, session, endpoint, params=None, n_tries=50,  base_delay=1, timeout=10):
        params = params or {}
        # reconnect_delay = 1
        for attempt in range(1, n_tries + 1):
            try:
                async with session.get(
                    self.BASE_URL + endpoint, params=params, timeout=timeout
                    ) as response:
                        response.raise_for_status()
                        # reconnect_delay = 1
                        return await response.json()
            except (asyncio.TimeoutError, ClientError) as e:
                print('.', end='')
                if attempt > 1:
                    logger.debug(f"Ошибка соединения: {self.BASE_URL + endpoint}. Попытка {attempt}/{n_tries}. {e}")

                await asyncio.sleep(5)
                # reconnect_delay = min(reconnect_delay * 2, 60)

    @abstractmethod
    def _create_symbol_name(self, symbol):
        pass

    @abstractmethod
    async def get_tickers(self, session):
        pass

    async def _get_candles_params_to_send(self):
        pass

    async def _get_candles_workflow_params(self):
        pass

    @abstractmethod
    def _get_instr_params(self):
        pass

    @abstractmethod
    def _parse_instr_data(self, data):
        pass

    @abstractmethod
    def _get_orderbook_params(self):
        pass

    @abstractmethod
    def _parse_orderbook_data(self, data):
        pass

    async def get_orderbook(self, session, symbol, limit):
        if self.category == 'linear':
            endpoint = self.ORDERBOOK_LINEAR_ENDPOINT
        elif self.category == 'spot':
            endpoint = self.ORDERBOOK_SPOT_ENDPOINT

        symbol = self._create_symbol_name(market_type=self.category, symbol=symbol)
        params = self._get_orderbook_params(symbol, limit)

        try:
            data = await self._send_request(session, endpoint=endpoint, params=params)
        except ClientResponseError:
            return {}
        return self._parse_orderbook_data(data)

    def get_instrument_data(self, symbol=''):
        if self.category == 'linear':
            endpoint = self.INSTR_LINEAR_ENDPOINT
        elif self.category == 'spot':
            endpoint = self.INSTR_SPOT_ENDPOINT

        params = self._get_instr_params(symbol)

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

    async def get_candles(self, session, symbol, interval, n_iters=1, end_date=None):
        assert interval in ('1m', '5m', '15m', '30m', '1h', '4h', '1d'), "possible interval values: '1m', '5m', '15m', '30m', '1h', '4h', '1d'"

        return await self._get_candles(session=session,
                                       symbol=symbol,
                                       interval=interval,
                                       n_iters=n_iters,
                                       end_date=end_date)

class BybitRestAPI(ExchangeRestAPI):
    EXCHANGE_NAME = 'bybit'
    BASE_URL = "https://api.bybit.com"
    INSTR_SPOT_ENDPOINT = "/v5/market/instruments-info"
    INSTR_LINEAR_ENDPOINT = "/v5/market/instruments-info"
    ORDERBOOK_SPOT_ENDPOINT = '/v5/market/orderbook'
    ORDERBOOK_LINEAR_ENDPOINT = '/v5/market/orderbook'
    FUNDING_HISTORY_ENDPOINT = '/v5/market/funding/history'

    def _create_symbol_name(self, symbol, **kwargs):
        return ''.join(symbol.split('_'))

    def _get_instr_params(self, symbol=''):
        return {'category': self.category, 'symbol': symbol, 'limit': 1000}

    def _parse_instr_data(self, data):
        instr_data = {}
        for ticker in data['result']['list']:
            if ticker['status'] != 'Trading':
                logger.warning(f'Bybit {ticker['symbol']} status is {ticker['status']}')

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

    def _get_orderbook_params(self, symbol, limit):
        return {'category': self.category, 'symbol': symbol, 'limit': limit}

    def _parse_orderbook_data(self, data):
        try:
            ask = data['result']['a']
            bid = data['result']['b']
            res_dic = {'ask': [[float(x[0]), float(x[1])] for x in ask],
                        'bid': [[float(x[0]), float(x[1])] for x in bid]}
            return res_dic
        except KeyError:
            return {}

    async def _get_candles(self, session, symbol, interval, n_iters=1, end_date=None):
        symbol = self._create_symbol_name(symbol)
        endpoint = '/v5/market/kline'
        params = {'category': self.category, 'symbol': symbol, 'limit': 1000,
                  'interval': self.interval_dic['bybit'][interval]}
        if end_date is None:
            end_date = ''

        cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
        hist_df = pd.DataFrame(columns=cols)

        try:
            for _ in range(n_iters):
                data = await self._send_request(session, endpoint=endpoint, params=params)

                if not data:
                    logger.warning(f"No data returned for {symbol}.")
                    break

                hist = data['result']['list']
                end_date = str(int(hist[-1][0]) - 1)
                params['end'] = end_date

                tdf = pd.DataFrame(hist, columns=cols)
                hist_df = pd.concat([hist_df if not hist_df.empty else None, tdf],
                                    ignore_index=True)

        except KeyError as e:
            # logger.warning(f'No data for {exc} {self.category}')
            pass
        except Exception as e:
            # logger.error(f"{exc} {self.category}. Error fetching data: {e}")
            pass

        hist_df[['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']] = hist_df[
            ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']].astype(float)

        hist_df['Date'] = pd.to_datetime(hist_df['Date'].astype(float), unit='ms')
        hist_df.index = hist_df['Date']
        hist_df.drop('Date', axis=1, inplace=True)
        hist_df.index = hist_df.index.tz_localize('UTC').tz_convert('Europe/Moscow')
        hist_df.sort_index(inplace=True)
        hist_df['Exchange'] = 'bybit'
        hist_df['Market_type'] = self.category

        return hist_df[['Open', 'High', 'Low', 'Close', 'Volume',
                            'Turnover', 'Exchange', 'Market_type']]

    async def get_tickers(self, session):
        endpoint = "/v5/market/tickers"
        params = {"category": self.category}
        exchange_rates = {}

        data = await self._send_request(session, endpoint=endpoint, params=params)

        for ticker in data['result']['list']:
            vol24h = int(float(ticker['turnover24h']))

            if ticker['symbol'].endswith('USDT') and vol24h > 1_000:
                sym = ticker['symbol'][:-4] + '_' + ticker['symbol'][-4:]

                if self.category == 'linear':
                    next_ft = datetime.fromtimestamp(int(ticker['nextFundingTime'][:-3])).strftime('%Y-%m-%d %H:%M')

                    exchange_rates[sym] = {'bid_price': float(ticker['bid1Price']), 'ask_price': float(ticker['ask1Price']),
                                           'bid_size': float(ticker['bid1Size']), 'ask_size': float(ticker['ask1Size']),
                                           'last_price': float(ticker['lastPrice']), 'index_price': float(ticker['indexPrice']),
                                           'vol24h_usdt': float(ticker['turnover24h']),
                                           'funding_rate': float(ticker['fundingRate']), 'next_fund_time': next_ft}
                elif self.category == 'spot':
                    exchange_rates[sym] = {'bid_price': float(ticker['bid1Price']), 'ask_price': float(ticker['ask1Price']),
                                           'bid_size': float(ticker['bid1Size']), 'ask_size': float(ticker['ask1Size']),
                                           'last_price': float(ticker['lastPrice']), 'vol24h_usdt': vol24h}
        return exchange_rates

    async def get_funding_history(self, session, symbol, start_date, end_date=None, limit=200):
        symbol = self._create_symbol_name(symbol)
        first_date = int(start_date.timestamp() * 1000)
        last_date = int(end_date.timestamp()*1000) if end_date else int(datetime.now().timestamp()*1000)

        main_df = pd.DataFrame(columns=['symbol', 'funding', 'ts'])

        while last_date > first_date:
            params = {'category': 'linear', 'symbol': symbol, 'limit': limit,
                    'startTime': first_date,
                    'endTime': last_date
                  }
            res = await self._send_request(session,
                                       endpoint=self.FUNDING_HISTORY_ENDPOINT,
                                       params=params)
            lst = res['result']['list']

            df = pd.DataFrame(lst).rename(columns={'fundingRate': 'funding',
                                               'fundingRateTimestamp': 'ts'})
            if not df.empty:
                last_date = int(df.iloc[-1]['ts']) - 1
                main_df = pd.concat([main_df, df])
            else:
                break

        main_df['date'] = pd.to_datetime(main_df['ts'], unit='ms', utc=True
                                    ).dt.tz_convert('Europe/Moscow')
        main_df = main_df.set_index('date').sort_index().drop('ts', axis=1)
        main_df['funding'] = main_df['funding'].astype(float)

        return main_df


class OKXRestAPI(ExchangeRestAPI):
    EXCHANGE_NAME = 'okx'
    BASE_URL = 'https://www.okx.com'
    INSTR_SPOT_ENDPOINT = '/api/v5/public/instruments'
    INSTR_LINEAR_ENDPOINT = '/api/v5/public/instruments'
    ORDERBOOK_SPOT_ENDPOINT = '/api/v5/market/books'
    ORDERBOOK_LINEAR_ENDPOINT = '/api/v5/market/books'

    def _create_symbol_name(self, market_type, symbol):
        sym = '-'.join(symbol.split('_'))
        if market_type == 'linear':
            sym += '-SWAP'
        return sym

    def _get_instr_params(self, symbol=''):
        if self.category == 'linear':
            return {'instType': 'SWAP'}
        elif self.category == 'spot':
            return {'instType': 'SPOT'}


    def _parse_instr_data(self, data):
        instr_data = {}
        for ticker in data['data']:
            if self.category == 'linear':
                if ticker['instFamily'].endswith('USDT'):
                    base = ticker['ctValCcy']
                    quote = ticker['settleCcy']
                    ct_val = float(ticker['ctVal'])
                    min_qty = float(ticker['minSz']) * ct_val
                    qty_step = float(ticker['lotSz']) * ct_val
                    price_scale = int(len(ticker['tickSz'].split('.')[-1]))

                    instr_data[base+'_'+quote] = {'ct_val': ct_val,
                                        'min_qty': min_qty,
                                        'qty_step': qty_step,
                                        'price_scale': price_scale}
            elif self.category == 'spot':
                if ticker['instId'].endswith('USDT'):
                    base = ticker['baseCcy']
                    quote = ticker['quoteCcy']
                    min_qty = float(ticker['minSz'])
                    qty_step = float(ticker['lotSz'])
                    price_scale = int(len(ticker['tickSz'].split('.')[-1]))

                    instr_data[base+'_'+quote] = {'min_qty': min_qty,
                                                  'qty_step': qty_step,
                                                  'price_scale': price_scale}
        return instr_data

    def _get_orderbook_params(self, symbol, limit):
        return {'instId': symbol, 'sz': limit}

    def _parse_orderbook_data(self, data):
        try:
            ask = data['data'][0]['asks']
            bid = data['data'][0]['bids']
            res_dic = {'ask': [[float(x[0]), float(x[1])] for x in ask],
                        'bid': [[float(x[0]), float(x[1])] for x in bid]}
            return res_dic
        except KeyError:
            return {}

    async def get_candles(self, session, symbol, interval, n_iters=1, end_date=None):
        symbol = self._create_symbol_name(symbol=symbol, market_type=self.category)
        endpoint = '/api/v5/market/history-candles'

        cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', '0', 'Turnover', '1']
        cols_to_drop = ['0', '1']

        params = {'instId': symbol,
                  'bar': self.interval_dic['okx'][interval]}
        n_iters *= 10

        if end_date is None:
            end_date = ''

        hist_df = pd.DataFrame(columns=cols)
        try:
            for _ in range(n_iters):
                data = await self._send_request(session, endpoint=endpoint, params=params)

                if not data:
                    logger.warning(f"No data returned for {symbol}.")
                    break

                hist = data['data']
                end_date = str(int(hist[-1][0]) - 1)
                params['after'] = end_date

                tdf = pd.DataFrame(hist, columns=cols)
                hist_df = pd.concat([hist_df if not hist_df.empty else None, tdf],
                    ignore_index=True)

        except KeyError as e:
            # logger.warning(f'No data for {exc} {self.category}')
            pass
        except Exception as e:
            # logger.error(f"{exc} {self.category}. Error fetching data: {e}")
            pass

        if cols_to_drop:
            try:
                hist_df.drop(cols_to_drop, axis=1, inplace=True)
            except KeyError:
                logger.error(f"Error drop columns. OKX, {self.category=}")
                raise KeyError

        hist_df[['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']] = hist_df[
            ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']].astype(float)

        hist_df['Date'] = pd.to_datetime(hist_df['Date'].astype(float), unit='ms')
        hist_df.index = hist_df['Date']
        hist_df.drop('Date', axis=1, inplace=True)
        hist_df.index = hist_df.index.tz_localize('UTC').tz_convert('Europe/Moscow')
        hist_df.sort_index(inplace=True)
        hist_df['Exchange'] = 'okx'
        hist_df['Market_type'] = self.category

        if self.category == 'linear':
            sym = symbol.split('-')
            ct_val = coin_information['okx_linear'][sym[0] + '_' + sym[1]]['ct_val']
            hist_df['Volume'] *= float(ct_val)

        return hist_df[['Open', 'High', 'Low', 'Close', 'Volume',
                            'Turnover', 'Exchange', 'Market_type']]

    async def get_tickers(self, session):
        category = self.category.upper() if self.category =='spot' else 'SWAP'

        endpoint = '/api/v5/market/tickers'
        params = {"instType": category}
        exchange_rates = {}

        data = await self._send_request(session, endpoint=endpoint, params=params)

        for ticker in data['data']:
            vol24h = int(float(ticker['volCcy24h']))
            coins = ticker['instId'].split('-')

            if coins[1] == 'USDT' and vol24h > 1_000:
                sym = coins[0] + '_' + coins[1]
                exchange_rates[sym] = {'bid_price': float(ticker['bidPx']), 'ask_price': float(ticker['askPx']),
                               'bid_size': float(ticker['bidSz']), 'ask_size': float(ticker['askSz']),
                               'last_price': float(ticker['last']), 'vol24h_usdt': vol24h}
        return exchange_rates


class GateIORestAPI(ExchangeRestAPI):
    EXCHANGE_NAME = 'gate'
    BASE_URL = 'https://api.gateio.ws'
    INSTR_SPOT_ENDPOINT = '/api/v4/spot/currency_pairs'
    INSTR_LINEAR_ENDPOINT = '/api/v4/futures/usdt/contracts'
    ORDERBOOK_SPOT_ENDPOINT = '/api/v4/spot/order_book'
    ORDERBOOK_LINEAR_ENDPOINT = '/api/v4/futures/usdt/order_book'
    FUNDING_HISTORY_ENDPOINT = '/api/v4/futures/usdt/funding_rate'
    TRADING_HISTORY_ENDPOINT = '/api/v4/futures/usdt/trades'

    def _create_symbol_name(self, symbol, **kwargs):
        return symbol

    def _get_instr_params(self, symbol=''):
        return {'Accept': 'application/json', 'Content-Type': 'application/json'}

    def _parse_instr_data(self, data):
        instr_data = {}
        for ticker in data:
            if self.category == 'spot':
                if ticker['id'].endswith('USDT'):
                    base = ticker['base']
                    quote = ticker['quote']
                    min_qty = float(ticker['min_base_amount'])

                    prec = int(ticker['amount_precision'])
                    qty_step = round(1 / (10 ** prec), prec)
                    price_scale = int(len(ticker['order_price_round'].split('.')[-1]))
                    instr_data[base+'_'+quote] = {'min_qty': min_qty,
                                                  'qty_step': qty_step,
                                                  'price_scale': price_scale}
            elif self.category == 'linear':
                if ticker['name'].endswith('USDT'):
                    base = ticker['name'].split('_')[0]
                    quote = ticker['name'].split('_')[-1]
                    ct_val = float(ticker['quanto_multiplier'])
                    min_qty = float(ticker['order_size_min']) * ct_val
                    qty_step = float(ticker['order_size_min']) * ct_val
                    fund_interval = ticker.get('funding_interval', 0) // 3600
                    next_fund_time = ticker.get('funding_next_apply', 0)
                    nft = datetime.fromtimestamp(next_fund_time).strftime('%Y-%m-%d %H:%M')
                    price_scale = int(len(ticker['order_price_round'].split('.')[-1]))

                    if ticker['in_delisting']:
                        logger.warning(f'Coin {ticker['name']} in delisting on Gate.io!')

                    instr_data[base+'_'+quote] = {'min_qty': min_qty,
                                                  'qty_step': qty_step,
                                                  'ct_val': ct_val,
                                                  'fund_interval': fund_interval,
                                                  'next_fund_time': nft,
                                                  'price_scale': price_scale}
        return instr_data

    def _get_orderbook_params(self, symbol, limit):
        if self.category == 'linear':
            return {'contract': symbol, 'limit': limit,
                'Accept': 'application/json', 'Content-Type': 'application/json'}
        elif self.category == 'spot':
            return {'currency_pair': symbol, 'limit': limit,
                'Accept': 'application/json', 'Content-Type': 'application/json'}

    def _parse_orderbook_data(self, data):
        try:
            res_dic = {'ask': [[float(x['p']), float(x['s'])] for x in data['asks']],
                    'bid': [[float(x['p']), float(x['s'])] for x in data['bids']]}

            return res_dic
        except TypeError:
            res_dic = {'ask': [[float(x[0]), float(x[1])] for x in data['asks']],
                    'bid': [[float(x[0]), float(x[1])] for x in data['bids']]}

            return res_dic
        except KeyError:
            return {}

    async def get_candles(self, session, symbol, interval, n_iters=1, end_date=None):
        params = {'Accept': 'application/json', 'Content-Type': 'application/json'}

        if self.category == 'spot':
            endpoint = '/api/v4/spot/candlesticks'
            cols = ['Date', 'Volume', 'Close', 'High', 'Low', 'Open', 'Turnover', '0']
            cols_to_drop = ['0']
            params['currency_pair'] = symbol
        elif self.category == 'linear':
            endpoint = '/api/v4/futures/usdt/candlesticks'
            cols = ['Date', 'Volume', 'Close', 'High', 'Low', 'Open', 'Turnover']
            cols_to_drop = []
            params['contract'] = symbol

        params['interval'] = self.interval_dic['gate'][interval]
        n_iters *= 10

        if end_date is None:
            end_date = ''

        hist_df = pd.DataFrame(columns=cols)

        try:
            for _ in range(n_iters):
                hist = await self._send_request(session, endpoint=endpoint, params=params)

                if not hist:
                    logger.warning(f"No data returned for {symbol}.")
                    break

                if self.category == 'spot':
                    end_date = str(int(hist[0][0]) - 1)
                elif self.category == 'linear':
                    end_date = str(int(hist[0]['t']) - 1)

                params['to'] = end_date
                tdf = pd.DataFrame(hist)
                hist_df = pd.concat([hist_df if not hist_df.empty else None, tdf],
                    ignore_index=True)

        except KeyError as e:
            # logger.warning(f'No data for {exc} {self.category}')
            pass
        except Exception as e:
            # logger.error(f"{exc} {self.category}. Error fetching data: {e}")
            pass

        if self.category == 'linear':
            hist_df.rename(columns={'o': 'Open', 'v': 'Volume', 't': 'Date',
                'c': 'Close', 'l': 'Low', 'h': 'High', 'sum': 'Turnover'},
                inplace=True)
            hist['Turnover'] = hist['Turnover'].astype(float)
        elif self.category == 'spot':
            hist_df.rename(columns={0: 'Date', 1: 'Volume', 2: 'Close',
                3: 'High', 4: 'Low', 5: 'Open', 6: 'Volume', 7: '0'},
                inplace=True)

        if cols_to_drop:
            try:
                hist_df.drop(cols_to_drop, axis=1, inplace=True)
            except KeyError:
                logger.error(f"Error drop columns. Gate.io, {self.category=}")
                raise KeyError

        hist_df[['Open', 'High', 'Low', 'Close', 'Volume']] = hist_df[
            ['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

        hist_df['Date'] = pd.to_datetime(hist_df['Date'].astype(float), unit='s')
        hist_df.index = hist_df['Date']
        hist_df.drop('Date', axis=1, inplace=True)
        hist_df.index = hist_df.index.tz_localize('UTC').tz_convert('Europe/Moscow')
        hist_df.sort_index(inplace=True)
        hist_df['Exchange'] = 'gate'
        hist_df['Market_type'] = self.category

        if self.category == 'linear':
            ct_val = coin_information['gate_linear'][symbol]['ct_val']
            hist_df['Volume'] *= float(ct_val)

        if self.category == 'linear':
            return hist_df[['Open', 'High', 'Low', 'Close', 'Volume',
                                'Turnover', 'Exchange', 'Market_type']]
        elif self.category == 'spot':
            return hist_df[['Open', 'High', 'Low', 'Close', 'Volume',
                                'Exchange', 'Market_type']]

    async def get_tickers(self, session):
        if self.category == 'spot':
            endpoint = '/api/v4/spot/tickers'
        elif self.category == 'linear':
            endpoint = '/api/v4/futures/usdt/tickers'
        params = {'Accept': 'application/json', 'Content-Type': 'application/json'}

        exchange_rates = {}

        data = await self._send_request(session, endpoint=endpoint, params=params)

        for ticker in data:
            if self.category == 'linear':
                vol24h = int(float(ticker['volume_24h_quote']))
            elif self.category == 'spot':
                vol24h = int(float(ticker['quote_volume']))
            sym = ticker['contract'] if self.category == 'linear' else ticker['currency_pair']

            if sym.endswith('USDT') and vol24h > 1_000:
                if self.category == 'spot':
                    try:
                        exchange_rates[sym] = {'bid_price': float(ticker['highest_bid']), 'ask_price': float(ticker['lowest_ask']),
                                       'last_price': float(ticker['last']), 'vol24h_usdt': vol24h}
                    except ValueError:
                        continue
                elif self.category == 'linear':
                    try:
                        exchange_rates[sym] = {'bid_price': float(ticker['highest_bid']), 'ask_price': float(ticker['lowest_ask']),
                                       'bid_size': float(ticker['highest_size']), 'ask_size': float(ticker['lowest_size']),
                                       'last_price': float(ticker['last']), 'index_price': float(ticker['index_price']),
                                       'vol24h_usdt': vol24h, 'funding_rate': float(ticker['funding_rate'])}
                    except ValueError:
                        continue
        return exchange_rates

    async def get_funding_history(self, session, symbol, start_date, end_date=None, limit=1):
        first_date = int(start_date.timestamp())
        last_date = int(end_date.timestamp()) if end_date else int(datetime.now().timestamp())

        params = {'Accept': 'application/json', 'Content-Type': 'application/json',
                  'contract': self._create_symbol_name(symbol),
                  'limit': limit, 'from': first_date,
                  'to': last_date}

        res = await self._send_request(session, endpoint=self.FUNDING_HISTORY_ENDPOINT, params=params)
        df = pd.DataFrame(res).rename(columns={'r': 'funding', 't': 'ts'})
        df['date'] = pd.to_datetime(df['ts'], unit='s')
        df['date'] = df['date'].dt.tz_localize('UTC')
        df['funding'] = df['funding'].astype(float)
        return df[['date', 'funding']].sort_index(ascending=False
                                ).set_index('date')

    async def get_trading_history(self, session, symbol, start_date, end_date=None, limit=1000):
        end_date = end_date if end_date else int(datetime.now().timestamp())
        first_date = int(start_date.timestamp())

        main_df = pd.DataFrame(columns=['date', 'contract', 'size', 'price', 'side'])

        while end_date > first_date:
            print(datetime.fromtimestamp(end_date))

            params = {'Accept': 'application/json', 'Content-Type': 'application/json',
                  'contract': self._create_symbol_name(symbol),
                  'limit': limit, 'to': end_date}

            res = await self._send_request(session,
                                        endpoint=self.TRADING_HISTORY_ENDPOINT,
                                        params=params)

            tr_df = pd.DataFrame(res)
            tr_df['date'] = pd.to_datetime(tr_df['create_time'], unit='s', utc=True)
            tr_df['side'] = np.where(tr_df['size'] > 0, 'buy', 'sell')
            tr_df['size'] = tr_df['size'].abs()

            try:
                end_date = int(tr_df.iloc[-1]['create_time'])
            except NameError:
                pass

            tr_df = tr_df.drop(['create_time', 'create_time_ms'], axis=1)
            main_df = pd.concat([main_df, tr_df])

        main_df['exchange'] = 'gate'
        main_df = main_df[['date', 'exchange', 'contract', 'size', 'price', 'side']
                       ].sort_values(by='date', ascending=True)


        return main_df[main_df['date'] > start_date].set_index('date')


class ExchangeManager:
    def __init__(self):
        self.exchanges = {}

    def add_market(self, name: str, exchange: ExchangeRestAPI):
        self.exchanges[name] = exchange

    async def get_prices(self):
        async with ClientSession() as session:
            tasks = {
                name: asyncio.create_task(exchange.get_tickers(session))
                for name, exchange in self.exchanges.items()
            }

            results = {}
            for name, task in tasks.items():
                results[name] = await task

            return results


    async def get_candles(self, symbol, interval, n_iters=1):
        async with ClientSession() as session:
            tasks = {
                name: exchange.get_candles(session, symbol, interval, n_iters)
                for name, exchange in self.exchanges.items()
            }

            results = {}
            for name, task in tasks.items():
                results[name] = await task

            return results

    def get_instrument_data(self, symbol=''):
        results = {}
        for name, task in self.exchanges.items():
            results[name] = task.get_instrument_data()

        return results

    async def get_orderbook(self, symbol, limit):
        async with ClientSession() as session:
            tasks = {
                name: asyncio.create_task(exchange.get_orderbook(session, symbol, limit))
                for name, exchange in self.exchanges.items()
            }

            results = {}
            for name, task in tasks.items():
                results[name] = await task

            return results

    async def get_funding_history(self, symbol, start_date, end_date=None, limit=1):
        async with ClientSession() as session:
            tasks = {
                name: asyncio.create_task(exchange.get_funding_history(session,
                                                                       symbol,
                                                                       start_date,
                                                                       end_date,
                                                                       limit))
                for name, exchange in self.exchanges.items()
            }

            return await asyncio.gather(*tasks.values(), return_exceptions=True)

    async def get_trading_history(self, symbol, start_date, end_date=None, limit=1):
        async with ClientSession() as session:
            tasks = {
                name: asyncio.create_task(exchange.get_trading_history(session,
                                                                       symbol,
                                                                       start_date,
                                                                       end_date,
                                                                       limit))
                for name, exchange in self.exchanges.items()
            }

            return await asyncio.gather(*tasks.values(), return_exceptions=True)
