from bot.config.credentials import host, user, password, db_name
from bot.core.db.postgres_manager import DBManager
from bot.config import credentials as cr

from bot.core.exceptions.trading import SetLeverageError, PlaceOrderError, NoSuchOrderError

from datetime import datetime, UTC
import requests
import hmac
import hashlib
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger('aiohttp').setLevel('ERROR')
logging.getLogger('asyncio').setLevel('ERROR')
logger = logging.getLogger()

def set_leverage(demo, symbol, leverage):
    exc = Trade(demo=demo)
    resp_lever = exc.set_leverage(market_type='linear', symbol=symbol, lever=leverage)

class Trade():
    def __init__(self, demo=False):
        self.demo = demo
        if self.demo:
            self.api_key = cr.BYBIT_DEMO_API_KEY
            self.secret_key = cr.BYBIT_DEMO_SECRET_KEY
            self.main_url = 'https://api-demo.bybit.com'
            logger.debug('Demo-mode is on!')
        else:
            self.api_key = cr.BYBIT_API_KEY
            self.secret_key = cr.BYBIT_SECRET_KEY
            self.main_url = 'https://api.bybit.com'
            logger.debug('Demo-mode is off!')

    def _create_symbol_name(self, symbol):
        return ''.join(symbol.split('_'))

    def hashing(self, query):
        return hmac.new(self.secret_key.encode('utf-8'), query.encode('utf-8'), hashlib.sha256).hexdigest()

    def _prepare_headers(self, ts, sign):
        headers = {
            'X-BAPI-API-KEY': self.api_key,
            'X-BAPI-TIMESTAMP': str(ts),
            'X-BAPI-SIGN': sign,
            'X-BAPI-RECV-WINDOW': str(5000),
            'Content-Type': 'application/json'
        }
        return headers

    def _place_order(self, market_type, symbol, side, order_type, qty, price=None,
                      trigger_price=None, stop_loss=None):
        url = self.main_url + '/v5/order/create'

        sym = self._create_symbol_name(symbol)
        side = side.capitalize()
        price = price if price else ''
        curr_time = int(datetime.now().timestamp()*1000)
        data = f'{{"category": "{market_type}","symbol": "{sym}","side": "{side}","orderType": "{order_type}","qty": "{qty}"}}'
        if price:
            data = data[:-1] + f',"price": "{price}"'

        if trigger_price:
            if side == 'Buy':
                data = data[:-1] + f',"triggerPrice": "{trigger_price}","triggerDirection": "1"}}'
            elif side == 'Sell':
                data = data[:-1] + f',"triggerPrice": "{trigger_price}", "triggerDirection": "2"}}'
        if stop_loss:
            data = data[:-1] + f',"stopLoss": "{stop_loss}","slTriggerBy": "LastPrice"}}'

        sign = self.hashing(str(curr_time) + self.api_key + '5000' + data)
        headers = self._prepare_headers(ts=curr_time, sign=sign)
        response = requests.post(url=url, headers=headers, data=data, timeout=(3, 4)).json()

        order_id = response.get('result', {}).get('orderId')
        if order_id:
            logger.debug(f'[PLACE ORDER] Bybit {market_type=} {sym=} {order_id=}')
            return order_id
        else:
            logger.error(f'[ORDER ERROR] Bybit {market_type=} {sym=}')
            raise PlaceOrderError(f'При постановке ордера на бирже Bybit возникла ошибка: {response['retMsg']}')

    def place_market_order(self, market_type, symbol, side, qty, stop_loss=None):
        return self._place_order(market_type=market_type,
                      symbol=symbol,
                      side=side,
                      order_type='market',
                      qty=qty,
                      stop_loss=stop_loss)

    def place_limit_order(self, market_type, symbol, side, qty, price, stop_loss=None, **kwargs):
        return self._place_order(market_type=market_type,
                      symbol=symbol,
                      side=side,
                      order_type='limit',
                      qty=qty,
                      price=price,
                      stop_loss=stop_loss)

    def place_conditional_order(self, market_type, symbol, side, qty, price, trigger_price,
                                 stop_loss=None):
        return self._place_order(market_type=market_type,
                      symbol=symbol,
                      side=side,
                      order_type='limit',
                      qty=qty,
                      price=price,
                      trigger_price=trigger_price,
                      stop_loss=stop_loss)

    def place_pair_order(self, market_type, symbol_1, side_1, qty_1, stop_loss_1,
                         symbol_2, side_2, qty_2, stop_loss_2):
        market_type = market_type.lower()
        side_1 = side_1.lower()
        side_2 = side_2.lower()
        assert market_type in ('linear', 'spot'), 'market_type should be "linear" or "spot"'
        assert side_1 in ('buy', 'sell'), 'side_1 should be "buy" or "sell"'
        assert side_2 in ('buy', 'sell'), 'side_2 should be "buy" or "sell"'

        curr_time = int(datetime.now().timestamp()*1000)

        url = self.main_url + '/v5/order/create-batch'

        sym_1 = self._create_symbol_name(symbol_1)
        sym_2 = self._create_symbol_name(symbol_2)
        side_1 = side_1.capitalize()
        side_2 = side_2.capitalize()

        order_1 = f'{{"symbol": "{sym_1}", "side": "{side_1}", "orderType": "Market", "qty": "{qty_1}"}}'
        order_2 = f'{{"symbol": "{sym_2}", "side": "{side_2}", "orderType": "Market", "qty": "{qty_2}"}}'



        if stop_loss_1:
            order_1 = order_1[:-1] + f', "stopLoss": "{stop_loss_1}", "slTriggerBy": "LastPrice"}}'
        if stop_loss_2:
            order_2 = order_2[:-1] + f', "stopLoss": "{stop_loss_2}", "slTriggerBy": "LastPrice"}}'


        data = f'{{"category": "{market_type}", "request": [{order_1}, {order_2}]}}'


        sign = self.hashing(str(curr_time) + self.api_key + '5000' + data)
        headers = self._prepare_headers(ts=curr_time, sign=sign)
        response = requests.post(url=url, headers=headers, data=data, timeout=(3, 4)).json()

        status = response.get('retMsg', '')

        if status == 'OK':
            ids = []

            for order in response['result']['list']:
                sym = order.get('symbol', '')
                id = order.get('orderId', '')
                ids.append(id)
                logger.debug(f'[PLACE ORDER] Bybit {market_type=} {sym=} {id=}')
            return ids
        else:
            logger.error(f'[ORDER ERROR] Bybit {market_type=} {sym_1=} {sym_2=}')
            raise PlaceOrderError(f'При постановке ордера на бирже Bybit возникла ошибка: {response['retMsg']}')


    def cancel_order(self, market_type, symbol, order_id, **kwargs):
        url = self.main_url + '/v5/order/cancel'
        sym = self._create_symbol_name(symbol)
        curr_time = int(datetime.now().timestamp()*1000)
        data = '{' + f'"category": "{market_type}","symbol": "{sym}","orderId": "{order_id}"' + '}'
        sign = self.hashing(str(curr_time) + self.api_key + '5000' + data)
        headers = self._prepare_headers(ts=curr_time, sign=sign)

        response = requests.post(url=url, headers=headers, data=data, timeout=(3, 4)).json()
        order_id = response.get('result', {}).get('orderId')
        if response.get('retCode') == 0 and response.get('retMsg') == 'OK':
            logger.debug(f'[CANCEL ORDER] Bybit {market_type=} {sym=} {order_id=}')
            return order_id
        else:
            logger.error(f'[ORDER ERROR] Bybit {market_type=} {sym=}')
            raise NoSuchOrderError(f'При отмене ордера на бирже Bybit возникла ошибка: {response['retMsg']}')

    def set_leverage(self, market_type, symbol, lever):
        url = self.main_url + '/v5/position/set-leverage'

        sym = self._create_symbol_name(symbol)
        curr_time = int(datetime.now().timestamp()*1000)
        data = '{' + f'"category": "{market_type}","symbol": "{sym}","buyLeverage": "{lever}","sellLeverage": "{lever}"' + '}'
        sign = self.hashing(str(curr_time) + self.api_key + '5000' + data)
        headers = self._prepare_headers(ts=curr_time, sign=sign)

        response = requests.post(url=url, headers=headers, data=data, timeout=(3, 4)).json()
        if response['retCode'] == 0 or (response['retCode'] == 110043
                                    and response['retMsg'] == 'leverage not modified'):
            logger.debug(f'Bybit {sym} leverage successfully set to {lever}')
        else:
            logger.error(f'[LEVERAGE ERROR] Bybit {sym}')
            raise SetLeverageError(f'При изменении плеча на бирже Bybit возникла ошибка: {response['retMsg']}')
        return response

    def _position_handler(self, resp):
        market_type = resp['result']['category']

        all_positions = []
        for data in resp['result']['list']:
            try:
                token = data['symbol']
                base, quote = token[:-4], token[-4:]
                token = base + '_' + quote

                usdt_amount = float(data['positionBalance']) # Начальная стоимость в usdt. Да, проверил.
                if usdt_amount == 0:
                    continue

                leverage = float(data['leverage'])
                price = float(data['avgPrice'])
                side = data['side'].lower()
                size = float(data['size'])
                realized_pnl = float(data['curRealisedPnl'])
                unrealized_pnl = float(data['unrealisedPnl'])

                pos = {'exchange': 'bybit', 'market_type': market_type,
                    'token': token, 'leverage': leverage, 'price': price,
                    'usdt_amount': usdt_amount, 'qty': size, 'order_side': side,
                    'realized_pnl': realized_pnl, 'unrealized_pnl': unrealized_pnl}
                all_positions.append(pos)

            except ValueError as err:
                print(data)
                raise NoSuchOrderError

        return all_positions

    def get_position(self, market_type, symbol):
        """
        Возвращает информацию по открытой позиции
        """
        url = self.main_url + '/v5/position/list'

        sym = self._create_symbol_name(symbol)
        curr_time = int(datetime.now().timestamp()*1000)
        query = f'category={market_type}&symbol={sym}'

        sign = self.hashing(str(curr_time) + self.api_key + '5000' + query)
        headers = self._prepare_headers(ts=curr_time, sign=sign)

        url = f'{url}?{query}'
        response = requests.get(url=url, headers=headers, timeout=(3, 4)).json()

        pos = self._position_handler(response)

        if pos:
            return pos[0]
        else:
            return None

    def get_all_positions(self, market_type):
        """
        Возвращает информацию по всем открытым позициям.
        """
        url = self.main_url + f'/v5/position/list'
        curr_time = int(datetime.now().timestamp()*1000)
        query = f'category={market_type}&settleCoin=USDT'

        sign = self.hashing(str(curr_time) + self.api_key + '5000' + query)
        headers = self._prepare_headers(ts=curr_time, sign=sign)

        url = f'{url}?{query}'
        response = requests.get(url=url, headers=headers, timeout=(3, 4)).json()

        positions = self._position_handler(response)
        return positions

    def _order_handler(self, resp, market_type):
        data = resp['result']['list'][0]

        token = data['symbol']
        if token.endswith('USDT'):
            base, quote = token[:-4], token[-4:]
            token = base + '_' + quote
        else:
            raise NotImplementedError

        try:
            order_type = data['orderType'].lower()
            status = data['orderStatus'].lower()
            price = float(data['avgPrice']) # Цена по которой ордер сматчился
            limit_price = float(data['price']) # Заявочная цена для лимитного ордера
            side = data['side'].lower()
            size = float(data['cumExecQty'])
            fee = -float(data['cumExecFee'])
            usdt_amount = size * price
        except ValueError as err:
            print(data)
            raise NoSuchOrderError

        return {'exchange': 'bybit', 'market_type': market_type, 'order_type': order_type,
                'status': status, 'token': token, 'price': price, 'limit_price': limit_price,
                'usdt_amount': usdt_amount, 'qty': size, 'order_side': side, 'fee': fee}

    def get_order(self, market_type, order_id, **kwargs):
        url = self.main_url + '/v5/order/realtime'

        curr_time = int(datetime.now().timestamp()*1000)
        query = f'category={market_type}&orderId={order_id}'

        sign = self.hashing(str(curr_time) + self.api_key + '5000' + query)
        headers = self._prepare_headers(ts=curr_time, sign=sign)

        url = f'{url}?{query}'
        response = requests.get(url=url, headers=headers, timeout=(3, 4)).json()

        # return response
        return self._order_handler(response, market_type)

    def get_order_history(self, market_type):
        url = self.main_url + '/v5/order/realtime'

        curr_time = int(datetime.now().timestamp()*1000)
        query = f'category={market_type}&settleCoin=USDT&limit=50'

        sign = self.hashing(str(curr_time) + self.api_key + '5000' + query)
        headers = self._prepare_headers(ts=curr_time, sign=sign)

        url = f'{url}?{query}'
        response = requests.get(url=url, headers=headers).json()

        return response
        try:
            return self._order_handler(response, market_type)
        except InvalidOperation:
            return None
