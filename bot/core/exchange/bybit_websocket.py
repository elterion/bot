import orjson
import hmac
import time
import logging
from datetime import datetime

from bot.core.exchange.data import Orderbook
from bot.core.exchange.base_websocket import BaseWebSocketClient
from bot.config import credentials as cr

logging.basicConfig(format="%(message)s", level=logging.INFO)

logger = logging.getLogger()

class BybitWebSocketClient(BaseWebSocketClient):
    def __init__(self, demo: bool = False):
        super().__init__(demo)

        self.exchange = 'bybit'
        self.api_key = cr.BYBIT_DEMO_API_KEY if self.demo else cr.BYBIT_API_KEY
        self.api_secret = cr.BYBIT_DEMO_SECRET_KEY if self.demo else cr.BYBIT_SECRET_KEY
        self.orderbooks = dict()
        self.last_update_time = round(datetime.timestamp(datetime.now()), 1)
        self.last_update_hist = int(datetime.timestamp(datetime.now()))

    def _create_symbol_name(self, symbol, **kwargs):
        return ''.join(symbol.split('_'))

    def prepare_signature(self, expires: int) -> str:
        return hmac.new(self.api_secret.encode("utf-8"),
                    f"GET/realtime{expires}".encode("utf-8"),
                    digestmod="sha256").hexdigest()

    async def authenticate(self, endpoint: str):
        expires = int(time.time() * 1000) + 1000
        signature = self.prepare_signature(expires)
        auth_msg = {"op": "auth", "args": [self.api_key, expires, signature]}
        await self.connections[endpoint].send(orjson.dumps(auth_msg))
        logger.debug(f"Sent authentication request to Bybit ({endpoint})")

    def get_stream_type(self, msg):
        stream_type = ''

        if msg.get('op', '') or msg.get('success', ''):
            stream_type = 'system_message'
        elif msg.get('topic', '').startswith('publicTrade'):
            stream_type = 'public_trades'
        elif msg.get('topic', '').startswith('orderbook'):
            stream_type = 'orderbook'
        elif msg.get('topic', '') == 'order':
            stream_type = 'order'

        return stream_type

    async def default_handler(self, msg):
        stream_type = self.get_stream_type(msg)

        if stream_type == 'system_message' and msg.get('success', ''):
            logger.info('Соединение с биржей ByBit установлено.')
        elif stream_type == 'system_message':
            logger.info(msg)
        elif stream_type == 'orderbook':
            await self.handle_orderbook_stream(msg, verbose=False)
        elif stream_type == 'order':
            await self.handle_order_stream(msg)
        else:
            logger.info(f'bybit default handler: {msg}')

    async def handle_orderbook_stream(self, msg, verbose=False):
        msg_type = msg.get('type', '')
        data = msg.get('data', {})
        tkn = data.get('s', '')
        symbol = tkn[:-4] + '_' + tkn[-4:]
        cts = msg.get('cts', 0) // 1000
        bid = data.get('b', {})
        ask = data.get('a', {})

        if symbol not in self.orderbooks:
            self.orderbooks[symbol] = Orderbook(symbol=symbol)

        ob = self.orderbooks[symbol]

        if msg_type == "snapshot":
            ob.update_snapshot(bid, ask, cts)

        elif msg_type == "delta":
            ob.update_delta(bid, ask, cts)

        now = datetime.now()
        current_ts = round(datetime.timestamp(now), 1)
        current_second = now.second

        if current_ts - self.last_update_time > 0.2:
            self.last_update_time = current_ts
            self.postgre_client.update_current_ob(self.orderbooks)

        if current_ts - self.last_update_hist > 3 and current_second % 5 == 0:
            self.postgre_client.update_tick_ob(self.orderbooks)
            self.postgre_client.set_system_state('ws')
            self.last_update_hist = current_ts

    async def handle_order_stream(self, msg):
        ct = datetime.now().strftime('%H:%M:%S')
        data_arr = msg['data']

        for data in data_arr:
            order_id = data['orderId']
            symbol = data['symbol'].replace('USDT', '_USDT')
            status = data['orderStatus'].lower()
            side = data['side'].lower()

            if status == 'new':
                qty = data['qty']
                price = data['price']
                print(f'[ORDER PLACED] {side} {qty} {symbol} at {price}')

            elif status == 'filled':
                qty = data['cumExecQty']

                price = float(data['avgPrice'])
                usdt_value = float(data['cumExecValue'])
                usdt_fee = abs(float(data['cumExecFee']))
                pnl = float(data['closedPnl'])

                print(f'{ct} {side} {qty} {symbol} for {usdt_value}; pnl: {pnl:.4f}; fee: {usdt_fee:.6f}')
            elif status == 'cancelled':
                qty = data['qty']
                price = data['price']
                print(f'[ORDER CANCELLED] {side} {qty} {symbol} at {price}')
            elif status == 'deactivated' or status == 'untriggered':
                pass
            elif status == 'triggered':
                print('Сработал триггер!')

            else:
                print(data)

    async def subscribe_position_stream(self):
        sub_msg = {"op": "subscribe", "args": ['position']}
        await self.subscribe(endpoint='private', sub_msg=sub_msg)

    async def subscribe_orderbook_stream(self, depth, tickers):
        bybit_tokens = list(self.coin_info['bybit_linear'].keys())
        token_list = [tok for tok in tickers if tok in bybit_tokens]
        logger.info(f'{len(token_list)} Bybit connections.')
        args = []

        for token in token_list:
            sym = self._create_symbol_name(token)
            args.append(f'orderbook.{depth}.{sym}')

        sub_msg = {"op": "subscribe", "args": args}
        await self.subscribe(endpoint='linear', sub_msg=sub_msg)
