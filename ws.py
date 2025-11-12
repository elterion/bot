import asyncio
from asyncio.exceptions import CancelledError
import logging
from time import sleep
from bot.core.exchange.bybit_websocket import BybitWebSocketClient
from bot.core.exceptions.connection import ConnectionLostError

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

async def manage_subscriptions(bybit_client=None, okx_client=None, gate_client=None, token_list=None):
    await bybit_client.subscribe_orderbook_stream(depth=50, tickers=token_list)

async def main(demo=False, token_list=None):
    bybit_client = BybitWebSocketClient(demo=demo)
    logger.info('Очищаем таблицу PostgreSQL')
    bybit_client.postgre_client.clear_table('current_ob')

    tasks = [
        bybit_client.connect("orderbook", auth_required=False),
        manage_subscriptions(bybit_client=bybit_client,
                             token_list=token_list),
    ]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    with open('./bot/config/tokens.txt', 'r') as file:
        token_list = [line.strip() for line in file.readlines()]

    demo = False

    while True:
        try:
            asyncio.run(main(demo=demo, token_list=token_list))
        except ConnectionLostError as err:
            sleep(10)
            logger.info('Рестарт программы.')
            asyncio.run(main(demo=demo, token_list=token_list))
        except (KeyboardInterrupt, CancelledError):
            logger.info('Завершение работы.')
            break
