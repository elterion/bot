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

    tasks = [
        bybit_client.connect("orderbook", auth_required=False),

        manage_subscriptions(bybit_client=bybit_client,
                             token_list=token_list),
    ]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    token_list = ['1INCH_USDT', 'APT_USDT', 'ARB_USDT', 'ARKM_USDT',
              'BLUR_USDT',
              'CELO_USDT', 'CHZ_USDT', 'CRV_USDT', 'CVX_USDT',
              'DOT_USDT', 'DYDX_USDT', 'FIL_USDT', 'FLOW_USDT',
              'GALA_USDT', 'GMT_USDT', 'GRT_USDT', 'JASMY_USDT',
              'IMX_USDT', 'IOTA_USDT', 'KAS_USDT', 'KSM_USDT',
              'LDO_USDT',
              'MANA_USDT', 'MANTA_USDT', 'MORPHO_USDT', 'MOVE_USDT', 'NEAR_USDT',
              'ONDO_USDT', 'OP_USDT', 'ORDI_USDT',
              'POL_USDT', 'RENDER_USDT', 'ROSE_USDT',
              'SAND_USDT', 'SEI_USDT', 'STRK_USDT',
              'STX_USDT', 'SUI_USDT', 'SUSHI_USDT',
              'TIA_USDT', 'VET_USDT', 'XRP_USDT',
              'ZEN_USDT', 'ZK_USDT']

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
