import argparse
from time import sleep
from datetime import datetime
import polars as pl
import pickle
from requests.exceptions import Timeout, ConnectionError

from bot.core.db.postgres_manager import DBManager
from bot.config.credentials import host, user, password, db_name
from bot.core.exchange.trade_api import Trade

def main(demo):
    if demo:
        print('DEMO mode.')
    else:
        print('========= REAL MONEY mode! =========')

    db_params = {'host': host, 'user': user, 'password': password, 'dbname': db_name}
    postgre_manager = DBManager(db_params)
    trade_manager = Trade(demo=demo)

    with open("./data/coin_information.pkl", "rb") as f:
        coin_information = pickle.load(f)

    ct = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{ct} Начинаем работу...')

    last_time = int(datetime.timestamp(datetime.now()))
    err_counter = 0

    while err_counter < 50:
        try:
            ct = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f'Текущее время: {ct}', end='\r')

            # Устанавливаем heartbeat отметку
            postgre_manager.set_system_state('trades_executor')

            # Загружаем данные
            pairs = postgre_manager.get_table('pairs', df_type='polars')
            pending_orders = pairs.filter(pl.col('status').is_in(['opening', 'closing']))
            active_orders = pairs.filter(pl.col('status') == 'active')

            now = int(datetime.timestamp(datetime.now()))

            # ------------ Обновление открытых позиций ------------
            if now - last_time > 10:
                active_positions = trade_manager.get_all_positions(market_type='linear')

                pairs_data = []

                for row in active_orders.iter_rows(named=True):
                    t1 = row['token_1']
                    t2 = row['token_2']

                    for position in active_positions:
                        if position['token'] == t1:
                            qty_1 = position['qty']
                            rpnl_1 = position['realized_pnl']
                            upnl_1 = position['unrealized_pnl']
                        elif position['token'] == t2:
                            qty_2 = position['qty']
                            rpnl_2 = position['realized_pnl']
                            upnl_2 = position['unrealized_pnl']

                    pairs_data.append([t1, t2, rpnl_1, rpnl_2, upnl_1, upnl_2])

                postgre_manager.update_pairs(pairs_data)
                last_time = int(datetime.timestamp(datetime.now()))

            # ------------ Обработка открытия и закрытия ордеров ------------
            for row in pending_orders.iter_rows(named=True):
                status = row['status']
                token_1 = row['token_1']
                token_2 = row['token_2']

                # Проверим время постановки ордера. Если ордер был открыт больше 10 секунд назад
                # и до сих пор не отправлен на биржу, то отменяем
                now = int(datetime.timestamp(datetime.now()))
                open_ts = int(datetime.timestamp(row['created_at']))

                if now - open_ts > 10 and status == 'opening':
                    postgre_manager.delete_pair_order(token_1, token_2)
                    continue

                side_1 = row['side_1']
                side_2 = row['side_2']
                qty_1 = row['qty_1']
                qty_2 = row['qty_2']
                price_1 = row['open_price_1']
                price_2 = row['open_price_2']
                leverage = row['leverage']

                ps_1 = coin_information['bybit_linear'][token_1]['price_scale']
                ps_2 = coin_information['bybit_linear'][token_1]['price_scale']

                if status == 'opening':
                    sl_1 = round(price_1 - 0.8 * price_1 / leverage, ps_1) if side_1 == 'long' else round(price_1 + 0.8 * price_1 / leverage, ps_1)
                    sl_2 = round(price_2 - 0.8 * price_2 / leverage, ps_2) if side_2 == 'long' else round(price_2 + 0.8 * price_2 / leverage, ps_2)
                    act_1 = 'Buy' if side_1 == 'long' else 'Sell'
                    act_2 = 'Buy' if side_2 == 'long' else 'Sell'

                    rsp = trade_manager.place_pair_order('linear', token_1, act_1, qty_1, sl_1, token_2, act_2, qty_2, sl_2)
                    for r in rsp:
                        res = trade_manager.get_order('linear', order_id=r)
                        if res['token'] == token_1:
                            open_price_1 = res['price']
                        if res['token'] == token_2:
                            open_price_2 = res['price']

                    postgre_manager.commit_pair_order(token_1, token_2, open_price_1, open_price_2)
                    print(f'{ct}. Open position. {act_1} {token_1[:-5]}; {act_2} {token_2[:-5]}')
                    err_counter = 0

                elif status == 'closing':
                    sl_1, sl_2 = None, None
                    act_1 = 'Buy' if side_1 == 'short' else 'Sell'
                    act_2 = 'Buy' if side_2 == 'short' else 'Sell'

                    rsp = trade_manager.place_pair_order('linear', token_1, act_1, qty_1, sl_1, token_2, act_2, qty_2, sl_2)

                    for r in rsp:
                        res = trade_manager.get_order('linear', order_id=r)
                        if res['token'] == token_1:
                            close_price_1 = res['price']
                            close_fee_1 = res['fee']
                        if res['token'] == token_2:
                            close_price_2 = res['price']
                            close_fee_2 = res['fee']

                    postgre_manager.complete_pair_order(token_1, token_2, close_price_1, close_price_2,
                                        close_fee_1, close_fee_2)
                    print(f'{ct}. Close position. {act_1} {token_1[:-5]}; {act_2} {token_2[:-5]}')
                    err_counter = 0

            sleep(0.5)

        except (Timeout, ConnectionError) as err:
            print('Timeout error:', err)
            sleep(5)
        except KeyError as err:
            print('Не удалось обновить позицию:', err)
            err_counter += 1
            sleep(5)
        except KeyboardInterrupt:
            print('\nЗавершение работы.')
            break
        except Exception as err:
            print(f'{ct} {type(err).__name__}: {err}')
            err_counter += 1
            sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Мониторинг торгового бота")
    parser.add_argument('--demo', action='store_true', help='Включить демонстрационный режим')
    args = parser.parse_args()
    demo = args.demo

    main(demo)
