import psycopg
from psycopg.rows import dict_row
import pandas as pd
import polars as pl
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

class DBManager:
    def __init__(self, db_params):
        self.conn = psycopg.connect(**db_params)
        self.conn.autocommit = True

    def close(self):
        self.conn.close()

    def set_system_state(self, module_name):
        Moscow_TZ = timezone(timedelta(hours=3))
        now = datetime.now(Moscow_TZ)
        ts = int(datetime.timestamp(now))
        time = now.strftime('%Y-%m-%d %H:%M:%S')

        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO system_state (module_name, update_time, update_ts)
                VALUES (%s, %s, %s)
                ON CONFLICT (module_name)
                DO UPDATE SET
                    update_time = EXCLUDED.update_time,
                    update_ts = EXCLUDED.update_ts
            """, (module_name, time, ts))

    def get_system_state(self, module_name):
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT update_ts
                FROM system_state
                WHERE module_name = %s
            """, (module_name,))
            result = cur.fetchone()
            return result[0] if result else None

    def update_funding_data(self, records):
        """
        Обновление данных фандинга. Если запись с таким же (token, exchange)
        уже существует, она будет заменена новыми данными.

        :param records: список кортежей, где каждый кортеж имеет вид:
                        (token, exchange, ask_price, bid_price, funding_rate, fund_interval, next_fund_time)
        """
        sql = """
        INSERT INTO funding_data (token, exchange, ask_price, bid_price, funding_rate, fund_interval, next_fund_time)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (token, exchange) DO UPDATE SET
            ask_price = EXCLUDED.ask_price,
            bid_price = EXCLUDED.bid_price,
            funding_rate = EXCLUDED.funding_rate,
            fund_interval = EXCLUDED.fund_interval,
            next_fund_time = EXCLUDED.next_fund_time;
        """
        if isinstance(records, pl.DataFrame):
            records = records.rows()

        with self.conn.cursor() as cursor:
            cursor.executemany(sql, records)

    def add_pair_order(self, token_1, token_2, created_at, side_1, side_2, qty_1, qty_2,
                       price_1, price_2, usdt_1, usdt_2, leverage, status):
        """Добавляет новый ордер в таблицу pairs"""

        # Если created_at не передан, используем текущее время
        if created_at is None:
            Moscow_TZ = timezone(timedelta(hours=3))
            created_at = datetime.now(Moscow_TZ).strftime('%Y-%m-%d %H:%M:%S')

        query = """
        INSERT INTO pairs (token_1, token_2, created_at, side_1, side_2, qty_1, qty_2,
        open_price_1, open_price_2, usdt_1, usdt_2, leverage, rpnl_1, rpnl_2, upnl_1, upnl_2,
        profit_1, profit_2, profit, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 0, 0, 0, 0, 0, 0, 0, %s)
        """

        with self.conn.cursor() as cursor:
            cursor.execute(query, (token_1, token_2, created_at, side_1, side_2, qty_1, qty_2,
                                   price_1, price_2, usdt_1, usdt_2, leverage, status))

    def update_pairs(self, pairs_data):
        """
        Пакетно обновляет записи в таблице pairs за один SQL-запрос.

        Для каждой записи обновляет значения PnL (реализованного и нереализованного)
        и пересчитывает производные показатели прибыли.

        Parameters
        ----------
        pairs_data : list of list
            Список элементов, где каждый элемент представляет собой список вида:
            [token_1, token_2, rpnl_1, rpnl_2, upnl_1, upnl_2]

            Где:
            - token_1, token_2 : ключевые идентификаторы пары
            - rpnl_1, rpnl_2 : реализованный PnL для первой и второй позиции
            - upnl_1, upnl_2 : нереализованный PnL для первой и второй позиции

        Notes
        -----
        - Обновляет столбцы: rpnl_1, rpnl_2, upnl_1, upnl_2, profit_1, profit_2, profit
        - profit_1 = rpnl_1 + upnl_1
        - profit_2 = rpnl_2 + upnl_2
        - profit = profit_1 + profit_2
        - Поиск записи для обновления осуществляется по составному ключу (token_1, token_2)
        - Операция выполняется атомарно - либо обновляются все записи, либо ни одной

        Examples
        --------
        >>> pairs_data = [
        ...     ["BTC", "ETH", 100.5, -50.2, 25.3, 10.1],
        ...     ["SOL", "ADA", -20.0, 30.5, 15.2, -5.5],
        ...     ["DOT", "LINK", 5.5, 7.8, -2.3, 3.1]
        ... ]
        >>> db_manager.update_pairs_batch(pairs_data)
        """

        query = """
            UPDATE pairs
            SET rpnl_1 = %s,
                rpnl_2 = %s,
                upnl_1 = %s,
                upnl_2 = %s,
                profit_1 = %s + %s,
                profit_2 = %s + %s,
                profit = (%s + %s) + (%s + %s)
            WHERE token_1 = %s AND token_2 = %s
        """

        # Подготавливаем данные для executemany
        batch_data = []
        for item in pairs_data:
            token_1, token_2, rpnl_1, rpnl_2, upnl_1, upnl_2 = item
            # Каждая запись содержит все параметры в порядке, соответствующем запросу
            batch_data.append((
                rpnl_1, rpnl_2, upnl_1, upnl_2,  # SET значения
                rpnl_1, upnl_1,                   # для profit_1
                rpnl_2, upnl_2,                   # для profit_2
                rpnl_1, upnl_1, rpnl_2, upnl_2,   # для profit
                token_1, token_2                  # WHERE условия
            ))

        with self.conn.cursor() as cursor:
            cursor.executemany(query, batch_data)

    def commit_pair_order(self, token_1, token_2, open_price_1, open_price_2):
        """Обновляет статус ордера на 'active' по ключу (token_1, token_2, side)"""
        query = """
            UPDATE pairs
            SET status = 'active',
                open_price_1 = %s,
                open_price_2 = %s

            WHERE token_1 = %s AND token_2 = %s
        """

        with self.conn.cursor() as cursor:
            cursor.execute(query, (open_price_1, open_price_2, token_1, token_2))

    def close_pair_order(self, token_1, token_2, side_1):
        """Обновляет статус ордера на 'closing' по ключу (token_1, token_2, side_1)"""
        query = """
            UPDATE pairs
            SET status = 'closing'
            WHERE token_1 = %s AND token_2 = %s AND side_1 = %s
        """

        with self.conn.cursor() as cursor:
            cursor.execute(query, (token_1, token_2, side_1))

    def complete_pair_order(self, token_1, token_2, close_price_1, close_price_2,
                            close_fee_1, close_fee_2):
        with self.conn.cursor() as cursor:
            cursor.execute("""
                SELECT token_1, token_2, created_at, side_1, side_2, qty_1, qty_2,
                    open_price_1, open_price_2, usdt_1, usdt_2, rpnl_1, rpnl_2, leverage
                FROM pairs
                WHERE token_1 = %s AND token_2 = %s
            """, (token_1, token_2))

            position = cursor.fetchone()

        if not position:
            raise Exception(f"Ордер не найден: {token_1, token_2}")

        t1 = token_1[:-5] if token_1.endswith('_USDT') else token_1
        t2 = token_2[:-5] if token_2.endswith('_USDT') else token_2

        open_time = position[2]
        side_1 = position[3]
        side_2 = position[4]
        qty_1 = position[5]
        qty_2 = position[6]
        open_price_1 = position[7]
        open_price_2 = position[8]
        open_fee_1 = position[11]
        open_fee_2 = position[12]
        leverage = position[13]

        fee_1 = open_fee_1 + close_fee_1
        fee_2 = open_fee_2 + close_fee_2

        if side_1 == 'long':
            pnl_1 = round(qty_1 * (close_price_1 - open_price_1) + fee_1, 8)
            pnl_2 = round(qty_2 * (open_price_2 - close_price_2) + fee_2, 8)
        elif side_1 == 'short':
            pnl_1 = round(qty_1 * (open_price_1 - close_price_1) + fee_1, 8)
            pnl_2 = round(qty_2 * (close_price_2 - open_price_2) + fee_2, 8)

        profit = round(pnl_1 + pnl_2, 8)

        Moscow_TZ = timezone(timedelta(hours=3))
        close_time = datetime.now(Moscow_TZ).strftime('%Y-%m-%d %H:%M:%S')

        query = """
            INSERT INTO trading_history (token_1, token_2, open_time, close_time, side_1, side_2, qty_1, qty_2,
                open_price_1, open_price_2, close_price_1, close_price_2, fee_1, fee_2,
                leverage, pnl_1, pnl_2, profit)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        with self.conn.cursor() as cursor:
            cursor.execute(query, (t1, t2, open_time, close_time, side_1, side_2, qty_1, qty_2,
                open_price_1, open_price_2, close_price_1, close_price_2, fee_1, fee_2,
                leverage, pnl_1, pnl_2, profit))

        self.delete_pair_order(token_1, token_2)

    def delete_pair_order(self, token_1, token_2):
        """Удаляет запись из таблицы pairs по ключу (token_1, token_2)"""
        query = """DELETE FROM pairs
        WHERE token_1 = %s AND token_2 = %s"""
        with self.conn.cursor() as cur:
            cur.execute(query, (token_1, token_2))

    def add_data_to_zscore_history(self, data):
        """
        Добавляет список записей в таблицу
        data: список кортежей в формате (ts, exchange, token_1, token_2, z_score, profit)
        """
        with self.conn.cursor() as cur:
            # Преобразуем данные в нужный формат
            records = [
                (ts, exchange, token_1, token_2, profit, z_score)
                for (ts, exchange, token_1, token_2, profit, z_score) in data
            ]

            # Выполняем массовую вставку
            cur.executemany(
                "INSERT INTO zscore_history (ts, exchange, token_1, token_2, profit, z_score) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                records
            )

    def get_zscore_history(self, token_1, token_2, start_ts, end_ts):
        query = """
            SELECT ts, time, exchange, token_1, token_2, profit, z_score
            FROM zscore_history
            WHERE token_1 = %s
              AND token_2 = %s
              AND ts >= %s
              AND ts <= %s
            ORDER BY time;
        """

        params = [token_1, token_2, start_ts, end_ts]

        with self.conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]

        return pl.DataFrame(rows, schema=colnames, orient="row")

    def add_orderbook(self, symbol: str,
                     time: datetime, bid_price: float, bid_volume: float,
                     ask_price: float, ask_volume: float):
        """
        Сохраняет состояние биржевого стакана в таблицу tick_ob

        Args:
            symbol: символ торговой пары
            time: время обновления
            bid_price: цена лучшего бида
            bid_volume: объем лучшего бида
            ask_price: цена лучшего аска
            ask_volume: объем лучшего аска
        """

        query = """
            INSERT INTO tick_ob (
                token, time,
                bid_price, bid_size, ask_price, ask_size
            ) VALUES (%s, %s, %s, %s, %s, %s)
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (
                symbol, time, bid_price, bid_volume, ask_price, ask_volume
            ))

    def add_orderbook_bulk(self, df):
        query = """
            INSERT INTO tick_ob (
                token, time, bid_price, bid_size, ask_price, ask_size
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (token, time)
                DO NOTHING
        """

        data = df.rows()
        with self.conn.cursor() as cur:
            cur.executemany(query, data)

    def update_tick_ob(self, orderbooks):
        rows = []
        for name, ob in orderbooks.items():
            top_bids = ob.get_top_n_bids(1)
            top_asks = ob.get_top_n_asks(1)
            update_ts = ob.update_ts

            update_time = datetime.fromtimestamp(update_ts, tz=ZoneInfo("Europe/Moscow"))

            row = (
                ob.symbol, update_time,
                top_bids[0][0], top_bids[0][1],
                top_asks[0][0], top_asks[0][1],
            )
            rows.append(row)

        query = """
                INSERT INTO tick_ob (
                    token, time, bid_price, bid_size, ask_price, ask_size
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (token, time) DO UPDATE SET
                    time = EXCLUDED.time,
                    bid_price = EXCLUDED.bid_price,
                    bid_size = EXCLUDED.bid_size,
                    ask_price = EXCLUDED.ask_price,
                    ask_size = EXCLUDED.ask_size
            """

        with self.conn.cursor() as cur:
            cur.executemany(query, rows)

    def update_current_ob(self, orderbooks, top_n = 5):
        """
        Массовая вставка нескольких ордербуков

        Args:
            orderbooks: список объектов Orderbook
            top_n: количество уровней для сохранения
        """
        rows = []
        for name, ob in orderbooks.items():
            top_bids = ob.get_top_n_bids(top_n)
            top_asks = ob.get_top_n_asks(top_n)
            update_ts = ob.update_ts

            while len(top_bids) < top_n:
                top_bids.append((0.0, 0.0))
            while len(top_asks) < top_n:
                top_asks.append((0.0, 0.0))

            update_time = datetime.fromtimestamp(update_ts, tz=ZoneInfo("Europe/Moscow"))

            row = (
                ob.symbol, ob.update_ts, update_time,
                top_bids[0][0], top_bids[0][1],
                top_bids[1][0], top_bids[1][1],
                top_bids[2][0], top_bids[2][1],
                top_bids[3][0], top_bids[3][1],
                top_bids[4][0], top_bids[4][1],
                top_asks[0][0], top_asks[0][1],
                top_asks[1][0], top_asks[1][1],
                top_asks[2][0], top_asks[2][1],
                top_asks[3][0], top_asks[3][1],
                top_asks[4][0], top_asks[4][1],
            )
            rows.append(row)

        query = """
            INSERT INTO current_ob (
                token, update_ts, update_time,
                bid_price_0, bid_volume_0, bid_price_1, bid_volume_1,
                bid_price_2, bid_volume_2, bid_price_3, bid_volume_3,
                bid_price_4, bid_volume_4,
                ask_price_0, ask_volume_0, ask_price_1, ask_volume_1,
                ask_price_2, ask_volume_2, ask_price_3, ask_volume_3,
                ask_price_4, ask_volume_4
            ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (token) DO UPDATE SET
                    update_ts = EXCLUDED.update_ts,
                    update_time = EXCLUDED.update_time,
                    bid_price_0 = EXCLUDED.bid_price_0,
                    bid_volume_0 = EXCLUDED.bid_volume_0,
                    bid_price_1 = EXCLUDED.bid_price_1,
                    bid_volume_1 = EXCLUDED.bid_volume_1,
                    bid_price_2 = EXCLUDED.bid_price_2,
                    bid_volume_2 = EXCLUDED.bid_volume_2,
                    bid_price_3 = EXCLUDED.bid_price_3,
                    bid_volume_3 = EXCLUDED.bid_volume_3,
                    bid_price_4 = EXCLUDED.bid_price_4,
                    bid_volume_4 = EXCLUDED.bid_volume_4,
                    ask_price_0 = EXCLUDED.ask_price_0,
                    ask_volume_0 = EXCLUDED.ask_volume_0,
                    ask_price_1 = EXCLUDED.ask_price_1,
                    ask_volume_1 = EXCLUDED.ask_volume_1,
                    ask_price_2 = EXCLUDED.ask_price_2,
                    ask_volume_2 = EXCLUDED.ask_volume_2,
                    ask_price_3 = EXCLUDED.ask_price_3,
                    ask_volume_3 = EXCLUDED.ask_volume_3,
                    ask_price_4 = EXCLUDED.ask_price_4,
                    ask_volume_4 = EXCLUDED.ask_volume_4
        """

        with self.conn.cursor() as cur:
            cur.executemany(query, rows)

    def get_orderbooks(
        self,
        symbol: str | None = None,
        interval: str = "5m",   # параметр агрегации
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pl.DataFrame:
        """
        Получает историю изменения ордербука по конкретной монете

        Args:
            symbol: символ торговой пары
            interval: окно агрегации (например: "5m", "1h")
            start_date: начало периода (UTC)
            end_date: конец периода (UTC)

        Returns:
            polars.DataFrame с историей ордербука
        """

        valid_intervals = {"5m", "1h", "4h"}
        if interval not in valid_intervals:
            raise ValueError(f"Недопустимый интервал: {interval}. Разрешено: {valid_intervals}")

        table = f"orderbook_{interval}"

        # Базовый запрос
        query = f"""
            SELECT time, token, price
            FROM {table}
        """

        params = []

        if symbol or start_date or end_date:
            query += " WHERE"

        if symbol is not None:
            query += " token = %s"
            params.append(symbol)

        # Фильтры по времени
        if start_date is not None:
            if symbol:
                query += " AND time >= %s"
            else:
                query += " time >= %s"
            params.append(start_date)

        if end_date is not None:
            if symbol or start_date:
                query += " AND time <= %s"
            else:
                query += " time <= %s"
            params.append(end_date)

        query += " ORDER BY time;"

        with self.conn.cursor() as cur:
            cur.execute(query, tuple(params))
            columns = [desc[0] for desc in cur.description]
            data = cur.fetchall()

        return pl.DataFrame(data, schema=columns, orient="row")

    def get_oldest_date_in_orderbook(self, token):
        query = """
            SELECT MIN("time") AS oldest_time
            FROM tick_ob
            WHERE token = %s;
        """

        with self.conn.cursor() as cur:
            # return pl.read_database(query, cur)
            cur.execute(query, [token])
            res = cur.fetchone()

        return res[0]

    def get_tick_ob(self, token=None, start_time=None, end_time=None):
        query = """
            SELECT token, time,
                   bid_price, bid_size, ask_price, ask_size
            FROM tick_ob
        """

        if token or start_time or end_time:
            query += " WHERE"

        params = []

        if token is not None:
            query += " token = %s"
            params.append(token)

        if start_time is not None:
            if token:
                query += " AND time >= %s"
            else:
                query += " time >= %s"
            params.append(start_time)

        if end_time is not None:
            if token or start_time:
                query += " AND time <= %s"
            else:
                query += " time <= %s"
            params.append(end_time)

        query += " ORDER BY time;"

        with self.conn.cursor() as cur:
            # return pl.read_database(query, cur)
            cur.execute(query, params)
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]

        return pl.DataFrame(rows, schema=colnames, orient="row")

    def clear_old_data(self, table, column, expiration_time, units):
        """
        Удаляет из таблицы 'table' все данные, которые старше 'expiration_time',
        измеренных в 'units' по столбцу 'column'.
        :param table: название таблицы, данные из которой необходимо удалить
        :param column: название столбца, содержащего временные значения
        :param expiration_time: время в часах
        :param units: единицы измерения времени: ('hour', 'hours', 'seconds', 'minutes')
        """
        assert units in ('hour', 'hours', 'seconds', 'minutes'), "units should be in ('hour', 'hours', 'seconds', 'minutes')"
        assert column in ('time', 'timestamp', 'time'), "column should be in ('time', 'time', 'timestamp')"

        query = f"DELETE FROM {table} WHERE {column} < NOW() - INTERVAL '{expiration_time} {units}';"
        with self.conn.cursor() as cur:
            cur.execute(query)

    def get_table(self, table_name, df_type='pandas'):
        """
        Получает все данные из заданной таблицы и возвращает их как pandas или polars DataFrame.
        """
        query = f"SELECT * FROM {table_name};"
        try:
            with self.conn.cursor() as cur:
                cur.execute(query)
                # Получение данных и названий столбцов
                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description]
                # Преобразование в DataFrame
                if df_type == 'pandas':
                    df = pd.DataFrame(rows, columns=columns)
                    if 'id' in df.columns:
                        df = df.set_index('id')
                elif df_type == 'polars':
                    df = pl.DataFrame(rows, schema=columns, orient="row")
                return df
        except Exception as e:
            self.conn.rollback()
            raise ValueError(f"Failed to fetch data from table '{table_name}': {e}")

    def get_columns(self, table_name):
        """Получить список столбцов таблицы"""
        query = """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position;
        """
        with self.conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(query, (table_name,))
            columns = cursor.fetchall()
            return columns

    def get_table_info(self, table_name: str, schema: str = "public") -> dict:
        """Возвращает метаданные о таблице: колонки, типы, ограничения, индексы."""
        with self.conn.cursor() as cur:
            # 1. Колонки и типы
            cur.execute("""
                SELECT
                    column_name,
                    data_type,
                    is_nullable,
                    column_default
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position;
            """, (schema, table_name))
            columns = cur.fetchall()

            # 2. Первичный ключ
            cur.execute("""
                SELECT
                    kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                WHERE tc.table_schema = %s
                  AND tc.table_name = %s
                  AND tc.constraint_type = 'PRIMARY KEY';
            """, (schema, table_name))
            pk = [row[0] for row in cur.fetchall()]

            # 3. Внешние ключи
            cur.execute("""
                SELECT
                    kcu.column_name,
                    ccu.table_name AS foreign_table,
                    ccu.column_name AS foreign_column
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage ccu
                    ON ccu.constraint_name = tc.constraint_name
                    AND ccu.table_schema = tc.table_schema
                WHERE tc.table_schema = %s
                  AND tc.table_name = %s
                  AND tc.constraint_type = 'FOREIGN KEY';
            """, (schema, table_name))
            fks = cur.fetchall()

            # 4. Индексы
            cur.execute("""
                SELECT
                    indexname,
                    indexdef
                FROM pg_indexes
                WHERE schemaname = %s AND tablename = %s;
            """, (schema, table_name))
            indexes = cur.fetchall()

        # Формируем результат
        return {
            "table": table_name,
            "schema": schema,
            "columns": [
                {
                    "name": c[0],
                    "type": c[1],
                    "nullable": c[2],
                    "default": c[3]
                }
                for c in columns
            ],
            "primary_key": pk,
            "foreign_keys": [
                {"column": fk[0], "ref_table": fk[1], "ref_column": fk[2]}
                for fk in fks
            ],
            "indexes": [
                {"name": idx[0], "definition": idx[1]}
                for idx in indexes
            ]
        }

    def clear_table(self, table_name):
        """Полностью очищает указанную таблицу"""
        query = f"TRUNCATE TABLE {table_name} RESTART IDENTITY CASCADE"
        try:
            with self.conn.cursor() as cur:
                cur.execute(query)
                self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise ValueError(f"Failed to truncate table '{table_name}': {e}")
