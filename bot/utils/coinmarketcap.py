import requests
from bot.config.credentials import coin_market_cap_api

def get_top_tokens(api_key=coin_market_cap_api,
                   limit=300, output_file=None):
    """
    Получает топ {limit} токенов с CoinMarketCap и сохраняет их в текстовый файл.

    Args:
        api_key (str): API-ключ для CoinMarketCap.
        limit (int): Количество токенов для загрузки.
        output_file (str, optional): Имя файла для сохранения списка токенов.
    """
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": api_key,
    }
    params = {
        "start": "1",  # Начать с первой криптовалюты
        "limit": limit,  # Количество записей
        "convert": "USD",  # Валюта для цены
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        # Извлекаем тикеры токенов
        tickers = [crypto["symbol"] for crypto in data["data"]]

        # Сохраняем тикеры в файл
        if output_file:
            with open(output_file, "w", encoding="utf-8") as file:
                file.write("\n".join(tickers))
            print(f"Список тикеров токенов успешно сохранён в файл: {output_file}")

        return tickers
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе: {e}")
        return []
