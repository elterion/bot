import os
import re

def load_tokens_from_file(file_name):
    """
    Считывает список токенов из текстового файла.

    :param file_name: Имя файла с токенами.
    :return: Список токенов.
    """
    try:
        with open(file_name, "r", encoding="utf-8") as file:
            tokens = file.read().splitlines()
        return tokens
    except FileNotFoundError:
        print(f"Файл {file_name} не найден.")
        return []

def get_saved_coins(data_folder):
    pattern = re.compile(r'^([^_]+)_agg_trades\.parquet$', re.IGNORECASE)
    coins = set()

    for filename in os.listdir(data_folder):
        file_path = os.path.join(data_folder, filename)
        
        # Проверяем, что это файл и соответствует шаблону
        if os.path.isfile(file_path):
            match = pattern.match(filename)
            if match:
                coin_name = match.group(1)
                coins.add(coin_name.upper())  # Для единообразия приводим к верхнему регистру

    return sorted(coins)