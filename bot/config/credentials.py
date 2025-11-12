from dotenv import load_dotenv
import os

BASEDIR = os.path.abspath(os.path.dirname('bot'))
load_dotenv(os.path.join(BASEDIR, '.env'))

BYBIT_DEMO_API_KEY = os.environ['BYBIT_DEMO_API_KEY']
BYBIT_API_KEY = os.environ['BYBIT_API_KEY']
BYBIT_DEMO_SECRET_KEY = os.environ['BYBIT_DEMO_SECRET_KEY']
BYBIT_SECRET_KEY = os.environ['BYBIT_SECRET_KEY']

BD_PASSWORD = os.environ['BD_PASSWORD']
coin_market_cap_api = os.environ['coin_market_cap_api']

host = "127.0.0.1"
user = "postgres"
password = BD_PASSWORD
db_name = "market"
