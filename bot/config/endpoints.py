WS_ENDPOINTS = {
    "bybit": {
        "private": {
            "prod": "wss://stream.bybit.com/v5/private",
            "demo": "wss://stream-demo.bybit.com/v5/private"
        },
        "spot": {
            "prod": "wss://stream.bybit.com/v5/public/spot"
        },
        "linear": {
            "prod": "wss://stream.bybit.com/v5/public/linear",
            "demo": "wss://stream.bybit.com/v5/public/linear"
        },
        "trade": {
            "prod": "wss://stream.bybit.com/v5/trade"
        }
    },
    "okx": {
        "private": {
            "prod": "wss://ws.okx.com:8443/ws/v5/private",
            "demo": "wss://wspap.okx.com:8443/ws/v5/private"
        },
        "linear": {
            "prod": "wss://ws.okx.com:8443/ws/v5/public",
            "demo": "wss://ws.okx.com:8443/ws/v5/public"
        },
    },
    "gate": {
        "private": {
            "prod": "wss://fx-ws.gateio.ws/v4/ws/usdt",
            "demo": "wss://fx-ws-testnet.gateio.ws/v4/ws/usdt"
        },
        "linear": {
            "prod": "wss://fx-ws.gateio.ws/v4/ws/usdt",
            "demo": "wss://fx-ws.gateio.ws/v4/ws/usdt"
        },
    }
}