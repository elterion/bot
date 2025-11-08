from decimal import Decimal, ROUND_DOWN, getcontext
getcontext().prec = 8

def get_step_info(coin_information, token, long_exc, short_exc):
    try:
        qty_long_step = coin_information[long_exc][token]['qty_step']
    except KeyError:
        raise Exception(f'Не могу найти токен "{token}" для биржи {long_exc}')

    try:
        qty_short_step = coin_information[short_exc][token]['qty_step']
    except KeyError:
        raise Exception(f'Не могу найти токен "{token}" для биржи {short_exc}')

    return max(qty_long_step, qty_short_step)

def get_min_qty(coin_information, token, long_exc, short_exc):
    try:
        qty_long_step = coin_information[long_exc][token]['min_qty']
    except KeyError:
        raise Exception(f'Не могу найти токен "{token}" для биржи {long_exc}')

    try:
        qty_short_step = coin_information[short_exc][token]['min_qty']
    except KeyError:
        raise Exception(f'Не могу найти токен "{token}" для биржи {short_exc}')

    return max(qty_long_step, qty_short_step)

def get_price_scale(coin_information, token, exc_full):
    try:
        price_scale = coin_information[exc_full][token]['price_scale']
        return 1 / 10**price_scale
    except KeyError:
        raise Exception(f'Не могу найти токен "{token}" для биржи {exc_full}')


def round_volume(volume: Decimal, qty_step: Decimal) -> Decimal:
    """
    Округляет значение volume в зависимости от qty_step:
      - Если qty_step == 1: округляем volume до целого (в меньшую сторону).
      - Если qty_step < 1: округляем volume до количества знаков после запятой,
        которое соответствует количеству знаков в qty_step.
      - Если qty_step > 1: округляем volume до ближайшего меньшего числа, кратного qty_step.
    """
    if not isinstance(volume, Decimal):
        volume = Decimal(volume)
    if not isinstance(qty_step, Decimal):
        qty_step = Decimal(qty_step)

    if qty_step == Decimal("1"):
        return Decimal(int(volume))
    elif qty_step < Decimal("1"):
        decimals = -qty_step.as_tuple().exponent
        quantizer = Decimal("1").scaleb(-decimals)
        return volume.quantize(quantizer, rounding=ROUND_DOWN)
    else:
        return (volume // qty_step) * qty_step
