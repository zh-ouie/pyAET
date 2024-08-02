from decimal import Decimal, ROUND_HALF_UP

def my_round_list(input_x, decimals='1'):
    """
    Round integers in a list. Because np.round(32.5) gives 32.

    Args:
        input_x (float): The input value.
        decimals(str): The decimals to keep. '1' for int, '0.01' for keep two decimals.

    Returns:
        float: The rounded integer.
    """
    rounded_x = input_x.copy().astype(int)
    for i, xi in enumerate(input_x):
        x = Decimal(str(xi))
        rounded_x_i = int(x.quantize(Decimal(decimals), rounding=ROUND_HALF_UP))
        rounded_x[i] = rounded_x_i
    return rounded_x

def my_round_num(input_x, decimals='1'):
    """
    Round integers. Because np.round(32.5) gives 32.

    Args:
        input_x (float): The input value.
        decimals(str): The decimals to keep. '1' for int, '0.01' for keep two decimals.

    Returns:
        float: The rounded integer.
    """
    x = Decimal(str(input_x))
    rounded_x = int(x.quantize(Decimal(decimals), rounding=ROUND_HALF_UP))
    return rounded_x