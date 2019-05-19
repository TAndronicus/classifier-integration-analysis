def round_to(number: float, resolution: int):
    if resolution < 0: raise Exception('Resolution negative')
    inter = pow(10, resolution) * number
    if inter - int(inter) < .5:
        rest = 0
    else:
        rest = 1
    return (int(inter) + rest) / pow(10, resolution)


def round_to_str(number: float, resolution: int):
    rounded = str(round_to(number, resolution))
    parts = rounded.split('.')
    if len(parts) == 1:
        return rounded + '.' + resolution * '0'
    else:
        return rounded + (resolution - len(parts[1])) * '0'
