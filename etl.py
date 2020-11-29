from math import pow, sqrt

from texttable import Texttable


def calculate_conf_matrix(acc, precision, recall, size, positive = 0):
    if precision == 0 or recall == 0: return [[0, round(size * (1 - acc) - positive)], [positive, round(acc * size)]]
    if precision == 1 and recall == 1: return [[positive, 0], [0, size - positive]]
    tp = round(size * (1 - acc) / (1 / precision + 1 / recall - 2))
    return [
        [
            tp,
            round(size * (1 - acc) * (1 - precision) / (precision * (1 / precision + 1 / recall - 2)))
        ],
        [
            round(size * (1 - acc) * (1 / recall - 1) / (1 / precision + 1 / recall - 2)),
            round(size * (acc - (1 - acc) / (1 / precision + 1 / recall - 2)))
        ]
    ]


def calculate_conf_matrix_binary(acc, mcc, size, positive):
    if mcc == 0 and round(acc * size) == positive: return one_row_matrix(size, positive)
    ap = pow(mcc, 2) * size * positive - pow(mcc, 2) * pow(positive, 2) + pow(size, 2) / 4 - size * positive + pow(positive, 2)
    bp = acc * pow(size, 3) / 2 - acc * pow(size, 2) * positive - pow(mcc, 2) * pow(size, 2) * positive + pow(mcc, 2) * size * pow(positive, 2) - pow(size, 3) / 2 + 3 * pow(
        size, 2) * positive / 2 - size * pow(positive, 2)
    cp = pow(acc, 2) * pow(size, 4) / 4 - acc * pow(size, 4) / 2 + acc * pow(size, 3) * positive / 2 + pow(size, 4) / 4 - pow(size, 3) * positive / 2 + pow(size, 2) * pow(
        positive, 2) / 4
    print(f'ap: {ap}, bp: {bp}, cp: {cp}')
    deltasq = sqrt(pow(bp, 2) - 4 * ap * cp)
    x1, x2 = (-bp + deltasq) / (2 * ap), (-bp - deltasq) / (2 * ap)
    tp1, tp2 = (size * (acc - 1) + positive + x1) / 2, (size * (acc - 1) + positive + x2) / 2
    fp1, fn1, tn1 = round(x1 - tp1), round(positive - tp1), round(acc * size - tp1)
    fp2, fn2, tn2 = round(x2 - tp2), round(positive - tp2), round(acc * size - tp2)
    if (abs(calculate_mcc(tp1, fp1, fn1, tn1) - mcc) > abs(calculate_mcc(tp2, fp2, fn2, tn2) - mcc)) and (x2 > 0):
        tp, x = tp2, x2
    else:
        tp, x = tp1, x1
    return [
        [
            round(tp),
            round(x - tp)
        ],
        [
            round(positive - tp),
            round(acc * size - tp)
        ]
    ]


def print_conf_matrix(matrix, verbose = False):
    t = Texttable()
    t.set_deco(0b1101)
    if verbose:
        t.add_rows([['', '', 'Actual class', ''], ['', '', 'P', 'N'], ['Predicted class', 'P', matrix[0][0], matrix[0][1]], ['', 'N', matrix[1][0], matrix[1][1]]])
    else:
        t.add_rows([[matrix[0][0], matrix[0][1]], [matrix[1][0], matrix[1][1]]])
    print(t.draw())


def one_row_matrix(size, positive):
    return [
        [positive, size - positive],
        [0, 0]
    ]


def calculate_mcc(tp, fp, fn, tn):
    denom = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0
    else:
        return (tp * tn - fp * fn) / denom
