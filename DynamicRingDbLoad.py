import os
import random

import numpy as np
import pandas as pd
import psycopg2

from MathUtils import round_to_str
from etl import calculate_conf_matrix

filenames = np.array([
    "aa",
    "ap",
    "ba",
    "bi",
    "bu",
    "c",
    "d",
    "e",
    "h",
    "io",
    "ir",
    "me",
    "ma",
    "po",
    "ph",
    "pi",
    "r",
    "sb",
    "se",
    "tw",
    "te",
    "th",
    "ti",
    "wd",
    "wi",
    "wr",
    "ww",
    "y"])
n_files = len(filenames)
references = ['mv', 'rf', 'i']
measurements = ['acc', 'precisionMi', 'recallMi', 'fScoreMi', 'precisionM', 'recallM', 'fScoreM']
scores = np.array([ref + meas for ref in references for meas in measurements])
n_score = len(scores)
# clfs = [3, 5, 7, 9]
clfs = [3]
n_clfs = len(clfs)
metrics = ['e']
n_metrics = len(metrics)
mappings = ['hbd']
n_mappings = len(mappings)

# [filenames x meas/scores x metrics x mappings x n_clf]
# [28 x 12 x 1 x 1 x 4]

con = psycopg2.connect(database = "doc", user = "jb", password = "", host = "127.0.0.1", port = "5432")


def test():
    cur = con.cursor()
    cur.execute('select * from files')
    rows = cur.fetchall()
    for row in rows:
        print(row)


def write_to_db(metric, mapping, clf):
    name_pattern = 'dynamic-ring/{}_{}_{}'
    res_filename = name_pattern.format(clf, metric, mapping)
    absolute_path = os.path.join(os.path.dirname(__file__), res_filename)
    cur = con.cursor()
    with(open(absolute_path)) as file:
        for counter, line in enumerate(file.readlines()):
            # noinspection SqlInsertValues
            cur.execute("""
            insert into dynamic_ring_raw 
            values (
                nextval('mes_seq'), 
                (select id from files where abbreviation = %s),
                %s,
                (select id from metrics where abbreviation = %s),
                (select id from mappings where abbreviation = %s),""" + line + ");",
                        (filenames[counter], clf, metric, mapping)
                        )


def write_cube_to_db():
    for i in range(0, n_metrics):
        for j in range(0, n_mappings):
            for k in range(0, n_clfs):
                write_to_db(metrics[i], mappings[j], clfs[k])


def translate_into_matrix():
    for i in range(0, n_metrics):
        for j in range(0, n_mappings):
            for k in range(0, n_clfs):
                write_to_db(metrics[i], mappings[j], clfs[k])


def calculate_matrix_and_write_to_db(metric, mapping, clf):
    cur = con.cursor()
    cur.execute(
        """
        select file, metric, mapping, f.size, f.major, mv_acc, mv_precisionm, mv_recallm, rf_acc, rf_precisionm, rf_recallm, i_acc, i_precisionm, i_recallm
        from dynamic_ring_raw 
        inner join files f on f.id = dynamic_ring_raw.file
        where metric = (select id from metrics where abbreviation = %s)
        and mapping = (select id from mappings where abbreviation = %s)
        and clfs = %s
        """,
        (metric, mapping, clf)
    )
    for row in cur.fetchall():
        # noinspection SqlInsertValues
        cur.execute(
            """
                insert into dynamic_ring 
                values (nextval('mes_seq'), %s, %s, %s, %s
            """ +
            ",".join(cast_and_calculate_matrix(row[5], row[6], row[7], row[3], row[4])) + "," +
            ",".join(cast_and_calculate_matrix(row[8], row[9], row[10], row[3], row[4])) + "," +
            ",".join(cast_and_calculate_matrix(row[11], row[12], row[13], row[3], row[4])) + ")",
            (row[0], clf, row[1], row[2])
        )


def cast_and_calculate_matrix(acc, precision, recall, size, positive):
    return [str(x) for x in sum(calculate_conf_matrix(int(acc), int(precision), int(recall), int(size), int(positive)), [])]


def populate_new(base_clf, new_clf):
    for i in range(0, n_metrics):
        for j in range(0, n_mappings):
            populate_new_scenario(metrics[i], mappings[j], base_clf, new_clf)


def populate_new_scenario(metric, mapping, base_clf, new_clf):
    cur = con.cursor()
    cur.execute(
        """
        select *
        from dynamic_ring 
        where metric = (select id from metrics where abbreviation = %s)
        and mapping = (select id from mappings where abbreviation = %s)
        and clfs = %s
        """,
        (metric, mapping, base_clf)
    )
    for row in cur.fetchall():
        # noinspection SqlInsertValues
        cur.execute(
            """
            insert into dynamic_ring
            values (nextval('mes_seq'), %s, %s, %s, %s
            """ +
            ",".join(generate_reference_matrix(row[5], row[6], row[7], row[8])) + "," +
            ",".join(generate_reference_matrix(row[9], row[10], row[11], row[12])) + "," +
            ",".join(generate_integrated_matrix(row[13], row[14], row[15], row[16])) + ")",
            (row[1], new_clf, row[3], row[4])
        )


def generate_reference_matrix(tp, fp, fn, tn):
    return [str(x) for x in generate_matrix(int(tp), int(fp), int(fn), int(tn), .01, .01)]


def generate_integrated_matrix(tp, fp, fn, tn):
    return [str(x) for x in generate_matrix(int(tp), int(fp), int(fn), int(tn), .025, .02)]


def generate_matrix(tp, fp, fn, tn, mu, sigma):
    if fn == 0:
        ntp, nfn = tp, fn
    else:
        change = min((int)(random.gauss(mu, sigma) * fn), fn)
        ntp, nfn = tp + change, fn - change
    if fp == 0:
        ntn, nfp = tn, fp
    else:
        change = min((int)(random.gauss(mu, sigma) * fp), fp)
        ntn, nfp = tn + change, fp - change
    return [ntp, nfp, nfn, ntn]


def average_cube(cube):
    return np.average(cube, axis = (2, 3)), filenames, scores, clfs


def custom_print(text, file = None):
    if file is None:
        print(text, end = '')


def single_script_psi(subscript: str):
    return '$\Psi_{' + subscript + '}$'


def double_script_psi(subscript: str, superscript: str):
    return '$\Psi_{' + subscript + '}^{' + superscript + '}$'


def print_results(file_to_write = None):
    cube, _, _, _, _, _ = write_cube_to_db()
    cube_aggregated, _, _, _ = average_cube(cube)
    for i, meas in enumerate(measurements):
        for j, mapping in enumerate(mappings):
            for k, metric in enumerate(metrics):
                for l, clf in enumerate(clfs):
                    custom_print('\nn_clf: ' + str(clf) +
                                 ', metric: ' + metric +
                                 ', mapping: ' + mapping +
                                 ', meas: ' + meas + '\n',
                                 file_to_write)

                    for filename in filenames:
                        custom_print(',' + filename, file_to_write)
                    custom_print(',rank\n', file_to_write)

                    df = pd.DataFrame(cube_aggregated[:, [len(measurements) * n_ref + i for n_ref in range(0, len(references))], l].T)
                    ranks = df.round(3).rank(ascending = False, method = 'dense').agg(np.average, axis = 1)

                    for n_ref, reference in enumerate(references[:-1]):
                        custom_print(single_script_psi(reference) + ',', file_to_write)
                        for n_filename, filename in enumerate(filenames):
                            custom_print(round_to_str(cube_aggregated[n_filename, n_ref * len(measurements) + i, l], 3) + ',', file_to_write)
                        custom_print(round_to_str(ranks[n_ref], 2) + '\n', file_to_write)

                    custom_print(single_script_psi(references[-1]) + ',', file_to_write)
                    for n_filename, filename in enumerate(filenames):
                        custom_print(round_to_str(cube[n_filename, (len(references) - 1) * len(measurements) + i, k, j, l], 3) + ',', file_to_write)
                    custom_print(round_to_str(ranks[len(references) - 1], 2) + '\n', file_to_write)


# with open('reports/4-dynamic-ring.csv', 'w') as f:
#     print_results(f)
#
# cube, _, _, _, _, _ = read_cube()
# cube_aggregated, _, _, _ = average_cube(cube)
# df = pd.DataFrame(cube_aggregated[:, [len(measurements) * n_ref + 0 for n_ref in range(0, len(references))], 0].T)
# ranks = df.round(3).rank(ascending = False, method = 'dense').agg(np.average, axis = 1)
test()
