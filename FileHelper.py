import xlwt


def save_merging_results_one_space_division(filenames: [], results: [], result_filename: str = 'results//Results.xls',
                                            sheetname: str = 'Result'):
    """Saves results of merging algorithm for one space division and one number of base classifier

    :param filenames: names of files being analysed
    :param results: resulting matrix
    :param result_filename: filename to write results to
    :param sheetname: sheetname to write results to
    :return:
    """
    workbook = xlwt.Workbook()
    workbook.add_sheet(sheetname)
    sheet = workbook.get_sheet(sheetname)
    sheet.write(0, 0, "filename")
    sheet.write(0, 1, "majority voting score")
    sheet.write(0, 2, "integrated classifier score")
    sheet.write(0, 3, "majority voting matthews correlation coefficient")
    sheet.write(0, 4, "integrated classifier matthews correlation coefficient")
    for i in range(len(filenames)):
        sheet.write(i + 1, 0, filenames[i])
        for j in range(len(results[i])):
            sheet.write(i + 1, j + 1, results[i][j])
    workbook.save(result_filename)


def save_merging_results(filenames: [], results_pro_space_division: [], space_division: [],
                         result_filename: str = 'results//Results.xls', sheetname: str = 'Result'):
    """Saves results of merging algorithm for one number of base classifier

    :param filenames: names of files being analysed
    :param results_pro_space_division: matrix of results pro space division
    :param space_division: matrix of space divisions
    :param result_filename: filename to write results to
    :param sheetname: sheetname to write results to
    :return:
    """
    workbook = xlwt.Workbook()
    workbook.add_sheet(sheetname)
    sheet = workbook.get_sheet(sheetname)
    sheet.write(0, 0, "subspaces")
    sheet.write(1, 0, "filename")
    for i in range(len(space_division)):
        sheet.write(0, 4 * i + 1, str(space_division[i]))
        sheet.write(1, 4 * i + 1, "mv_s")
        sheet.write(1, 4 * i + 2, "i_s")
        sheet.write(1, 4 * i + 3, "mv_mcc")
        sheet.write(1, 4 * (i + 1), "i_mcc")
    for i in range(len(filenames)):
        sheet.write(i + 2, 0, filenames[i])
        for j in range(len(space_division)):
            for k in range(len(results_pro_space_division[j][i])):
                sheet.write(i + 2, 4 * j + k + 1, results_pro_space_division[j][i][k])
    workbook.save(result_filename)
