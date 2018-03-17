import xlwt


def save_merging_results(filenames: [], results: []):
    workbook = xlwt.Workbook()
    workbook.add_sheet("Result")
    sheet = workbook.get_sheet("Result")
    sheet.write(0, 0, "filename")
    sheet.write(0, 1, "majority voting score")
    sheet.write(0, 2, "integrated classifier score")
    sheet.write(0, 3, "majority voting matthews correlation coefficient")
    sheet.write(0, 4, "integrated classifier matthews correlation coefficient")
    for i in range(len(filenames)):
        sheet.write(i + 1, 0, filenames[i])
        for j in range(len(results[i])):
            sheet.write(i + 1, j + 1, results[i][j])
    workbook.save("Results.xlsx")
