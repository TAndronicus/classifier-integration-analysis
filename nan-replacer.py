import os

catalog_name = "dtd-displacement"
str_from = "NaN"
str_to = "0"

for file in os.listdir(catalog_name):
    fin = open(catalog_name + "/" + file, "rt")
    data = fin.read()
    data = data.replace(str_from, str_to)
    fin.close()
    fout = open(catalog_name + "/" + file, "wt")
    fout.write(data)
    fout.close()
