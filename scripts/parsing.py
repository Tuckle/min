import os
import csv
import datetime

# head = "","Time","Tp","Cl","pH","Redox","Leit","Trueb","Cl_2","Fm","Fm_2","EVENT"
csv_path = r"data"

def read_data(path):
    def _convert(row):       
        return int(row[0]), datetime.datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S"), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), int(row[9]), int(row[10]), eval(row[11].capitalize())
    with open(path) as f:
        c = csv.reader(f, delimiter=",")
        for line in c:
            if line[0] != "" and 'NA' not in line:
            	line = _convert(line)
            	yield line

data = list(read_data(csv_path))
