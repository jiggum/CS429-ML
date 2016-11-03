import requests
import re

def get_dataset(url,parse_func):
    source = requests.get(url)
    data = []
    if(source):
        data = parse_func(source.text)
    return data

def parse_1(source_txt):
    data = []
    data_pre = source_txt.strip().split('\n')
    data = [[float(element) for element in re.split('\s.{0,2}:',line)] for line in data_pre]
    return data

def parse_2(source_txt):
    data = []
    data_pre = source_txt.strip().split('\n')
    data = [[element for element in re.split('\s',line.strip())] for line in data_pre]
    for line in data:
        for i in range(len(line)):
            line[i] = int(line[i].replace(":1",""))
    c_data = []
    for line in data:
        c_line = []
        for i in range(130):
            c_line.append(0)
        c_line[0] = line[0]
        for i in line:
            c_line[i] = 1;
        c_data.append(c_line)
    return c_data