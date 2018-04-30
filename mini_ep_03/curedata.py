#!/bin/python3
import statistics
from matplotlib import pyplot as plt

vals = [[],[],[],[],[],[],[]]
for i in (1, 10, 100):
    for j in (10000, 100000, 1000000):
        file = open('{0}{1}.txt'.format(j, i), 'r')
        summ = 0
        values = []
        print(file.name)
        for line in file:
            print(line, end="")
            line = line.rstrip()
            values.append(float(line))
        print(statistics.mean(values))
    
        res_file = open("wow", 'w')
