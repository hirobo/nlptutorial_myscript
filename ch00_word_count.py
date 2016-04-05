# coding: utf-8
from collections import defaultdict

counts = defaultdict(lambda: 0)

file_path = "../test/00-input.txt"
file = open(file_path, "r")

for line in file:
    words = line.replace("\t", " ").strip().split(" ")
    for w in words:
        counts[w] += 1

for key in counts:
    print key, counts[key]