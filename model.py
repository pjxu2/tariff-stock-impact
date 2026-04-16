import torch
import pandas as pd
import numpy as np
import csv

with open('tariff_news_headlines.csv', newline='') as headlines:
    reader = csv.reader(headlines)
    for row in reader:
        