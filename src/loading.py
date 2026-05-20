import numpy as np
import pandas as pd

df = pd.read_csv("data\Housing.csv")


df.head(1)

df.tail(10)
df.info()
df.shape