import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



if __name__ == "__main__":
    df = pd.read_csv('./bench_results/fastmst_segment/bench_results/oberhofen_7680_4320.csv', index_col=0)
    print(df.head())
