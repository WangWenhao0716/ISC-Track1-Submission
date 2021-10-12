import pandas as pd
all_ = pd.read_csv('final_50k_agg.csv')
all_ = all_.sort_values(by='score',ascending=False)

all_ = all_.iloc[:50_0000]
all_.to_csv('submit_50k.csv',index=False)
