import pandas as pd

df = pd.read_csv('256_320_200_50k.csv')

ls = list(df['query_id'])
names = [l.split('_')[0] for l in ls]
df['query_id'] = names

ls = list(df['reference_id'])
names = [l.split('_')[0] for l in ls]
df['reference_id'] = names

df_clean =df.groupby(['query_id','reference_id']).max()

df_clean.to_csv('256_320_200_50k_agg.csv',index=True)
