import pandas as pd

df = pd.read_csv('final_50k.csv')

ls = list(df['query_id'])
names = [l.split('_')[0] for l in ls]
df['query_id'] = names

ls = list(df['reference_id'])
names = [l.split('_')[0] for l in ls]
df['reference_id'] = names

df_clean =df.groupby(['query_id','reference_id']).mean()

df_clean.to_csv('final_50k_agg.csv',index=True)
