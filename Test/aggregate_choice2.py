import pandas as pd

df_1 = pd.read_csv('submit_50k.csv')
df_2 = pd.read_csv('final/V5_baseline_CC_pair/V5-baseline-CC-234-50k-VD-pair.csv')
df = pd.concat((df_1,df_2))

ls = list(df['query_id'])
names = [l.split('_')[0] for l in ls]
df['query_id'] = names

ls = list(df['reference_id'])
names = [l.split('_')[0] for l in ls]
df['reference_id'] = names

df_clean =df.groupby(['query_id','reference_id']).max()

df_clean.to_csv('submit_50k_choice2.csv',index=True)
