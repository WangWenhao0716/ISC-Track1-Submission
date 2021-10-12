import pandas as pd
df = pd.read_csv('256_320_200_50k.csv')
df_max = pd.read_csv('256_320_200_50k_agg.csv')

df = df.drop_duplicates(keep='first', inplace=False)

q = list(df['query_id'])
r = list(df['reference_id'])
q_new = []
for i in range(len(q)):
    if('_' in q[i]):
        q_new.append(q[i].split('_')[0])
    else:
        q_new.append(q[i])
r_new = []
for i in range(len(r)):
    if('_' in r[i]):
        r_new.append(r[i].split('_')[0])
    else:
        r_new.append(r[i])

df['query_id'] = list(q_new)
df['reference_id'] = list(r_new)

set_diff_df = pd.concat([df, df_max, df_max]).drop_duplicates(keep=False)
set_diff_df_max = set_diff_df.groupby(['query_id','reference_id'],as_index=False).max()

final = pd.concat((set_diff_df_max,df_max))

final.to_csv('final_50k.csv',index=False)
