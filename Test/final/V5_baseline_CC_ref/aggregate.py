import pandas as pd

v_4 = pd.read_csv('50/predictions_dev_queries_50k_normalized_exp.csv')
v_5 = pd.read_csv('ibn/predictions_dev_queries_50k_normalized_exp.csv')
v_6 = pd.read_csv('152/predictions_dev_queries_50k_normalized_exp.csv')

v_4_query = list(v_4['query_id'])
v_4_reference = list(v_4['reference_id'])
v_4_com = []
for i in range(len(v_4)):
    v_4_com.append((v_4_query[i],v_4_reference[i]))
    
v_5_query = list(v_5['query_id'])
v_5_reference = list(v_5['reference_id'])
v_5_com = []
for i in range(len(v_5)):
    v_5_com.append((v_5_query[i],v_5_reference[i]))


v_6_query = list(v_6['query_id'])
v_6_reference = list(v_6['reference_id'])
v_6_com = []
for i in range(len(v_6)):
    v_6_com.append((v_6_query[i],v_6_reference[i]))

inter_45 = list(set(v_4_com).intersection(set(v_5_com)))
inter_46 = list(set(v_4_com).intersection(set(v_6_com)))
inter_456 = list(set(inter_45).intersection(set(inter_46)))

new_456 = pd.DataFrame()
q = []
for i in range(len(inter_456)):
    q.append(inter_456[i][0])
r = []
for i in range(len(inter_456)):
    r.append(inter_456[i][1])
new_456['query_id'] = q
new_456['reference_id'] = r
df_2 = pd.merge(new_456, v_4, on=['query_id','reference_id'], how='inner')
df_3 = pd.merge(new_456, v_5, on=['query_id','reference_id'], how='inner')
df_4 = pd.merge(new_456, v_6, on=['query_id','reference_id'], how='inner')
fast_456 = pd.concat((df_2,df_3,df_4))

fast_456.to_csv('R-baseline-CC-234-50k.csv',index=False)
