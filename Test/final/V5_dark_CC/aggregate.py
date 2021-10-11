import pandas as pd

v_5 = pd.read_csv('50/predictions_dev_queries_50k_normalized_exp_VD.csv')
v_6 = pd.read_csv('152/predictions_dev_queries_50k_normalized_exp_VD.csv')
v_7 = pd.read_csv('ibn/predictions_dev_queries_50k_normalized_exp_VD.csv')

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
    
v_7_query = list(v_7['query_id'])
v_7_reference = list(v_7['reference_id'])
v_7_com = []
for i in range(len(v_7)):
    v_7_com.append((v_7_query[i],v_7_reference[i]))
    
inter_56 = list(set(v_5_com).intersection(set(v_6_com)))
inter_57 = list(set(v_5_com).intersection(set(v_7_com)))
inter_67 = list(set(v_6_com).intersection(set(v_7_com)))

new_56 = pd.DataFrame()
q = []
for i in range(len(inter_56)):
    q.append(inter_56[i][0])
r = []
for i in range(len(inter_56)):
    r.append(inter_56[i][1])
new_56['query_id'] = q
new_56['reference_id'] = r
df_1 = pd.merge(new_56, v_5, on=['query_id','reference_id'], how='inner')
df_2 = pd.merge(new_56, v_6, on=['query_id','reference_id'], how='inner')
fast_56 = pd.concat((df_1,df_2))

new_57 = pd.DataFrame()
q = []
for i in range(len(inter_57)):
    q.append(inter_57[i][0])
r = []
for i in range(len(inter_57)):
    r.append(inter_57[i][1])
new_57['query_id'] = q
new_57['reference_id'] = r
df_1 = pd.merge(new_57, v_5, on=['query_id','reference_id'], how='inner')
df_2 = pd.merge(new_57, v_7, on=['query_id','reference_id'], how='inner')
fast_57 = pd.concat((df_1,df_2))

new_67 = pd.DataFrame()
q = []
for i in range(len(inter_67)):
    q.append(inter_67[i][0])
r = []
for i in range(len(inter_67)):
    r.append(inter_67[i][1])
new_67['query_id'] = q
new_67['reference_id'] = r
df_1 = pd.merge(new_67, v_6, on=['query_id','reference_id'], how='inner')
df_2 = pd.merge(new_67, v_7, on=['query_id','reference_id'], how='inner')
fast_67 = pd.concat((df_1,df_2))

fast = pd.concat((fast_56,fast_57,fast_67))
fast.to_csv('V5-dark-CC-234-50k-VD.csv',index=False)