import numpy as np
import pandas as pd

def clean_small_query(df):
    df=df.drop_duplicates(keep='first')
    query_wrong_100 = list(np.load('query_wrong_100_VD.npy'))
    wrong = df[df.query_id.isin(query_wrong_100)]
    df=df.append(wrong)
    df=df.drop_duplicates(keep=False)
    return df

def clean_small_query_150(df):
    df=df.drop_duplicates(keep='first')
    query_wrong_150 = list(np.load('query_wrong_150_VD.npy'))
    wrong = df[df.query_id.isin(query_wrong_150)]
    df=df.append(wrong)
    df=df.drop_duplicates(keep=False)
    return df
  
def clean_small_query_200(df):
    df=df.drop_duplicates(keep='first')
    query_wrong_200 = list(np.load('query_wrong_200_VD.npy'))
    wrong = df[df.query_id.isin(query_wrong_200)]
    df=df.append(wrong)
    df=df.drop_duplicates(keep=False)
    return df
  
def clean_face_reference(df):
    df = df.copy()
    df=df.drop_duplicates(keep='first')
    ls = list(df['query_id'])
    names = [l.split('_')[0] for l in ls]
    df['query_id'] = names
    ls = list(df['reference_id'])
    names = [l.split('_')[0] for l in ls]
    df['reference_id'] = names
    face_del = list(np.load('face_del.npy'))
    wrong = df[df.reference_id.isin(face_del)]
    df=df.append(wrong)
    df=df.drop_duplicates(keep=False)
    return df

  
V5_baseline_CC_234 = pd.read_csv('./V5_baseline_CC/V5-baseline-CC-234-50k-VD.csv')
V5_baseline_CC_234=clean_small_query_200(V5_baseline_CC_234)

R_baseline_CC_234 = pd.read_csv('./V5_baseline_CC_ref/R-baseline-CC-234-50k.csv')#82.8
R_baseline_CC_234=clean_face_reference(R_baseline_CC_234)#83.748

V5_blur_CC_234 = pd.read_csv('./V5_blur_CC/V5-blur-CC-234-50k-VD.csv')#84.4
V5_blur_CC_234=clean_small_query(V5_blur_CC_234)

V5_face_CC_234 = pd.read_csv('./V5_face_CC/V5-face-CC-234-50k-VD.csv')#85.2
V5_face_CC_234=clean_small_query(V5_face_CC_234)

V5_color_CC_234 = pd.read_csv('V5_color_CC/V5-color-CC-234-50k-VD.csv')#86.4
V5_color_CC_234=clean_small_query(V5_color_CC_234)

V5_Dark_CC_234 = pd.read_csv('V5_dark_CC/V5-dark-CC-234-50k-VD.csv')#86.5
V5_Dark_CC_234=clean_small_query(V5_Dark_CC_234)

V5_baseline_BW_234 = pd.read_csv('V5_baseline_BW/V5-baseline-BW-234-50k-VD.csv')#87.1
V5_baseline_BW_234=clean_small_query_200(V5_baseline_BW_234)

V5_U_CC_234 = pd.read_csv('V5_u_CC/V5-u-CC-234-50k-VD.csv')#87.2
V5_U_CC_234=clean_small_query_200(V5_U_CC_234)

V5_color_BW_234 = pd.read_csv('V5_color_BW/V5-color-BW-234-50k-VD.csv')#87.4
V5_color_BW_234=clean_small_query_200(V5_color_BW_234)

V5_blur_BW_234 = pd.read_csv('V5_blur_BW/V5-blur-BW-234-50k-VD.csv')#87.5
V5_blur_BW_234 = clean_small_query(V5_blur_BW_234)

V5_face_BW_234 = pd.read_csv('V5_face_BW/V5-face-BW-234-50k-VD.csv')#87.615
V5_face_BW_234 = clean_small_query_150(V5_face_BW_234)

V5_opa_CC_234 = pd.read_csv('V5_opa_CC/V5-opa-CC-234-50k-VD.csv')#87.2
V5_opa_CC_234=clean_small_query_200(V5_opa_CC_234)

step_11 = pd.concat((V5_baseline_CC_234,R_baseline_CC_234,V5_blur_CC_234,V5_face_CC_234,V5_color_CC_234,
                    V5_Dark_CC_234,V5_baseline_BW_234, V5_U_CC_234,V5_color_BW_234, V5_blur_BW_234, V5_face_BW_234,
                    V5_opa_CC_234))


q = ['Q%05d_0'%i for i in range(50000, 100000)] 
for j in range(1000,1020):
    q = q + ['Q%05d_%d'%(i,j) for i in range(50000, 100000)]
print("This is designed for Phase 2: the numbers of querys are from 50000 to 99999!!!")
not_in = step_11[~step_11.query_id.isin(q)]
score_new = np.array(list(not_in['score']))
score_new = score_new-0.05
not_in['score'] = score_new

step_12 = pd.concat((not_in,step_11[step_11.query_id.isin(q)]))
step_12.to_csv('step_11_50k.csv',index=False)







