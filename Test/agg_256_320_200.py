import pandas as pd

_256 = pd.read_csv('./final/step_11_50k.csv')
_320 = pd.read_csv('./final_320/step_11_50k.csv')
_200 = pd.read_csv('./final_200/step_11_50k.csv')

final = pd.concat((_256,_320,_200))
final.to_csv("256_320_200_50k.csv",index=False)
