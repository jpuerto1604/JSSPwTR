import pandas as pd
import numpy as np
import os

df1 = pd.DataFrame(np.array([[0,6,8,10,12],[12,0,6,8,10],[10,6,0,6,8],[8,8,6,0,6],[6,10,8,6,0]]),columns=["LU","M1","M2",'M3',"M4"],index=["LU","M1","M2",'M3',"M4"])
df2 = pd.DataFrame(np.array([[0,4,6,8,6],[6,0,2,4,2],[8,12,0,2,4],[6,10,12,0,2],[4,8,10,12,0]]),columns=["LU","M1","M2",'M3',"M4"],index=["LU","M1","M2",'M3',"M4"])
df3 = pd.DataFrame(np.array([[0,2,4,10,12],[12,0,2,8,10],[10,12,0,6,8],[4,6,8,0,2],[2,4,6,12,0]]),columns=["LU","M1","M2",'M3',"M4"],index=["LU","M1","M2",'M3',"M4"])
df4 = pd.DataFrame(np.array([[0,4,8,10,14],[18,0,4,6,10],[20,14,0,8,6],[12,8,6,0,6],[14,14,12,6,0]]),columns=["LU","M1","M2",'M3',"M4"],index=["LU","M1","M2",'M3',"M4"])

def t_times(layout,start,end):
    match layout:
        case 1:
            return df1.loc[start,end]
        case 2:
            return df2.loc[start,end]
        case 3:
            return df3.loc[start,end]
        case 4:
            return df4.loc[start,end]

xls = pd.read_excel('/Users/julian/Documentos/Thesis/Data.xlsx',sheet_name='Macrodata', usecols='F:H')
