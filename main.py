import pandas as pd
import numpy as np

#Loading of transport layout times
df1 = pd.DataFrame(np.array([[0,6,8,10,12],[12,0,6,8,10],[10,6,0,6,8],[8,8,6,0,6],[6,10,8,6,0]]),columns=["LU","M1","M2",'M3',"M4"],index=["LU","M1","M2",'M3',"M4"])
df2 = pd.DataFrame(np.array([[0,4,6,8,6],[6,0,2,4,2],[8,12,0,2,4],[6,10,12,0,2],[4,8,10,12,0]]),columns=["LU","M1","M2",'M3',"M4"],index=["LU","M1","M2",'M3',"M4"])
df3 = pd.DataFrame(np.array([[0,2,4,10,12],[12,0,2,8,10],[10,12,0,6,8],[4,6,8,0,2],[2,4,6,12,0]]),columns=["LU","M1","M2",'M3',"M4"],index=["LU","M1","M2",'M3',"M4"])
df4 = pd.DataFrame(np.array([[0,4,8,10,14],[18,0,4,6,10],[20,14,0,8,6],[12,8,6,0,6],[14,14,12,6,0]]),columns=["LU","M1","M2",'M3',"M4"],index=["LU","M1","M2",'M3',"M4"])

#Loading of processing times and machines data
xls = pd.read_excel('/Users/julian/Documentos/Thesis/Data.xlsx', sheet_name='Macrodata', usecols='F:H, J:P, R:X')
data=pd.DataFrame(xls)
data = data.fillna('')
p_times =pd.DataFrame(data.iloc[:,:10].to_numpy(),columns=["Set","Job","nj","P1","P2","P3","P4","P5","P6","P7"])
m_data = pd.DataFrame(data.iloc[:, [0, 1, 2] + list(range(10, data.shape[1]))].to_numpy(), columns=["Set", "Job", "nj"] + [f"M{i}" for i in range(1, data.shape[1] - 9)])

#Function to retrieve transport times with strings
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

#Retrieve the data from the sequence to perform based on a specific set
def jobs(nset):
    return m_data[m_data['Set'] == nset]

#Retrieve the data from the processing times based on a specific set
def processing(nset):
    return p_times[p_times['Set'] == nset]