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
data.loc[:,'nj'] = data.loc[:,'nj']+1
data = data.fillna('')
p_times =pd.DataFrame(data.iloc[:,:10].to_numpy(),columns=["Set","Job","nj","P1","P2","P3","P4","P5","P6","P7"])
m_data = pd.DataFrame(data.iloc[:, [0, 1, 2] + list(range(10, data.shape[1]))].to_numpy(), columns=["Set", "Job", "nj"] + [f"M{i}" for i in range(1, data.shape[1] - 9)])

#Function to retrieve transport times with strings
def t_times(layout, start, end):
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
    return m_data[m_data['Set'] == nset].iloc[:, 1:].reset_index(drop=True)

#Retrieve the data from the processing times based on a specific set
def processing(nset):
    return p_times[p_times['Set'] == nset].iloc[:, 1:].reset_index(drop=True)

#Function to calculate the total processing time of a job in a specific set
def total_processing(nset, job):
    times = processing(nset)
    return times[times['Job'] == job].iloc[0, 3:].sum()

# Function to calculate the total transport time of a job in a specific set
def transport_time(layout, nset, job):
    # Retrieve the sequence of jobs
    jobs_data = jobs(nset)
    # Retrieve the job data
    job_data = jobs_data[jobs_data['Job'] == job].iloc[0, 2:].to_numpy()
    # Count the number of operations
    nops = np.count_nonzero(job_data)
    # Initialize the total transport time
    total = 0
    # Iterate over the sequence of jobs to calculate the total transport time
    try:
        for i in range(nops - 1):
            total += t_times(layout, job_data[i], job_data[i + 1])
        return total
    except:
        return 0

# Function to calculate the total transport time for all jobs in a specific set
def total_transport(layout, nset):
    # Retrieve the sequence of jobs
    jobs_data = jobs(nset)
    # Initialize the total transport time
    total = 0
    # Iterate over all jobs to calculate the total transport time
    for i in range(jobs_data.shape[0]):
        total += transport_time(layout, nset, jobs_data.iloc[i, 0])
    return total

#Serialize all jobs from a set in a single array with an ID
def serialization(nset):
    jobs_data = jobs(nset)
    jobs_data = jobs_data[jobs_data.columns[2:]].to_numpy()
    serialized_jobs = []
    for job_id, job in enumerate(jobs_data, start=1):
        for pos, machine in enumerate(job):
            if machine != '':
                serialized_jobs.append((job_id, pos + 1, machine))
    jobs_data = np.array(serialized_jobs)
    jobs_data = pd.DataFrame(jobs_data,columns=["Job","Position","Machine"])
    return jobs_data