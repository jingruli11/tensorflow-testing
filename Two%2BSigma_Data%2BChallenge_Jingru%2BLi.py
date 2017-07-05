
# coding: utf-8

# # Two Sigma Data Challenge_MTA Dataset_Jingru Li

# ##  Data Extraction and Cleansing

# In[1]:

# import libraries
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit
import datetime


# In[2]:

# get a list of data file names
ls_c = glob.glob('/Users/Michael/Downloads/python/tensorflow-testing/two sigma current/*.txt')
ls_p = glob.glob('/Users/Michael/Downloads/python/tensorflow-testing/two sigma pre/*.txt')


# In[3]:

# load current datasets into a pandas dataframe
start = timeit.default_timer()

df_current = pd.read_csv('two sigma current/turnstile_170624.txt', sep = ',')
ls_c = ls_c[:-1]

for file in ls_c:
    df_current = df_current.append(pd.read_csv(file, sep = ','))

stop = timeit.default_timer()
print(stop - start)


# In[4]:

# define a function to flatten the file shape
def flat(df):
    col_1 = ['C/A','UNIT','SCP','DATE1','TIME1','DESC1','ENTRIES1','EXITS1']
    col_2 = ['C/A','UNIT','SCP','DATE2','TIME2','DESC2','ENTRIES2','EXITS2']
    col_3 = ['C/A','UNIT','SCP','DATE3','TIME3','DESC3','ENTRIES3','EXITS3']
    col_4 = ['C/A','UNIT','SCP','DATE4','TIME4','DESC4','ENTRIES4','EXITS4']
    col_5 = ['C/A','UNIT','SCP','DATE5','TIME5','DESC5','ENTRIES5','EXITS5']
    col_6 = ['C/A','UNIT','SCP','DATE6','TIME6','DESC6','ENTRIES6','EXITS6']
    col_7 = ['C/A','UNIT','SCP','DATE7','TIME7','DESC7','ENTRIES7','EXITS7']
    col_8 = ['C/A','UNIT','SCP','DATE8','TIME8','DESC8','ENTRIES8','EXITS8']
    colnew = ['C/A','UNIT','SCP','DATE','TIME','DESC','ENTRIES','EXITS']
    cols = [col_1, col_2, col_3, col_4, col_5, col_6, col_7, col_8]
    temp = []
    for i in range(8):
        df_temp = df[cols[i]]
        df_temp = df_temp.rename(columns = {df_temp.columns[3]: 'DATE', 
                                            df_temp.columns[4]: 'TIME',
                                            df_temp.columns[5]: 'DESC', 
                                            df_temp.columns[6]: 'ENTRIES',
                                            df_temp.columns[7]: 'EXITS'})
        temp.append(df_temp)
    df_trans = pd.concat(temp)
    df_trans.dropna(inplace = True)
    return df_trans


# In[5]:

# load pre 10/18/2014 datasets into a pandas dataframe
start = timeit.default_timer()
# ls_p = glob.glob('/Users/Michael/Downloads/python/tensorflow-testing/two sigma pre/*.txt')
# since the pre datasets does not have column names, the column names will be manually 
# embedded into the dataframe
col_nms = ['C/A','UNIT','SCP','DATE1','TIME1','DESC1','ENTRIES1','EXITS1',
           'DATE2','TIME2','DESC2','ENTRIES2','EXITS2',
           'DATE3','TIME3','DESC3','ENTRIES3','EXITS3',
           'DATE4','TIME4','DESC4','ENTRIES4','EXITS4',
           'DATE5','TIME5','DESC5','ENTRIES5','EXITS5',
           'DATE6','TIME6','DESC6','ENTRIES6','EXITS6',
           'DATE7','TIME7','DESC7','ENTRIES7','EXITS7',
           'DATE8','TIME8','DESC8','ENTRIES8','EXITS8']

# since file 'turnstile_120714.txt' (114) and file 'turnstile_120505.txt' (104) are messier 
# than other files, these two files will be cleansed and loaded seperately.

df_prea = pd.read_csv(ls_p[-1], sep = ',', names = col_nms)
df_prea = flat(df_prea)
ls_p = ls_p[:-1]
del ls_p[114]
del ls_p[104]
n = int(len(ls_p)/2)

for file in ls_p[:n]:
    df_tempa = pd.read_csv(file, sep = ',', names = col_nms)
    df_tempa = flat(df_tempa)
    df_prea = df_prea.append(df_tempa)

df_preb = pd.read_csv(ls_p[n], sep = ',', names = col_nms)
df_preb = flat(df_preb)    
for file in ls_p[n+1:]:
    df_tempb = pd.read_csv(file, sep = ',', names = col_nms)
    df_tempb = flat(df_tempb)
    df_preb = df_preb.append(df_tempb)

stop = timeit.default_timer()
print(stop - start)
print(len(ls_p))
print(df_prea.shape)
print(df_preb.shape)


# In[6]:

# load and flatten file 'turnstile_120714.txt' (114)
f_114 = ls_p[114]

df_114 = pd.read_csv(f_114, sep = ',', names = col_nms, skiprows = 10)
df_114_f = flat(df_114)


# In[7]:

# load and flatten file 'turnstile_120505.txt' (104)
f_104 = ls_p[104]
df_104a = pd.read_csv(f_104, sep = ',', names = col_nms, nrows = 16384)
df_104b = pd.read_csv(f_104, sep = ',', names = col_nms, skiprows = 16384)
df_104 = df_104a.append(df_104b)
df_104_f = flat(df_104)


# In[8]:

# append all 4 dataframes together to form a large dataframe that holds all data pre
# Since there are 6 different types of description, only those labeled as 'REGULAR' will be
# considered without further information
# concat date and time together and change the datatype to datetime
start = timeit.default_timer()
df_pre = df_prea.append(df_preb).append(df_104_f).append(df_114_f)
df_pre = df_pre[df_pre['DESC'] == 'REGULAR']
datetime = df_pre.DATE + ' ' + df_pre.TIME
df_pre['DATETIME'] = pd.to_datetime(datetime, format='%m-%d-%y %H:%M:%S')
df_pre.head()
stop = timeit.default_timer()
print(stop - start)
print(df_pre.shape)


# In[9]:

# load the excel file that shows the relationship among units, stations, lines, etc.
df_rel = pd.read_excel('Remote-Booth-Station.xls')
df_rel['CONDITION'] = df_rel['Booth'] + '-' + df_rel['Remote']

print(df_rel.shape)
print(df_rel.head())


# In[10]:

df_2013 = df_pre[df_pre['DATE'].str[-2:] == '13']

df_2013['TURNSTILE'] = df_2013['C/A'] + '-' + df_2013['UNIT'] + '-' + df_2013['SCP']
df_2013['CONDITION'] = df_2013['C/A'] + '-' + df_2013['UNIT']
df_2013= pd.merge(df_2013, df_rel[['CONDITION','Station']],left_on = 'CONDITION', right_on = 'CONDITION', how = 'left')


# In[11]:

df_2013.sort_values(['Station', 'TURNSTILE', 'DATETIME'])
# define a function to calculate ENTRIES_MARGINAL and EXITS_MARGINAL

def marginal_change(i):
    if len(i) != 2:
        return np.nan
    last, now = i
    # if last > now then the odometer has been reset. The value will be discarded.
    return now - last if last <= now else now
marginal_change_rolling = lambda ser: ser.rolling(window=2).apply(marginal_change)
# compute marginal change of entries and exits using rolling window
df_grouped = df_2013.groupby(['TURNSTILE','Station'])
df_2013['EXITS_MARGINAL']   = df_grouped['EXITS'].apply(marginal_change_rolling)
df_2013['ENTRIES_MARGINAL'] = df_grouped['ENTRIES'].apply(marginal_change_rolling)
# for some periods the stations are closed. They will be marked as NaN
for col in ['ENTRIES_MARGINAL', 'EXITS_MARGINAL']:
    df_2013.ix[df_2013[col] < 0, col] = np.nan


# In[12]:

df_2013.head(20)


# In[14]:

plt.figure()
sample = 1000000
hist = { 'bins':700, 'range':(0,7000), 'figsize':(20, 8) }
df_sample = df_2013.sample(n=sample).loc[:,['EXITS_MARGINAL','ENTRIES_MARGINAL']]
df_sample.hist(**hist)
plt.show()


# In[15]:

# clearly counts beyond 6000 decreased sharply. Thus 6000 will be used to filter outliers.
df_2013 = df_2013[(df_2013['ENTRIES_MARGINAL'] <= 6000) & (df_2013['EXITS_MARGINAL'] <= 6000)]
df_2013[['EXITS_MARGINAL','ENTRIES_MARGINAL']].describe()


# In[16]:

df_2013['BUSY_MARGINAL'] = df_2013['ENTRIES_MARGINAL'] + df_2013['EXITS_MARGINAL']


# Notes on data extraction and cleansing:
# 1. The problem may not require the full dataset to be loaded, however I always like to challenge myself loading and cleansing large dataset with limited RAM capacity. 
# 2. The current dataset is extremely clean. However the pre dataset has 2 files that messes up the whole process. Thus I used bisection search to find these two files and dealt with them accordingly.
# 3. Since the RAM capacity in my laptop is limited, the pre dataset is too huge to load at once. So I broke it down to two parts. Combining the two messy parts, I concatenated all four subsets to form a complete dataframe for the data pre 10/18/2014.
# 4. The station information is joined into the dataframe based on UNIT and C/A variables.
# 5. The turnstile is defined using C/A, UNIT and SCP.
# 6. The future questions only focus on year 2013.
# 7. The marginal change of entries and exits as well as business are computed for future use.

# *************************

# ## Data Analysis:

# Question 1: Which station has the most number of units as of today?

# In[17]:

most_units_station = df_rel[['Station', 'Remote']].groupby(['Station']).agg('count').sort_values('Remote', ascending = False).head(1)
print(most_units_station.index[0], ' has the most number of units (', most_units_station.values[0][0], ' units) as of today.')


# 34 ST-PENN STA has the most number of units (14 units) as of today.

# ***************************************************************************************

# Question 2: What is the total number of entries & exits across the subway system for August 1, 2013?

# In[18]:

from datetime import datetime
date_condition = (df_2013.DATETIME >= datetime(2013,8,1)) & (df_2013.DATETIME < datetime(2013,8,2))
df_2013_0801 = df_2013[date_condition]


# In[19]:

entries_080113 = np.sum(df_2013_0801['ENTRIES_MARGINAL'])
exits_080113 = np.sum(df_2013_0801['EXITS_MARGINAL'])
    
print('The total number of entries on August 01, 2013 is: ', entries_080113)
print('The total number of exits on August 01, 2013 is: ', exits_080113)


# In[20]:

df_2013_0801.head()


# ******

# Question 3:  Let’s define the busy-ness as sum of entry & exit count. What station was the busiest on August 1, 2013? What turnstile was the busiest on that date?

# In[21]:

df_station_080113 = df_2013_0801.groupby(['Station']).agg(np.sum)
busy_station = df_station_080113.sort_values('BUSY_MARGINAL',ascending = False).head(1).index[0]
df_turnstile_080113 = df_2013_0801.groupby(['TURNSTILE']).agg(np.sum)
busy_turnstile = df_turnstile_080113.sort_values('BUSY_MARGINAL',ascending = False).head(1).index[0]
print('The busiest station on Aug 01, 2013 is: ', busy_station)
print('The busiest turnstile on Aug 01, 2013 is: ', busy_turnstile)


# ***

# Question 4: What stations have seen the most usage growth/decline in 2013?

# In[22]:

df_2013['MONTH'] = df_2013['DATETIME'].dt.month
df_2013_monthly = df_2013.groupby(['Station','MONTH']).sum().reset_index()
df_2013_monthly.head()


# In[23]:

df_2013_monthly['BUSY_PERCENTAGE'] = df_2013_monthly.groupby('Station').pct_change().BUSY_MARGINAL
df_2013_monthly_growth = df_2013_monthly.groupby('Station').agg(np.mean)
df_2013_monthly.head()


# In[24]:

# The top 5 and last 5 stations of average monthly percentage change of business in 2013
df_2013_monthly_growth.dropna(inplace=True)
df_2013_monthly_growth.sort_values('BUSY_PERCENTAGE', ascending = False, inplace = True)
print('Top 5 growth usage stations:')
print(df_2013_monthly_growth.head(5).loc[:, ['BUSY_PERCENTAGE']])
df_2013_monthly_growth.sort_values('BUSY_PERCENTAGE', ascending = True, inplace = True)
print('\n','Top 5 decline usage stations:')
print(df_2013_monthly_growth.head(5).loc[:, ['BUSY_PERCENTAGE']])


# ***

# Question 5: What dates in 2013 are the least busy? Could you identify days in 2013 on which stations were not operating at full capacity or closed entirely?

# In[25]:

from datetime import timedelta
df_2013['DAY'] = df_2013['DATETIME'].dt.dayofyear
df_2013_daily = df_2013.groupby(['DAY']).agg(np.sum)
df_2013_daily.sort_values('BUSY_MARGINAL', ascending = True, inplace = True)
print('The least busy dates in 2013 are:')
df_2013_daily.head().loc[:,['BUSY_MARGINAL']]


# In[26]:

# identify stations in specific days that were not operating at full capacity or closed entirely
# basically entries with NaN values.
df_2013_station_daily = df_2013.groupby(['Station', 'DAY']).agg(np.sum)
df_2013_station_daily.head(20)


# In[27]:

df_2013_closed_daily = df_2013_station_daily[df_2013_station_daily['BUSY_MARGINAL'] == 0]
print('The stations closed on specific dates are: ')

df_2013_closed_daily['BUSY_MARGINAL']


# In[28]:

df_2013_station_daily['BUSY_MARGINAL'].describe()


# In[29]:

# assume stations without operating full capacity has less than 100 in 'BUSY_MARGINAL' column.
print('Stations not operating full capacity:')
df_2013_daily_nfull = df_2013_station_daily[(df_2013_station_daily['BUSY_MARGINAL'] < 100) & (df_2013_station_daily['BUSY_MARGINAL'] != 0)]['BUSY_MARGINAL']
df_2013_daily_nfull


# ***

# ## Data Visualization 

# Question 1: Plot the daily row counts for data files in Q3 2013

# In[30]:

q3_range = (df_2013.DATETIME >= datetime(2013,7,1)) & (df_2013.DATETIME < datetime(2013,10,1))
df_2013_q3 = df_2013[q3_range]
df_2013_q3_counts = df_2013_q3.groupby(['DATE']).size()
plt.figure()
df_2013_q3_counts.plot(kind = 'area', figsize = (20,8), title = 'Daily Row Counts in Q3 2013')
plt.show()


# ***

# Question 2: Plot the daily total number of entries & exits across the system for Q3 2013

# In[31]:

df_2013_q3_daily_ee = df_2013_q3[['DATE','ENTRIES_MARGINAL','EXITS_MARGINAL']].groupby(['DATE']).agg(np.sum)
plt.figure()
df_2013_q3_daily_ee.plot(kind = 'line', figsize = (20,8), title = 'Daily total number of entries & exits across the system for Q3 2013')
plt.show()


# ***

# Question 3:  Plot the mean and standard deviation of the daily total number of entries & exits for each month in Q3 2013 for station 34 ST-PENN STA

# In[32]:

df_2013_q3_penn = df_2013_q3[df_2013_q3['Station'] == '34 ST-PENN STA']
df_2013_q3_penn = df_2013_q3_penn[['MONTH', 'ENTRIES_MARGINAL','EXITS_MARGINAL']]
df_2013_q3_penn_mean_std = df_2013_q3_penn[['MONTH','ENTRIES_MARGINAL', 'EXITS_MARGINAL']].groupby(['MONTH']).agg([np.mean, np.std])
plt.figure()
df_2013_q3_penn_mean_std.plot(kind = 'bar', figsize = (20,8), title = 'Mean and standard deviation of daily total number of entries & exits for each month in Q3 2013 for station 34 ST-PENN STA')
plt.show()


# ***

# Question 4: Plot 25/50/75 percentile of the daily total number of entries & exits for each month in Q3 2013 for station 34 ST-PENN STA

# In[33]:

df_2013_q3_penn_percentile = df_2013_q3_penn[['MONTH','ENTRIES_MARGINAL', 'EXITS_MARGINAL']].groupby(['MONTH']).quantile([0.25, 0.5, 0.75])
plt.figure()
df_2013_q3_penn_percentile.loc[7].plot(kind = 'box', figsize = (20,8), title = 'July quantiles of daily total number of entries & exits for each month in Q3 2013 for station 34 ST-PENN STA')
df_2013_q3_penn_percentile.loc[8].plot(kind = 'box', figsize = (20,8), title = 'August quantiles of daily total number of entries & exits for each month in Q3 2013 for station 34 ST-PENN STA')
df_2013_q3_penn_percentile.loc[9].plot(kind = 'box', figsize = (20,8), title = 'September quantiles of daily total number of entries & exits for each month in Q3 2013 for station 34 ST-PENN STA')
plt.show()


# ***

# Question 5:  Plot the daily number of closed stations and number of stations that were not operating at full capacity in Q3 2013

# In[34]:

df_2013_closed_count = df_2013_closed_daily['BUSY_MARGINAL'].groupby('DAY').count().rename('CLOSED')
df_2013_nfull_count = df_2013_daily_nfull.groupby('DAY').count().rename('NOT_FULL_CAPACITY')
df_2013_closed_nfull = pd.concat([df_2013_closed_count, df_2013_nfull_count], axis = 1)
df_2013_closed_nfull['DAYS'] = df_2013_closed_nfull.index


# In[35]:

temp = df_2013_q3[['DAY', 'DATE']].groupby(['DAY','DATE']).count()
temp['DATE2'] = temp.index.levels[1]
temp.index = temp.index.droplevel(1)

df_2013_closed_nfull = temp.merge(df_2013_closed_nfull, left_index = True, right_index = True, how = 'left')
df_2013_closed_nfull = df_2013_closed_nfull[['DATE2', 'CLOSED', 'NOT_FULL_CAPACITY']]
df_2013_closed_nfull.set_index('DATE2', inplace = True)
df_2013_closed_nfull


# In[36]:

plt.figure()
df_2013_closed_nfull.plot(kind = 'area', figsize = (20,8), title = 'Daily number of closed stations and number of stations that were not operating at full capacity in Q3 2013')
plt.show()

