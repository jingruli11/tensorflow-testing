import pandas as pd 
import numpy as np 

df = pd.read_csv('turnstile_170624.txt', sep = ',')
df = df.append(pd.read_csv('turnstile_170617.txt', sep = ','))
df = df.append(pd.read_csv('turnstile_170610.txt', sep = ','))
df = df.append(pd.read_csv('turnstile_170603.txt', sep = ','))
df = df.append(pd.read_csv('turnstile_170527.txt', sep = ','))
df = df.append(pd.read_csv('turnstile_170520.txt', sep = ','))
df = df.append(pd.read_csv('turnstile_170506.txt', sep = ','))
df = df.append(pd.read_csv('turnstile_170401.txt', sep = ','))
df = df.append(pd.read_csv('turnstile_170408.txt', sep = ','))
df = df.append(pd.read_csv('turnstile_170415.txt', sep = ','))
df = df.append(pd.read_csv('turnstile_170422.txt', sep = ','))
df = df.append(pd.read_csv('turnstile_170429.txt', sep = ','))
df = df.append(pd.read_csv('turnstile_170325.txt', sep = ','))

print(len(df))

