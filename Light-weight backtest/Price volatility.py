#%%
# This is to label 5 min/ 15mins variation larger than cetain percent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
#%%
df = pd.read_csv('E:\Quant\Bitcoin autotrade//15min.csv',usecols=range(0,6),\
   names=['time','open','high','low','close','volume'], header=None) 
df['time'] = pd.to_datetime(df['time'],unit='ms')
#df=df.set_index(0)
df['return'] = np.log(df.close) - np.log(df.close.shift(1))
#df['weighted price']=df[4]*df[5]
#df['1h_return']=df[4].pct_change(4)
df=df.dropna()
df.to_csv('E:/Quant/Learn backtrader/data/15min.csv',index=None)
#%%
# devide into different 
df_list=np.array_split(df,30)
#%% Plot ACF
#plot_pacf(df['return'], lags=30, method="ywm")
plot_acf(df['return'], lags=30)
plt.ylim(-0.2, 0.2)
plt.show()
#%%
# calculate ema
# regression to get significant parameter
# different emv range regression
def reg_ewm(df=df,col='return',time_span=20):
    X = df[col].ewm(span=time_span,min_periods=0,adjust=False,ignore_na=False).mean()
    y = df[col]
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    res={'pvalue':est2.pvalues[col],'aic':est2.aic,'beta':est2.params[col]}
    return res

#%%
# loop to get result
def get_res_list(ran=range(10,20),fun=reg_ewm,args=(df,'return')):
    res_list=[]
    for i in ran:
        para=args+(i,)
        res=fun(*para)
        res_list.append(res)
    result=pd.DataFrame(res_list)
    return result
#%%
# different lag terms regression
def reg_lag_rt(df=df,col='return',time_span=1):
    df['lag'] = df[col].shift(time_span)
    df1=df.dropna()
    X = df1['lag']
    y = df1[col]
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    res={'pvalue':est2.pvalues['lag'],'aic':est2.aic,'beta':est2.params['lag']}
    return res

d=get_res_list(ran=range(1,100),fun=reg_lag_rt,args=(df1,'return'))
#%%
def reg_lag_vol(df=df,col=5,time_span=1):
    df1=df.dropna()
    X = df1[col].shift(time_span)
    y = df1['return']
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2,missing='drop')
    est2 = est.fit()
    res={'pvalue':est2.pvalues[col],'aic':est2.aic,'beta':est2.params[col]}
    return res

d=get_res_list(ran=range(1,10),fun=reg_lag_vol,args=(df,5))
#%%
# Get rolling volumn weighted variance
def reg_var(df=df,col='return',time_span=10):
    #target_col='weighted price'
    # price variance
    target_col=4
    X = df[target_col].rolling(time_span).var()
    y = df[col]
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2,missing='drop')
    est2 = est.fit()
    print(est2)
    res={'pvalue':est2.pvalues[target_col],'aic':est2.aic,'beta':est2.params[target_col]}
    return res
#%%
d=get_res_list(ran=range(1,10),fun=reg_var)

#%%
X = df[5]
y = df['cumsum']
X2 = sm.add_constant(X)
est = sm.OLS(y, X2,missing='drop')
est2 = est.fit()
print(est2.summary())

