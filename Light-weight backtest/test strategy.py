# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 09:42:53 2021

@author: haoch
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import numpy as np
import multiprocess as mp
from itertools import repeat
import warnings 
import math
warnings.filterwarnings('ignore')
#%%
#df['return']=df[4].pct_change(1)
# get buy and sell signal and return dataframe
def get_crossover(df,slow_span=24,fast_span=10):
    df['macd_fast']=df.iloc[:,4].ewm(span=fast_span,min_periods=fast_span,adjust=False,ignore_na=True).mean()
    df['macd_slow']=df.iloc[:,4].ewm(span=slow_span,min_periods=slow_span,adjust=False,ignore_na=True).mean()
    df['flag']=df.apply(lambda x:1 if x['macd_fast']>x['macd_slow'] else -1,axis=1)
    df1=df.dropna()
    return df1

#get total return after transaction fee
# get total number of transactions
def get_tans_return(df1):
    df1['next_1'] = df1['flag'].shift(1)
    df1=df1.dropna()
    num=len(df1[df1['next_1']!=df1['flag']])
    sm= (math.exp(df1[df1['flag']==1]['return'].sum()) -1 )
         #- (math.exp(df1[df1['flag']==-1]['return'].sum()) -1 )
    sm=sm-num*0.002
    return sm, num


# assign each transaction with a unique ID
def flag_transaction(df):
    df=df.reset_index(drop=True)
    ID=0
    df['tradeID']=0
    for current in range(1,len(df.index)):
        previous = current -1
        if df['flag'][current]==df['flag'][previous]:
                df['tradeID'][current]=ID
        else:
            df['tradeID'][current]=ID
            ID+=1
            if current==len(df.index):
                continue
            else:
                df['tradeID'][current+1]=ID
    return df

# get max/min cummulative return for each trade
def analyze_each_ID(df):
    df=df.groupby('tradeID').apply(lambda group: group.iloc[:-1]).reset_index(drop=True)
    df=df[df['tradeID']>0]
    df['cumsum'] = df.groupby('tradeID')['return'].transform(pd.Series.cumsum)
    # long : mean, min, max and accumulative return for each transaction
    df1=df[df['flag']==1]
    result=df1.groupby('tradeID').agg({'cumsum': ['median', 'min', 'max'],\
                                     'return':lambda x: math.exp(x.sum())-1.002})
    # short : 
    # df2=df[df['flag']==-1]
    # df2['cumsum']=-df2['cumsum']
    # df2['return']=-df2['return']
    # short=df2.groupby('tradeID').agg({'cumsum': ['median', 'min', 'max'],\
    #                                   'return':lambda x: math.exp(x.sum())-1.002})
    # result=pd.concat([long,short])

    minimum=math.exp(result['cumsum']['min'].dropna().min())-1-0.002
    max_loss=math.exp(result.iloc[:,3].dropna().min())-1-0.002
    max_profit=math.exp(result.iloc[:,3].dropna().max())-1-0.002
    median_profit=math.exp(result.iloc[:,3].dropna().median())-1-0.002
    return60 = math.exp(result.iloc[:,3].dropna().quantile(0.6))-1-0.002
    
    unit={"extreme_loss":minimum,"max_loss":max_loss,\
            "max_profit":max_profit,"median_profit":median_profit,\
           "return60":return60}
    return unit

#%%
# find the optimal time span for macd strategy
def level1_strategy(df):
    test_list=[]
    for fast in range(1,100):
        for slow in range (fast,100):
            df1=get_crossover(df,slow,fast)
            df1=flag_transaction(df1)
            res_unit=analyze_each_ID(df1)
            test_list.append(res_unit)  
            print(res_unit)
    res_df=pd.DataFrame(test_list)
    return res_df
#%%
# input result from level1 strategy
def level2_strategy(df,slow,fast):
    res_list=[]
    for i in range(1,100):
        # get sample
        start=random.randint(1000, len(df)-1000)
        end=start+1000
        sample=df[start:end].reset_index(drop=True)

        sample=get_crossover(sample,slow,fast)
        sample=flag_transaction(sample)
        res_unit=analyze_each_ID(sample)
        res_list.append(res_unit)   
    res_df=pd.DataFrame(res_list)
    return res_df
#%%
# result got from level1 strategy fun
# get top return strategies
#result=result.sort_values(by=['return'],ascending=False)
#top=result.head(100).reset_index(drop=True)
#top=pd.read_csv('E:/Quant/Learn backtrader/data/top.csv')
#%%
def analyse_top(top,df):
    top = top.reset_index(drop=True)
    analysis_list=[]
    for ind, row in top.iterrows():
        slow=row['slow']
        fast=row['fast']
        res=level2_strategy(df,slow,fast)
        
        extreme_loss_med=res['extreme_loss'].dropna().median()
        extreme_loss_max=res['extreme_loss'].dropna().min()
        max_loss_med=res['max_loss'].dropna().median()
        max_loss_max=res['max_loss'].dropna().min()
        max_profit_med=res['max_profit'].dropna().median()
        max_profit_max=res['max_profit'].dropna().max()
        median_profit_med=res['median_profit'].dropna().median()
        median_profit_max=res['median_profit'].dropna().max()
        tot_return_med=res['tot_return'].dropna().median()

        analysis_unit={'slow':slow,'fast':fast,
                       "tot_return_1y":row['return'],"tot_trade_no":row['trade_no'],\
                       'extreme_loss_med':extreme_loss_med,'extreme_loss_max':extreme_loss_max,
                       'max_loss_med':max_loss_med,'max_loss_max':max_loss_max,
                       'max_profit_med':max_profit_med,'max_profit_max':max_profit_max,
                       'median_profit_med':median_profit_med,'median_profit_max':median_profit_max,
                       'tot_return_med':tot_return_med}
                
        analysis_list.append(analysis_unit)
    return pd.DataFrame(analysis_list)
#%%
def plot_res(df1):
    fig = plt.figure()
    ax1 = fig.add_subplot(4,1,1)
    ax2 = fig.add_subplot(4,1,2)
    ax3 = fig.add_subplot(4,1,3)
    #ax4 = fig.add_subplot(4,1,3)
    #sns.displot(result['return']['sum'],ax=ax4)
    df1['close'].plot(ax=ax1,alpha=0.5, color='red')
    df1['return'].plot(ax=ax2,alpha=0.5,color='green')
    df1['volume'].plot(ax=ax3)
    pass
#%%
#df.to_csv('E:/Quant/Learn backtrader/data/df.csv',index=None)
#result.head(200).reset_index(drop=True).to_csv('E:/Quant/Learn backtrader/data/top.csv',index=None)
#%%
def multi_top_analyse():
    df=pd.read_csv('E:/Quant/Learn backtrader/data/df.csv')
    top=pd.read_csv('E:/Quant/Learn backtrader/data/top.csv')
    top_list=np.array_split(top,5)
    pool=mp.Pool(processes=5)
    mult_res = pool.starmap(analyse_top, zip(top_list,repeat(df)))
    pool.close()
    pool.join()
    return mult_res
#%%
if __name__ == '__main__':
    res_list=multi_top_analyse()