#%%
import backtrader as bt
import datetime
from backtrader.dataseries import TimeFrame
import pandas as pd
import math
from learning.multiple_time_class import MultiStrategy

#data=pd.read_csv('1minutes.csv',header=None)
#data[0]=data[0].apply(lambda x:float(x)/1000.0 )
#data_demo= data.tail(60000).to_csv('data_demo.csv',index=False)
      

cerebro = bt.Cerebro()
data = bt.feeds.GenericCSVData(
    dataname='data\data_demo.csv',
    compression =15,
    timeframe=bt.TimeFrame.Minutes,
    dtformat=lambda x: datetime.datetime.utcfromtimestamp(float(x)),
    datetime=0,
    open=1,
    high=2,
    low=3,
    close=4,
    volume=5,
    openinteres=-1,
)
cerebro.adddata(data,name='hs15m')
cerebro.resampledata(data, name='hs1h',timeframe=bt.TimeFrame.Minutes,compression=60)
cerebro.resampledata(data, name='hs1d',timeframe=bt.TimeFrame.Days)
cerebro.addobserver(bt.observers.Trades)
cerebro.addobserver(bt.observers.BuySell)

cerebro.addstrategy(MultiStrategy)

cerebro.broker.setcash(1000000)

cerebro.broker.setcommission(0.002)

cerebro.run()

cerebro.plot(style='bar')


