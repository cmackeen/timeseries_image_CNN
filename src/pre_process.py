
from datetime import timedelta, datetime, timezone
import sys, os, time, random
import pandas as pd
import json
import csv
import sqlite3
from sqlite3 import Error
import talib as tb
from math import pi
import pandas as pd
from scipy.signal import find_peaks,argrelextrema
from sklearn.preprocessing import MinMaxScaler,Normalizer,StandardScaler


def ichimoku(price_objs):
    prdf = price_objs
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2))
    period9_high = prdf.rolling(window=9).max()
    period9_low = prdf.rolling(window=9).min()
    tenkan_sen = (period9_high + period9_low) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low)/2))
    period26_high = prdf.rolling(window=26).max()
    period26_low =prdf.rolling(window=26).min()
    kijun_sen = (period26_high + period26_low) / 2

    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2))
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2))
    period52_high = prdf.rolling(window=52).max()
    period52_low = prdf.rolling(window=52).min()
    senkou_span_b = ((period52_high + period52_low) / 2).shift(26)

    # The most current closing price plotted 22 time periods behind (optional)
    chikou_span = prdf.shift(-22)  # 22 according to investopedia

    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

class FinData():
    def __init__(self,data_file):
        self.candles=pd.read_csv(data_file)
        
    def window(self):
        df=self.candles
        df['date'] = df['timestamp'].apply(lambda x: datetime.utcfromtimestamp(x))
        df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M %p')
        df=df.drop(['timestamp'],axis=1)
        df.columns=['open','high','low','close','volume','date']
        df["date"] = pd.to_datetime(df["date"],format='%Y-%m-%d %H:%M %p')

        window={'start':'2013-1-28 12:00:00','end':'2020-4-28 12:00:00'}
        dft=df[df['date']<pd.to_datetime(window['end'])]
        dft=dft[dft['date']>pd.to_datetime(window['start'])]

        dfo=df[df['date']>pd.to_datetime(window['end'])]
        self.data_in=dft
        self.data_out=dfo
        return 



    def ta_columns(self):
        basema=200


        def columnizer(dft):
            dft['MA60']=tb.MA(dft.close.values,timeperiod=60)
            dft['MA200']=tb.MA(dft.close.values,timeperiod=200)
            dft['MA400']=tb.MA(dft.close.values,timeperiod=400)
            dft['MA800']=tb.MA(dft.close.values,timeperiod=800)
            dft['HTL']=tb.HT_TRENDLINE(dft.close.values)
            dft['UBB'],dft['BB'],dft['LBB']=tb.BBANDS(dft.close.values,timeperiod=60, 
                                                        nbdevup=2, nbdevdn=2)

            dft['RSI']=tb.MA(tb.RSI(dft.close.values,timeperiod=800),basema)
            dft['MOM']=tb.MA(tb.MOM(dft.close.values,timeperiod=800),basema)
            dft['DX']=tb.MA(tb.DX(dft.high, dft.low, dft.close, timeperiod=800),basema)
            dft['ATR']=tb.ATR(dft.high,dft.low,dft.close)
            dft['AD']=tb.AD(dft.high,dft.low,dft.close,dft.volume)
            for i in range(5):
                dft['ichi'+str(i)]=ichimoku(dft.close)[i]

            dft['HTBB']=dft.HTL-dft.BB
            dft['HTBB_v']=dft.HTBB.diff()
            dft['BBv']=dft.BB.diff()
            dft['dBB']=dft.UBB-dft.LBB
            dft['brLBB']=dft.close-dft.LBB
            dft['ichspan']=dft.ichi2-dft.ichi3

            dft['UBBv']=dft.UBB.diff()
            dft['LBBv']=dft.LBB.diff()

            dft['deltma']=dft.MA200-dft.MA60
            dft['deltma_v']=dft.deltma.diff()
            dft['RSI_v']=dft.RSI.diff()
            dft['volume_v']=dft.volume.diff()
            dft['close_v']=dft.close.diff()
            return dft

        self.data_in=columnizer(self.data_in)
        self.data_out=columnizer(self.data_out)


    

    def peak_loc(self,prom=100,dis=5):
        
        """
        Summary line.

        Extended description of function.

        Parameters
        ----------
        prom : int
            Prominence threshodl to identify peak  (default=100)
        dis : int
            Minimum distance between peaks  (default=5)

        Returns
        -------
        4 arrays
            peaks, peak exclusion neighborhood, out-of-data peaks, and corresponding exclusion neighborhood

        """


        peaks, _= find_peaks(self.data_in.close.values,distance=dis,prominence=prom)
        troughs, _= find_peaks(-1*self.data_in.close.values,distance=dis,prominence=prom)

        troughs
        peakso, _= find_peaks(self.data_out.close.values,distance=dis,prominence=prom)
        troughso, _= find_peaks(-1*self.data_out.close.values,distance=dis,prominence=prom)

        #excludes region neer peaks when building null (class=0) snapshots
        exclude_pk=[i+j for j in range(-5,5) for i in peaks]
        exclude_pko=[i+j for j in range(-5,5) for i in peakso]
        
        return peaks,exclude_pk,peakso,exclude_pko

