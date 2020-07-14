from math import pi
import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_file, save,show
from bokeh.models import LinearAxis, Range1d
from bokeh.palettes import Category20 
import itertools
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler,Normalizer,StandardScaler



class Plotter():


    def __init__(self,fin_data_ta,peaks):
        self.dft=fin_data_ta
        self.featnorm=['MOM','DX','deltma','deltma_v','RSI','ichspan','HTBB',
            'dBB','brLBB','BBv','ATR','AD','UBBv','LBBv']
        self.featabs=['MA60','MA200','MA400','MA800','date','close','BB','UBB',
            'LBB','HTL','ichi0','ichi1','ichi2','ichi3','ichi4']
        self.peaks=peaks
        
    def scaling(self):

        
        self.featls=self.featabs+self.featnorm
        data=self.dft
        features = data[self.featnorm]

        ct = ColumnTransformer([
                ('somename', MinMaxScaler(), self.featnorm)
            ], remainder='passthrough')

        dft2=ct.fit_transform(features)
        df3=pd.DataFrame(dft2)
        df3.columns=self.featnorm
        for c in self.featabs:
            df3[c]=data.reset_index()[c].copy()
        df3=df3.fillna(value=0)
        self.data_scaled=df3
        self.featls.remove('date')
        self.featabs.remove('date')
        
    def save(self):
        dft=self.dft
        df3=self.data_scaled
        print(dft.columns)
        inc = dft.close > dft['open']
        dec = dft['open'] > dft.close
        w = .01*.12*.5*(1./6)*60*60*1000*100 # half day in ms


        TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
        p = figure(x_axis_type="datetime", y_axis_type="linear",tools=TOOLS, plot_width=1000, title = "candlesp")
        p.xaxis.major_label_orientation = pi/4
        p.grid.grid_line_alpha=0.3
        p.extra_y_ranges = {"ind": Range1d(start=-8, end=3)}
        p.add_layout(LinearAxis(y_range_name="ind"), 'right')

        k=0
        for f in ['close','MA60']:
            p.line(df3['date'], df3[f],  line_width=3,line_alpha=.76,line_color=Category20[20][k%20],legend_label=str(f))
            k+=1

        k=0
        for f in self.featnorm:
            p.line(df3['date'], df3[f]-k,  line_width=3,line_alpha=.76,line_color=Category20[20][k%20],legend_label=str(f),y_range_name="ind")
            k+=1

        p.triangle(dft.date.iloc[self.peaks],1*dft.close.iloc[self.peaks],size=5, color="red")
        p.segment(dft.date.iloc[self.peaks],2.5,dft.date.iloc[self.peaks],-18,line_width=1.3, line_alpha=.3,color="red",y_range_name="ind")

        p.legend.location = "top_left"
        p.y_range=Range1d(-18000, 12000)

        output_file(filename='candles_and_ta.html',title="candlesp")
        save(p)
        return

