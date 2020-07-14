from src.utils.gramian import *
import numpy as np
import math
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np 
import torch

exclusion_zone_peaks_out=[]
exclusion_zone_peaks=[]

def tensor_agg(train_df, train_peaks, out_df, out_peaks,seql=60,w_tot=60,pre_wait=3,
               wavelet1='BBv',gram1='ichspan',gram2='HTL',
               exclude_pko=exclusion_zone_peaks_out,exclude_pk=exclusion_zone_peaks):

    outer_index=list(out_df.reset_index().drop(exclude_pko).index)
    outer_indexo=outer_index[60:len(out_df)-60]
    nn=2*len(out_peaks)-7
    tfo = torch.empty(size=(nn-7,3,w_tot,seql))
    lfo=torch.empty(size=(nn-7,1),dtype=int)
    ii=0
    tran=GAF()


    for j in range(nn):
        if j>6:
            if j < len(out_peaks):
                dft_seqo=out_df.reset_index().iloc[out_peaks[j]-seql-pre_wait:out_peaks[j]-pre_wait]
                lfo[ii]=1

            
            if j >= len(out_peaks):
                r=np.random.choice(outer_indexo)
                dft_seqo=out_df.reset_index().iloc[r-seql:r].fillna(0)
                lfo[ii]=0

            sig=dft_seqo[wavelet1]fillna(0)
            widths = np.logspace(0, 1.2,w_tot)
            tfcw_bbv = torch.tensor(signal.cwt(sig, signal.ricker, widths))

            gr1=dft_seqo[gram1].values.reshape(-1,1)
            gr2=dft_seqo[gram2].values.reshape(-1,1)
            tfg1=torch.tensor(tran.transform(gr1)[0])
            tfg2=torch.tensor(tran.transform(gr2)[0])

            tfo[ii]=torch.stack([tfcw_bbv, tfg1 , tfg2])
            ii=ii+1

    outer_index=list(out_df.reset_index().drop(exclude_pko).index)
    out_df.index
    len(out_df)

    seql=60
    w_tot=60

    outer_index=list(train_df.drop(exclude_pk).index)
    outer_index=outer_index[100:len(train_df)-100]
    nn=2*len(train_peaks)
    tf = torch.empty(size=(nn-7,3,w_tot,seql))
    lf=torch.empty(size=(nn-7,1),dtype=int)
    ii=0
    tran=GAF()


    for j in range(nn):
        if j>6:
            if j < len(train_peaks):
                train_df_seq=train_df.iloc[train_peaks[j]-seql-pre_wait:train_peaks[j]-pre_wait]
                lf[ii]=1

            
            if j >= len(train_peaks):
                r=np.random.choice(outer_index)
                train_df_seq=train_df.iloc[r-seql:r].fillna(0)
                lf[ii]=0

            sig=train_df_seq[wavelet1].dropna()
            widths = np.logspace(0, 1.2,w_tot)
            tfcw_bbv = torch.tensor(signal.cwt(sig, signal.ricker, widths))

            gr1=train_df_seq[gram1].values.reshape(-1,1)
            gr2=train_df_seq[gram2].values.reshape(-1,1)
            tfg1=torch.tensor(tran.transform(gr1)[0])
            tfg2=torch.tensor(tran.transform(gr2)[0])

            tf[ii]=torch.stack([tfcw_bbv, tfg1 , tfg2])
            
            ii=ii+1
            
    return tf,lf,tfo,lfo
