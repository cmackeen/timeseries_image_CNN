from src.pre_process import *
from src.label_tensorfy import *
from src.train import *
from src.utils.plot_ta import *  



def main(filename='./data/BTC_USDT_5m.csv',plot_bool=True,train_bool=False):

    fin_data=FinData(filename)
    fin_data.window()
    fin_data.ta_columns()
    peaks,exclusion_zone_peaks,peaks_out,exclusion_zone_peaks_out=fin_data.peak_loc()

    if plot_bool==True:
        pl=Plotter(fin_data.data_in,peaks)
        pl.scaling()
        pl.save()

    if train_bool==True:
        tf,lf,tfo,lfo=tensor_agg(fin_data.data_in, peaks,fin_data.data_out, peaks_out)
        tr_set,tt_set,to_set=tensor_load(tf,lf,tfo,lfo)

        tf_length=len(tf)
        trained_model=train(tr_set,tf_length,EPOCH=8)
        model_eval(trained_model,to_set)
    
    return


main()
