import pandas as pd
from eat.io import uvfits
from eat.inspect import utils as ut
import os

def import_uvfits_set(path_data_0,data_subfolder,path_vex,path_out,out_name,tavg='scan',exptL=[3597,3598,3599,3600,3601],
    bandL=['lo','hi'],only_parallel=True,filend=".uvfits"):

    #path_vex= '/home/maciek/VEX/'
    #path_out = '/home/maciek/import_data/er4/'
    #path_data_0 = '/data/2017-april/ce/er4/
    #data_subfolder = '6.uvfits/'
    #out_name = 'polcal_after_apcal_scan_coh_avg_8b'
    if not os.path.exists(path_out):
        os.makedirs(path_out) 
    df = pd.DataFrame({})
    for band in bandL:  
        for expt in exptL:
            path0 = path_data_0+'hops-'+band+'/'+data_subfolder+str(expt)+'/'
            for filen in os.listdir(path0):
                if filen.endswith(filend): 
                    print('processing ', filen)
                    df_foo = uvfits.get_df_from_uvfit(path0+filen,path_vex=path_vex,force_singlepol='',band=band,round_s=0.1,only_parallel=only_parallel)
                    if 'std_by_mean' in df_foo.columns:
                        df_foo.drop('std_by_mean',axis=1,inplace=True)
                    df_foo['std_by_mean'] = df_foo['amp']
                    df_scan = ut.coh_avg_vis(df_foo.copy(),tavg=tavg,phase_type='phase')
                    df = pd.concat([df,df_scan.copy()],ignore_index=True)
                else:
                    pass         
    df.to_pickle(path_out+out_name+'.pic')
