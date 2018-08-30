import pandas as pd
from eat.io import uvfits
from eat.inspect import utils as ut
import os

path0 = '/data/2017-april/ce/er4/aips/e17e11-4-lo-ver1/'
path_vex= '/home/maciek/VEX/'
path_out = '/home/maciek/validation/AIPS_SARA_ER4/'
if not os.path.exists(path_out):
    os.makedirs(path_out)
filend='.fittp'
only_parallel=True
df=pd.DataFrame({})
out_name='AIPS_SARA_ER4_3601_LO'
for filen in os.listdir(path0):
    if filen.endswith(filend): 
        print('processing ', filen)
        try:
            df_foo = uvfits.get_df_from_uvfit(path0+filen,path_vex=path_vex,force_singlepol='',band='lo',round_s=0.1,only_parallel=only_parallel)
            df_scan = ut.coh_avg_vis(df_foo.copy(),tavg='scan',phase_type='phase')
            df = pd.concat([df,df_scan.copy()],ignore_index=True)
        except: pass

df.to_hdf(path_out+out_name+'.h5', key=out_name, mode='w',format='table')