import pandas as pd
import numpy as np
from eat.io import uvfits
from eat.inspect import utils as ut
from eat.inspect import closures as cl
import os

path0 = '/data/2017-april/ce/er4/aips/e17e11-4-lo-ver1-ASIAA-31DEC18/'
path_vex= '/home/maciek/VEX/'
path_out = '/home/maciek/validation/AIPS_ASIAA_ER4/'
if not os.path.exists(path_out):
    os.makedirs(path_out)
filend='.fittp'
only_parallel=True

#####################
#SCAN AVERAGED VIS
#####################

df=pd.DataFrame({})
out_name='AIPS_ASIAA_ER4_3601_LO_SCAN_AVG'
for filen in os.listdir(path0):
    if filen.endswith(filend): 
        print('processing ', filen)
        try:
            df_foo = uvfits.get_df_from_uvfit(path0+filen,path_vex=path_vex,force_singlepol='',band='lo',round_s=0.1,only_parallel=only_parallel,rescale_noise=True)
            df_scan = ut.coh_avg_vis(df_foo.copy(),tavg='scan',phase_type='phase')
            df = pd.concat([df,df_scan.copy()],ignore_index=True)
        except: pass

df.to_hdf(path_out+out_name+'.h5', key=out_name, mode='w',format='table')

#####################
#10s AVERAGED VIS
#####################

df=pd.DataFrame({})
out_name='AIPS_ASIAA_ER4_3601_LO_10s_AVG'
for filen in os.listdir(path0):
    if filen.endswith(filend): 
        print('processing ', filen)
        try:
            df_foo = uvfits.get_df_from_uvfit(path0+filen,path_vex=path_vex,force_singlepol='',band='lo',round_s=0.1,only_parallel=only_parallel)
            df_scan = ut.coh_avg_vis(df_foo.copy(),tavg=10.,phase_type='phase')
            df = pd.concat([df,df_scan.copy()],ignore_index=True)
        except: pass

df.to_hdf(path_out+out_name+'.h5', key=out_name, mode='w',format='table')

#####################
#CLOSURES
#####################


print("Saving closure phases from scan-averaged data...")
#closure phases from scan-averaged segments
filen='AIPS_ASIAA_ER4_3601_LO_SCAN_AVG.h5'
data=pd.read_hdf(path_out+filen,filen.split('.')[0])
bsp = cl.all_bispectra(data,phase_type='phase')
bsp.drop('fracpols',axis=1,inplace=True)
bsp.drop('snrs',axis=1,inplace=True)
bsp.drop('amps',axis=1,inplace=True)
out_name='cp_'+filen.split('.')[0]
bsp.to_hdf(path_out+out_name+'.h5', key=out_name, mode='w',format='table')

print("Saving scan-averaged closure phases from 10s data...")
#closure phases in 10s segments >> scan averaged
filen='AIPS_ASIAA_ER4_3601_LO_10s_AVG.h5'
data=pd.read_hdf(path_out+filen,filen.split('.')[0])
bsp = cl.all_bispectra(data,phase_type='phase')
bsp.drop('fracpols',axis=1,inplace=True)
bsp.drop('snrs',axis=1,inplace=True)
bsp.drop('amps',axis=1,inplace=True)
bsp_sc = ut.coh_avg_bsp(bsp,tavg='scan')
out_name='cp_sc_'+filen.split('.')[0]
bsp_sc.to_hdf(path_out+out_name+'.h5', key=out_name, mode='w',format='table')

print("Saving log closure amplitudes from scan-averaged data...")
#log closure amplitudes from scan-averaged segments
filen='AIPS_ASIAA_ER4_3601_LO_SCAN_AVG.h5'
data=pd.read_hdf(path_out+filen,filen.split('.')[0])
quad=cl.all_quadruples_new(data,ctype='logcamp',debias='camp')
quad.drop('snrs',axis=1,inplace=True)
quad.drop('amps',axis=1,inplace=True)
out_name='lca_'+filen.split('.')[0]
quad['scan_id'] = list(map(np.int64,quad.scan_id))
quad.to_hdf(path_out+out_name+'.h5', key=out_name, mode='w',format='table')

print("Saving scan-averaged log closure amplitudes from 10s data...")
#log closure amplitudes phases in 10s segments >> scan averaged
filen='AIPS_ASIAA_ER4_3601_LO_10s_AVG.h5'
data=pd.read_hdf(path_out+filen,filen.split('.')[0])
quad=cl.all_quadruples_new(data,ctype='logcamp',debias='camp')
quad.drop('snrs',axis=1,inplace=True)
quad.drop('amps',axis=1,inplace=True)
quad_sc=ut.avg_camp(quad,tavg='scan')
out_name='lca_sc_'+filen.split('.')[0]
quad_sc['scan_id'] = list(map(np.int64,quad_sc.scan_id))
quad_sc.to_hdf(path_out+out_name+'.h5', key=out_name, mode='w',format='table')