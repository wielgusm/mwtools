import eat, os, sys
sys.path.append('/home/maciek/mwtools/')
from mwtools import importing
import pandas as pd
import numpy as np
from eat.inspect import closures as cl
from eat.inspect import utils as ut

#common directories
path_data_0 = '/data/2017-april/ce/er4v2/'
path_vex= '/home/maciek/VEX/'
path_out = '/home/maciek/validation/test_reduction/'
if not os.path.exists(path_out):
    os.makedirs(path_out)

data_subfolder='7.+apriori/'
'''
print("Saving scan averaged data...")
#scan averaged data
out_name = 'apc_sc'
importing.import_uvfits_set(path_data_0,data_subfolder,path_vex,path_out,out_name,tavg='scan',exptL=[3597,3598,3599,3600,3601],
    bandL=['lo','hi'],only_parallel=False,filend=".uvfits",out_hdf=True)

print("Saving incoherently scan averaged data...")
#incoh scan averaged data
out_name = 'apc_sc_incoh'
importing.import_uvfits_set(path_data_0,data_subfolder,path_vex,path_out,out_name,tavg='scan',exptL=[3597,3598,3599,3600,3601],
    bandL=['lo','hi'],only_parallel=False,filend=".uvfits",incoh_avg=True,out_hdf=True)
'''
print("Saving 10s data...")
#10s segmented data
out_name='apc_10s'
importing.import_uvfits_set(path_data_0,data_subfolder,path_vex,path_out,out_name,tavg=10.,exptL=[3597,3598,3599,3600,3601],
    bandL=['lo','hi'],only_parallel=False,filend=".uvfits",out_hdf=True)

print("Saving closure phases from scan-averaged data...")
#closure phases from scan-averaged segments
filen='apc_sc.h5'
data=pd.read_hdf(path_out+filen,filen.split('.')[0])
bsp = cl.all_bispectra(data,phase_type='phase')
bsp.drop('fracpols',axis=1,inplace=True)
bsp.drop('snrs',axis=1,inplace=True)
bsp.drop('amps',axis=1,inplace=True)
out_name='cp_'+filen.split('.')[0]
bsp.to_hdf(path_out+out_name+'.h5', key=out_name, mode='w',format='table')

print("Saving scan-averaged closure phases from 10s data...")
#closure phases in 10s segments >> scan averaged
filen='apc_10s.h5'
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
filen='apc_sc.h5'
data=pd.read_hdf(path_out+filen,filen.split('.')[0])
quad=cl.all_quadruples_new(data,ctype='logcamp',debias='camp')
quad.drop('snrs',axis=1,inplace=True)
quad.drop('amps',axis=1,inplace=True)
out_name='lca_'+filen.split('.')[0]
quad['scan_id'] = list(map(np.int64,quad.scan_id))
quad.to_hdf(path_out+out_name+'.h5', key=out_name, mode='w',format='table')

print("Saving scan-averaged log closure amplitudes from 10s data...")
#log closure amplitudes phases in 10s segments >> scan averaged
filen='apc_10s.h5'
data=pd.read_hdf(path_out+filen,filen.split('.')[0])
quad=cl.all_quadruples_new(data,ctype='logcamp',debias='camp')
quad.drop('snrs',axis=1,inplace=True)
quad.drop('amps',axis=1,inplace=True)
quad_sc=ut.avg_camp(quad,tavg='scan')
out_name='lca_sc_'+filen.split('.')[0]
quad_sc['scan_id'] = list(map(np.int64,quad_sc.scan_id))
quad_sc.to_hdf(path_out+out_name+'.h5', key=out_name, mode='w',format='table')