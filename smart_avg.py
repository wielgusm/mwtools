import ehtim as eh
import pandas as pd
import numpy as np

#path0 = '/data/2017-april/postproc/ER6/M87/4.netcal_dcal/hops_3597_M87_LO.dcal_full.uvfits'
#obs = eh.obsdata.load_uvfits(path0,polrep='circ')
#obs.add_scans(info='vex',filepath='/home/maciek/VEX/e17d05.vex')
#vexpath='/home/maciek/VEX/e17d05.vex'

def avg_full(obs,return_type='rec'):

    vis = eh.statistics.dataframes.make_df(obs)
    grouping=['tau1','tau2','t1','t2']
    vis['number'] = 1
    aggregated = {'datetime': np.min, 'time': np.mean,
    'number': lambda x: len(x), 'u':np.mean, 'v':np.mean,'tint': np.sum}

    if obs.polrep=='stokes':
        vis1='vis'; vis2='qvis'; vis3='uvis'; vis4='vvis'
        sig1='sigma'; sig2='qsigma'; sig3='usigma'; sig4='vsigma'
    elif obs.polrep=='circ':
        vis1='rrvis'; vis2='llvis'; vis3='rlvis'; vis4='lrvis'
        sig1='rrsigma'; sig2='llsigma'; sig3='rlsigma'; sig4='lrsigma'

    #AVERAGING-------------------------------    

    meanF = lambda x: np.nanmean(np.asarray(x))
    #meanerrF = lambda x: bootstrap(np.abs(x), np.mean, num_samples=num_samples,wrapping_variable=False)
    def meanerrF(x):
        x = np.asarray(x)
        x = x[x==x]

        if len(x)>0: ret = np.sqrt(np.sum(x**2)/len(x)**2)
        else: ret = np.nan +1j*np.nan
        return ret

    aggregated[vis1] = meanF
    aggregated[vis2] = meanF
    aggregated[vis3] = meanF
    aggregated[vis4] = meanF
    aggregated[sig1] = meanerrF
    aggregated[sig2] = meanerrF
    aggregated[sig3] = meanerrF
    aggregated[sig4] = meanerrF

    #ACTUAL AVERAGING
    #print('to agg: ',np.shape(vis))
    #print(vis)
    vis_avg = vis.groupby(grouping).agg(aggregated).reset_index()

    vis_avg['amp'] = list(map(np.abs,vis_avg[vis1]))
    vis_avg['phase'] = list(map(lambda x: (180./np.pi)*np.angle(x),vis_avg[vis1]))
    vis_avg['snr'] = vis_avg['amp']/vis_avg[sig1]     
    
    if return_type=='rec':
        if obs.polrep=='stokes':
            return eh.statistics.dataframes.df_to_rec(vis_avg,'vis')
        elif obs.polrep=='circ':
            return eh.statistics.dataframes.df_to_rec(vis_avg,'vis_circ')
    elif return_type=='df':
        return vis_avg
    
    
def smart_coherent_avg(obs,vexpath,tavgmax):
    if obs.polrep != 'circ':
        obs = obs.switch_polrep('circ')
    tavg0=tavgmax
    scans_upd=[]
    for sc in obs.scans:
        if sc[1]>19.0:
            scans_upd.append([sc[0]-24,sc[1]-24])
        else:
            scans_upd.append([sc[0],sc[1]]) 
    scans_upd=np.array(scans_upd)
    obs.scans = scans_upd

    times=np.zeros(2*len(obs.scans))
    for ind,x in enumerate(obs.scans):
        times[2*ind]=x[0]
        times[2*ind+1]=x[1]
    tstartL=[x[0] for x in obs.scans]
    tstopL=[x[1] for x in obs.scans]

    which_scan=np.digitize(obs.data['time'],bins=times)

    which_scans = sorted(list(set(which_scan)))

    scan_obsdataL=[]

    #vol=0
    for Nscan in which_scans:
        tstart = tstartL[int((Nscan-1)/2)]
        tstop = tstopL[int((Nscan-1)/2)]
        obsloc=obs.flag_UT_range(UT_start_hour=tstart, UT_stop_hour=tstop,output='flagged')
        scan_obsdataL.append(obsloc)
        #vol+=np.shape(obsloc.data)[0]

    #scan_loc = scan_obsdataL[0]
    subscans_list=[]
    for indsc,scan_loc in enumerate(scan_obsdataL):
        
        print('#===================')
        print('#===================')
        print('Scan number ', indsc)
        # list of all baselines present in the scan
        bslL =sorted(list(set([x[0]+'+'+x[1] for x in scan_loc.data[['t1','t2']]])))

        df = pd.DataFrame({})
        unique_times = sorted(list(set(scan_loc.data['time'])))
        #df['time']= unique_times
        for tloc in unique_times:
            #print(tloc)
            #data only for that time point t
            obs_loc_in_time = scan_loc.data[scan_loc.data['time']==tloc]
            dict_is_observation={}
            dict_is_observation['time']=[tloc]
            code=''
            for bsl in bslL:
                t1 = bsl[0:2]
                t2 = bsl[3:5]
                is_observation = np.shape(obs_loc_in_time[(obs_loc_in_time['t1']==t1)&(obs_loc_in_time['t2']==t2)])[0]
                #print(obs_loc_in_time.data[obs_loc_in_time.data[]])
                #print([t1,t2])
                code+=str(is_observation)
                dict_is_observation[bsl]=[is_observation]
            dict_is_observation['code']=[code]
            dfloc=pd.DataFrame(dict_is_observation)
            df=pd.concat([df,dfloc],ignore_index=True)

        #this adds info about which subscan is it to df
        #one should add a break to not average for too long
        #print(df)
        #tavg in secons #max integration time in seconds
        tbreak_h = tavg0/3600.
        subscan_num=[]
        numscan=1
        code_loc_minus1 = df.iloc[0]['code']
        t0 = df.iloc[0]['time']

        for ind,row in list(df.iterrows()):

            if (row['code']!=code_loc_minus1)|(row['time']-t0 > tbreak_h):
                #print('break!')
                #print(code_loc_minus1)
                #print(row['code'])
                #print(row['code']!=code_loc_minus1)
                #print((row['time']-t0 > tbreak_h))
                #print("tbreak: ",tbreak_h)
                #print("tavg: ",tavg)
                t0 = row['time']
                numscan+=1

            subscan_num.append(numscan)
            code_loc_minus1=row['code']
        
        df['subscan']=subscan_num
        df['mintime']=df['time']
        df['maxtime']=df['time']

        #ranges of times for subscans
        teps = 1e-5
        df_subscan_times = df.groupby('subscan').agg({'mintime':np.min,'maxtime':np.max}).reset_index()

        df_subscan_times['maxtime']+=1e-5
        df_subscan_times['mintime']-=1e-5
        print('broken into subscans: ',np.shape(df_subscan_times)[0])
        #subscans_list = []
        for ind,row in df_subscan_times.iterrows():
            #print(row['mintime'],row['maxtime'])
            data_subs = obs.flag_UT_range(UT_start_hour=row['mintime'], UT_stop_hour=row['maxtime'],output='flagged')
            ##put coherent avg here
            #dfsub = eh.statistics.dataframes.make_df()
            #tavgloc = 1.2*np.abs(np.max(data_subs.data['time'])-np.min(data_subs.data['time']))
            #data_subs = data_subs.avg_coherent(inttime=3600.*tavgloc)
            #data_subs_obsdata=avg_full(data_subs)
            data_subs.data = avg_full(data_subs)
            subscans_list.append(data_subs)
        print('#===================')
        print('#===================')
    obsnew = eh.obsdata.merge_obs(subscans_list)
    return obsnew