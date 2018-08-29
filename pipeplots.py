import numpy as np
import pandas as pd
import seaborn as sns
from eat.io import hops, util
from eat.hops import util as hu
import matplotlib.pyplot as plt
from eat.inspect import utils as ut

dict_col_sour = {'1055+018': 'tan',
     '1749+096': 'crimson',
     '1921-293': 'mediumblue',
     '3C273': 'lime',
     '3C279': 'magenta',
     '3C454.3': 'cyan',
     '3C84': 'blueviolet',
     'BLLAC': 'orange',
     'CENA': 'darkgreen',
     'CTA102': 'yellow',
     'J0006-0623': 'tomato',
     'J0132-1654': 'olivedrab',
     'J1733-1304': 'salmon',
     'J1924-2914': 'saddlebrown',
     'M87': 'k',
     'NGC1052': 'dodgerblue',
     'OJ287': 'gold',
     'SGRA': 'red'}

SMT2Z = {'ALMA': 'A', 'APEX': 'X', 'JCMT': 'J', 'LMT':'L', 'SMR':'R', 'SMA':'S', 'SMT':'Z', 'PV':'P','SPT':'Y'}
Z2SMT = {v: k for k, v in SMT2Z.items()}
Z2SMT['P'] = 'IRAM30'
SMT2Zb = {'ALMA': 'A', 'APEX': 'X', 'JCMT': 'J', 'LMT':'L', 'SMR':'R', 'SMA':'S', 'SMT':'Z', 'IRAM30':'P','SPT':'Y'}

palette_dict = {'ALMA-APEX':'k','JCMT-SMA':'k','SMT-LMT':'lime','ALMA-LMT':'mediumblue','APEX-LMT':'mediumblue',
    'SMT-SMA':'red','SMT-JCMT':'red','LMT-SMA':'cyan','JCMT-LMT':'cyan',
    'ALMA-SMT':'magenta','APEX-SMT':'magenta','ALMA-SPT':'blueviolet','APEX-SPT':'blueviolet',
    'ALMA-IRAM30':'orange','APEX-IRAM30':'orange','ALMA-SMA':'darkgreen','ALMA-JCMT':'darkgreen','APEX-SMA':'darkgreen','APEX-JCMT':'darkgreen',
    'LMT-SPT':'yellow','LMT-IRAM30':'tomato','SMA-SPT':'olivedrab','JCMT-SPT':'olivedrab',
    'SMT-SPT':'salmon', 'IRAM30-SPT':'saddlebrown','IRAM30-SMA':'tan','JCMT-IRAM30':'tan',
    'SMT-IRAM30':'dodgerblue'}

palette_dict_rev = {k.split('-')[1]+'-'+k.split('-')[0]:v for k, v in palette_dict.items()}
palette_dict = {**palette_dict, **palette_dict_rev}

def plot_amp_days(data,sour, bars_on=False, only_parallel=True,logscale=True,palette_dict=palette_dict):

    if only_parallel==True:
        foo=data[(data.source==sour)&(data.polarization.str[0]==data.polarization.str[1])].copy()
    else:
        foo=data[(data.source==sour)].copy()
    foo['baseline']=list(map(lambda x: Z2SMT[x[0]]+'-'+Z2SMT[x[1]],foo.baseline))
    exptL=list(foo.expt_no.unique())
    nplots=(len(exptL)+1)
    ncols=2
    #nrows=np.maximum(2,int(np.ceil(nplots/ncols)))
    nrows=int(np.ceil(nplots/ncols))

    if 'uvdist' not in foo.columns:
        foo['uvdist'] = np.sqrt(foo['u']**2 + foo['v']**2)
    exptD = dict(zip(range(nplots),['all']+exptL))

    if nrows>1:
        fig, ax = plt.subplots(nrows,ncols,sharey='all',sharex='all',figsize=(ncols*7,nrows*5))
        couP = 0 
        for couR in range(nrows):
            for couC in range(ncols):
                if couP<nplots:
                    exptLoc = exptD[couP]
                    if exptLoc=='all':
                        foo1 = foo
                    else:
                        foo1=foo[foo.expt_no==exptLoc]

                    for base in list(foo1.baseline.unique()):

                        foo2=foo1[foo1.baseline==base]
                        if ('ALMA' in base) or (('SMA' in base)&('JCMT' not in base)):
                            ax[couR,couC].errorbar(foo2.uvdist,foo2.amp,bars_on*foo2.sigma,fmt='o',mfc='none',ms=8,color=palette_dict[base],label=SMT2Zb[base.split('-')[0]]+SMT2Zb[base.split('-')[1]])
                        else:
                            ax[couR,couC].errorbar(foo2.uvdist,foo2.amp,bars_on*foo2.sigma,fmt='x',ms=5,color=palette_dict[base],label=SMT2Zb[base.split('-')[0]]+SMT2Zb[base.split('-')[1]])
                    if (couR==0)&(couC==0):
                        ax[couR,couC].legend(bbox_to_anchor=(-0.15, 1.52))
                    ax[couR,couC].grid()
                    ax[couR,couC].set_title(sour+' | '+str((exptD[couP])))
                    ax[couR,couC].set_xlabel('UT time')
                    ax[couR,couC].set_ylabel('amplitudes')
                    if logscale==True:
                        ax[couR,couC].set_yscale("log", nonposy='clip')
                    ax[couR,couC].set_ylim(ymin=0.1*np.min(foo.amp))
                    ax[couR,couC].set_ylim(ymax=1.1*np.max(foo.amp))
                    couP+=1
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(1,ncols,sharey='all',sharex='all',figsize=(ncols*7,nrows*5))
        couP = 0 
        for couC in range(ncols):
            if couP<nplots:
                exptLoc = exptD[couP]
                if exptLoc=='all':
                    foo1 = foo
                else:
                    foo1=foo[foo.expt_no==exptLoc]

                for base in list(foo1.baseline.unique()):

                    foo2=foo1[foo1.baseline==base]
                    if ('ALMA' in base) or (('SMA' in base)&('JCMT' not in base)):
                        ax[couC].errorbar(foo2.uvdist,foo2.amp,bars_on*foo2.sigma,fmt='o',mfc='none',ms=8,color=palette_dict[base],label=SMT2Zb[base.split('-')[0]]+SMT2Zb[base.split('-')[1]])
                    else:
                        ax[couC].errorbar(foo2.uvdist,foo2.amp,bars_on*foo2.sigma,fmt='x',ms=5,color=palette_dict[base],label=SMT2Zb[base.split('-')[0]]+SMT2Zb[base.split('-')[1]])
                if (couC==0):
                    ax[couC].legend(bbox_to_anchor=(-0.15, 1.52))
                ax[couC].grid()
                ax[couC].set_title(sour+' | '+str((exptD[couP])))
                ax[couC].set_xlabel('UT time')
                ax[couC].set_ylabel('amplitudes')
                if logscale==True:
                    ax[couC].set_yscale("log", nonposy='clip')
                ax[couC].set_ylim(ymin=0.1*np.min(foo.amp))
                ax[couC].set_ylim(ymax=1.1*np.max(foo.amp))
                couP+=1
        plt.tight_layout()
    
    nscan=np.shape(foo)[0]
    nbase = len(foo.baseline.unique())
    print(sour)
    print("{} detections on {} baselines".format(nscan,nbase))
    print("median snr {}".format(np.median(foo.snr)))
    print("=========================================")
    plt.show()


def compare_uvf_apc(apc_sc,uvf_sc):

    uvf,apc=ut.match_frames(uvf_sc.copy(),apc_sc.copy(),['source','band','polarization','scan_id','baseline'])
    apc['var_before'] = uvf['std_by_mean']
    apc['var_after'] = apc['std_by_mean']
    apc['after2before'] = apc['var_after']/apc['var_before']
    data=apc.copy()
    sns.set_style('whitegrid')
    baseL = sorted(list(data.baseline.unique()))
    ncol = int(np.ceil(len(baseL)/2))
    num_base = (np.ceil((np.asarray(range(2*ncol))+0.1)/2))
    diccol=dict(zip(baseL,num_base))
    apc['basenum'] = list(map(lambda x: diccol[x],apc.baseline))

    for basenum in sorted(apc.basenum.unique()):
        data=apc[apc.basenum==basenum]
        max_plot = np.maximum(np.max(data.var_before),np.max(data.var_after))
        min_plot = np.minimum(np.min(data.var_before),np.min(data.var_after))
        sg=sns.lmplot(data=data,x='var_before',y='var_after',hue='source',col='baseline',fit_reg=False,sharey=False,sharex=False,
                     palette=dict_col_sour)
        ax1 = sg.fig.axes[0]
        hm1y=ax1.get_ylim()
        hm1x=ax1.get_xlim()
        ax1.plot([0,max_plot],[0,max_plot],'k--')
        ax1.set_ylim(hm1y)
        ax1.set_xlim(hm1x)
        ax2 = sg.fig.axes[1]
        hm2x=ax2.get_xlim()
        hm2y=ax2.get_ylim()
        ax2.plot([0,max_plot],[0,max_plot],'k--')
        ax2.set_ylim(hm2y)
        ax2.set_xlim(hm2x)
        plt.show()

    return data[['datetime','source','expt_no','scan_id','polarization','band','baseline','var_before','var_after','after2before']].copy()


def compare_coherence_time(coh0,incoh0):
    coh,incoh = ut.match_frames(coh0,incoh0,['scan_id','baseline','polarization','band'])

    coh['amp_coh'] = np.sqrt(coh['amp']**2 - coh['sigma']**2)
    coh['amp_incoh'] = incoh['amp']
    coh['sigma_coh'] = coh['sigma']
    coh['sigma_incoh'] = incoh['sigma']
    coh['coh2incoh'] = coh['amp_coh']/coh['amp_incoh']
    data=coh.copy()
    sns.set_style('whitegrid')
    baseL = sorted(list(data.baseline.unique()))
    ncol = int(np.ceil(len(baseL)/2))
    num_base = (np.ceil((np.asarray(range(2*ncol))+0.1)/2))
    diccol=dict(zip(baseL,num_base))
    coh['basenum'] = list(map(lambda x: diccol[x],apc.baseline))

    for basenum in sorted(coh.basenum.unique()):
        data=coh[coh.basenum==basenum].copy()
        max_plot = np.maximum(np.max(data.var_before),np.max(data.var_after))
        min_plot = np.minimum(np.min(data.var_before),np.min(data.var_after))
        sg=sns.lmplot(data=data,x='amp_coh',y='amp_incoh',hue='source',col='baseline',fit_reg=False,sharey=False,sharex=False,
                     palette=dict_col_sour)
        ax1 = sg.fig.axes[0]
        hm1y=ax1.get_ylim()
        hm1x=ax1.get_xlim()
        ax1.plot([0,max_plot],[0,max_plot],'k--')
        ax1.set_ylim(hm1y)
        ax1.set_xlim(hm1x)
        ax2 = sg.fig.axes[1]
        hm2x=ax2.get_xlim()
        hm2y=ax2.get_ylim()
        ax2.plot([0,max_plot],[0,max_plot],'k--')
        ax2.set_ylim(hm2y)
        ax2.set_xlim(hm2x)
        plt.show()

    return data[['datetime','source','expt_no','scan_id','polarization','band','baseline','amp_coh','amp_incoh','coh2incoh','sigma_coh','sigma_incoh']].copy()


def bandpass_amplitude_consistency(data0,xmax=10):

    data_lo, data_hi = ut.match_frames(data0[data0.band=='lo'].copy(),data0[data0.band=='hi'].copy(),['scan_id','baseline','polarization'])
    data = data_lo.copy()
    data['amp_lo'] = data_lo['amp']
    data['amp_hi'] = data_hi['amp']
    data['sigma_lo'] = data_lo['sigma']
    data['sigma_hi'] = data_hi['sigma']
    data['sigma'] = np.sqrt(data['sigma_lo']**2 + data['sigma_hi']**2)
    data['amp_diff'] = np.asarray(data['amp_lo']) - np.asarray(data['amp_hi'])
    data['rel_diff'] = np.asarray(data['amp_diff'])/np.asarray(data['sigma'])

    nbins = int(np.sqrt(np.shape(data)[0]))
    bins = np.linspace(-xmax,xmax,nbins)
    x=np.linspace(-xmax,xmax,128)
    plt.hist(data['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
    plt.grid()
    plt.plot(x,np.exp(-(x)**2/2)/np.sqrt(2.*np.pi),'k')
    plt.axvline(0,color='k')
    plt.xlabel('(LO-HI)/(thermal error)')
    plt.title('All data')
    plt.show()

    sourceL = sorted(list(data.source.unique()))
    nplots=len(sourceL)
    ncols=2
    nrows=int(np.ceil(nplots/ncols))
    fig, ax = plt.subplots(nrows,ncols,sharey='all',sharex='all',figsize=(ncols*7,nrows*5))

    for cou,sour in enumerate(sourceL):
        nbins = int(np.sqrt(np.shape(data[data.source==sour])[0]))
        bins = np.linspace(-xmax,xmax,nbins)
        nrowL = int(np.floor(cou/2))
        ncolL = cou%ncols
        ax[nrowL,ncolL].hist(data[data.source==sour]['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
        ax[nrowL,ncolL].plot(x,np.exp(-(x)**2/2)/np.sqrt(2.*np.pi),'k')
        ax[nrowL,ncolL].grid()
        ax[nrowL,ncolL].axvline(0,color='k')
        ax[nrowL,ncolL].set_xlabel('(LO-HI)/(thermal error)')
        ax[nrowL,ncolL].set_title(sour)
    plt.show()
    return data


def bandpass_amplitude_rel_consistency(data0,xmax=2.):

    data_lo, data_hi = ut.match_frames(data0[data0.band=='lo'].copy(),data0[data0.band=='hi'].copy(),['scan_id','baseline','polarization'])
    data = data_lo.copy()
    data['amp_lo'] = data_lo['amp']
    data['amp_hi'] = data_hi['amp']
    data['sigma_lo'] = data_lo['sigma']
    data['sigma_hi'] = data_hi['sigma']
    data['sigma'] = np.sqrt(data['sigma_lo']**2 + data['sigma_hi']**2)
    data['amp_diff'] = np.asarray(data['amp_lo']) - np.asarray(data['amp_hi'])
    data['amp_mean'] = 0.5*(np.asarray(data['amp_lo']) + np.asarray(data['amp_hi']))
    data['rel_diff'] = np.asarray(data['amp_diff'])/np.asarray(data['amp_mean'])

    nbins = int(np.sqrt(np.shape(data)[0]))
    bins = np.linspace(-xmax,xmax,nbins)
    plt.hist(data['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
    plt.grid()
    plt.axvline(0,color='k')
    plt.xlabel('2*(LO-HI)/(L0 + HI)')
    plt.title('All data')
    plt.show()

    sourceL = sorted(list(data.source.unique()))
    nplots=len(sourceL)
    ncols=2
    nrows=int(np.ceil(nplots/ncols))
    fig, ax = plt.subplots(nrows,ncols,sharey='all',sharex='all',figsize=(ncols*7,nrows*5))

    for cou,sour in enumerate(sourceL):
        nbins = int(np.sqrt(np.shape(data[data.source==sour])[0]))
        bins = np.linspace(-xmax,xmax,nbins)
        nrowL = int(np.floor(cou/2))
        ncolL = cou%ncols
        ax[nrowL,ncolL].hist(data[data.source==sour]['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
        ax[nrowL,ncolL].grid()
        ax[nrowL,ncolL].axvline(0,color='k')
        ax[nrowL,ncolL].set_xlabel('2*(LO-HI)/(L0 + HI)')
        #ax[nrowL,ncolL].set_xlim([-xmax,xmax])
        ax[nrowL,ncolL].set_title(sour)
    plt.show()
    return data


def polar_amplitude_consistency(data0,xmax=10):

    data_rr, data_ll = ut.match_frames(data0[data0.polarization=='LL'].copy(),data0[data0.polarization=='RR'].copy(),['scan_id','baseline','band'])
    data = data_rr.copy()
    data['amp_rr'] = data_rr['amp']
    data['amp_ll'] = data_ll['amp']
    data['sigma_rr'] = data_rr['sigma']
    data['sigma_ll'] = data_ll['sigma']
    data['sigma'] = np.sqrt(data['sigma_rr']**2 + data['sigma_ll']**2)
    data['amp_diff'] = data['amp_rr'] - data['amp_ll']
    data['rel_diff'] = data['amp_diff']/data['sigma']

    nbins = int(np.sqrt(np.shape(data)[0]))
    bins = np.linspace(-xmax,xmax,nbins)
    x=np.linspace(-xmax,xmax,128)
    plt.hist(data['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
    plt.grid()
    plt.plot(x,np.exp(-(x)**2/2)/np.sqrt(2.*np.pi),'k')
    plt.axvline(0,color='k')
    plt.xlabel('(RR-LL)/(thermal error)')
    plt.title('All data')
    plt.show()

    sourceL = sorted(list(data.source.unique()))
    nplots=len(sourceL)
    ncols=2
    nrows=int(np.ceil(nplots/ncols))
    fig, ax = plt.subplots(nrows,ncols,sharey='all',sharex='all',figsize=(ncols*7,nrows*5))

    for cou,sour in enumerate(sourceL):
        nbins = int(np.sqrt(np.shape(data[data.source==sour])[0]))
        bins = np.linspace(-xmax,xmax,nbins)
        nrowL = int(np.floor(cou/2))
        ncolL = cou%ncols
        ax[nrowL,ncolL].hist(data[data.source==sour]['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
        ax[nrowL,ncolL].plot(x,np.exp(-(x)**2/2)/np.sqrt(2.*np.pi),'k')
        ax[nrowL,ncolL].grid()
        ax[nrowL,ncolL].axvline(0,color='k')
        ax[nrowL,ncolL].set_xlabel('(RR-LL)/(thermal error)')
        ax[nrowL,ncolL].set_title(sour)
    plt.show()
    return data


def polar_amplitude_rel_consistency(data0,xmax=2.):

    data_rr, data_ll = ut.match_frames(data0[data0.polarization=='LL'].copy(),data0[data0.polarization=='RR'].copy(),['scan_id','baseline','band'])
    data = data_rr.copy()
    data['amp_rr'] = data_rr['amp']
    data['amp_ll'] = data_ll['amp']
    data['sigma_rr'] = data_rr['sigma']
    data['sigma_ll'] = data_ll['sigma']
    data['sigma'] = np.sqrt(data['sigma_rr']**2 + data['sigma_ll']**2)
    data['amp_diff'] = data['amp_rr'] - data['amp_ll']
    data['amp_mean'] = 0.5*(data['amp_rr']+data['amp_ll'])
    data['rel_diff'] = data['amp_diff']/data['amp_mean']

    nbins = int(np.sqrt(np.shape(data)[0]))
    bins = np.linspace(-xmax,xmax,nbins)
    plt.hist(data['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
    plt.grid()
    plt.axvline(0,color='k')
    plt.xlabel('2*(RR-LL)/(RR+LL)')
    plt.title('All data')
    plt.show()

    sourceL = sorted(list(data.source.unique()))
    nplots=len(sourceL)
    ncols=2
    nrows=int(np.ceil(nplots/ncols))
    fig, ax = plt.subplots(nrows,ncols,sharey='all',sharex='all',figsize=(ncols*7,nrows*5))

    for cou,sour in enumerate(sourceL):
        nbins = int(np.sqrt(np.shape(data[data.source==sour])[0]))
        bins = np.linspace(-xmax,xmax,nbins)
        nrowL = int(np.floor(cou/2))
        ncolL = cou%ncols
        ax[nrowL,ncolL].hist(data[data.source==sour]['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
        ax[nrowL,ncolL].grid()
        ax[nrowL,ncolL].axvline(0,color='k')
        ax[nrowL,ncolL].set_xlabel('2*(RR-LL)/(RR+LL)')
        ax[nrowL,ncolL].set_title(sour)
    plt.show()
    return data