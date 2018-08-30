import numpy as np
import pandas as pd
import seaborn as sns
from eat.io import hops, util
from eat.hops import util as hu
import matplotlib.pyplot as plt
from eat.inspect import utils as ut
from eat.inspect import closures as cl

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

def plot_amp_days(data,sour, bars_on=False,logscale=True,polarizations=['LL','RR'], bands=['lo','hi'],palette_dict=palette_dict):

    data = data[list(map(lambda x: x in polarizations, data.polarization))]
    data = data[list(map(lambda x: x in bands, data.band))]
    foo = data[data.source==sour].copy()

    nscan=np.shape(foo)[0]
    nbase = len(foo.baseline.unique())
    print(sour)
    print("{} detections on {} baselines".format(nscan,nbase))
    print("median snr {}".format(np.median(foo.snr)))
    print("=========================================")

    bins = np.logspace(0.5*np.log10(np.min(foo.snr)),1.1*np.log10(np.max(foo.snr)),np.sqrt(nscan))
    plt.hist(foo[~foo.baseline.str.contains('A')].snr,bins=bins,histtype='step',linewidth=2,density=False,label='non-ALMA baselines')
    plt.hist(foo[foo.baseline.str.contains('A')].snr,bins=bins,histtype='step',linewidth=2,density=False,label='ALMA baselines')
    plt.xscale('log')
    plt.xlabel('snr')
    plt.ylabel('detections')
    plt.grid()
    plt.title(sour)
    plt.legend()
    plt.show()

    
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
                    ax[couR,couC].set_xlabel('UV distance')
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
                ax[couC].set_xlabel('UV distance')
                ax[couC].set_ylabel('amplitudes')
                if logscale==True:
                    ax[couC].set_yscale("log", nonposy='clip')
                ax[couC].set_ylim(ymin=0.1*np.min(foo.amp))
                ax[couC].set_ylim(ymax=1.1*np.max(foo.amp))
                couP+=1
        plt.tight_layout()
    plt.show()


def compare_uvf_apc(apc_sc,uvf_sc,by_scan_id=False, polarizations=['LL','RR'],bands=['lo','hi']):
    
    uvf_sc = uvf_sc[list(map(lambda x: x in polarizations, uvf_sc.polarization))]
    foo_uvf = uvf_sc[list(map(lambda x: x in bands, uvf_sc.band))].copy()
    
    apc_sc = apc_sc[list(map(lambda x: x in polarizations, apc_sc.polarization))]
    foo_apc = apc_sc[list(map(lambda x: x in bands, apc_sc.band))].copy()
    
    uvf,apc=ut.match_frames(foo_uvf.copy(),foo_apc.copy(),['source','band','polarization','scan_id','baseline'])
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
        if by_scan_id==False:
            sg=sns.lmplot(data=data,x='var_before',y='var_after',hue='source',col='baseline',fit_reg=False,sharey=False,sharex=False,
                     palette=dict_col_sour, scatter_kws={'alpha':0.3})
        else:
            sg=sns.lmplot(data=data,x='scan_id',y='after2before',hue='source',col='baseline',fit_reg=False,sharey=False,sharex=False,
                     palette=dict_col_sour, scatter_kws={'alpha':0.3})
        ax1 = sg.fig.axes[0]
        hm1y=ax1.get_ylim()
        hm1x=ax1.get_xlim()
        
        ax2 = sg.fig.axes[1]
        hm2x=ax2.get_xlim()
        hm2y=ax2.get_ylim()
        
        if by_scan_id==False:
            ax1.plot([0,max_plot],[0,max_plot],'k--')
            ax2.plot([0,max_plot],[0,max_plot],'k--')
        else:
            ax1.plot(hm1x,[1,1],'k--')
            ax2.plot(hm1x,[1,1],'k--')
        
        ax1.set_ylim(hm1y)
        ax1.set_xlim(hm1x)
        ax2.set_ylim(hm2y)
        ax2.set_xlim(hm2x)

        for lh in sg._legend.legendHandles: 
            lh.set_alpha(1)
            lh._sizes = [50] 
        plt.show()

    return apc[['datetime','source','expt_no','scan_id','polarization','band','baseline','var_before','var_after','after2before']].copy()


def compare_coherence_time(coh0,incoh0,dict_col_sour=dict_col_sour,snr_cut=0, polarizations=['LL','RR'],bands=['lo','hi']):

    coh0 = coh0[list(map(lambda x: x in polarizations, coh0.polarization))]
    coh00 = coh0[list(map(lambda x: x in bands, coh0.band))].copy()
    
    incoh0 = incoh0[list(map(lambda x: x in polarizations, incoh0.polarization))]
    incoh00 = incoh0[list(map(lambda x: x in bands, incoh0.band))].copy()

    coh,incoh = ut.match_frames(coh00[coh00.snr>snr_cut].copy(),incoh00[incoh00.snr>snr_cut].copy(),['scan_id','baseline','polarization','band'])
    #print(np.shape(coh),np.shape(incoh))
    coh['amp_coh'] = np.sqrt(np.maximum(0,coh['amp']**2 - coh['sigma']**2))
    coh['amp_incoh'] = incoh['amp']
    coh['sigma_coh'] = coh['sigma']
    coh['sigma_incoh'] = incoh['sigma']
    coh['coh2incoh'] = coh['amp_coh']/coh['amp_incoh']
    coh['snr_coh'] = coh['snr']
    coh['snr_incoh'] = incoh['snr']
    data=coh.copy()
    sns.set_style('whitegrid')
    baseL = sorted(list(data.baseline.unique()))
    ncol = int(np.ceil(len(baseL)/2))
    num_base = (np.ceil((np.asarray(range(2*ncol))+0.1)/2))
    diccol=dict(zip(baseL,num_base))
    coh['basenum'] = list(map(lambda x: diccol[x],coh.baseline))

    for basenum in sorted(coh.basenum.unique()):
        data=coh[coh.basenum==basenum].copy()
        max_plot = np.maximum(np.max(data.amp_coh),np.max(data.amp_incoh))
        min_plot = np.minimum(np.min(data.amp_coh),np.min(data.amp_incoh))
        #sg=sns.lmplot(data=data,x='amp_coh',y='amp_incoh',hue='source',col='baseline',fit_reg=False,sharey=False,sharex=False,
        #             palette=dict_col_sour)
        sg=sns.lmplot(data=data,x='scan_id',y='coh2incoh',hue='source',col='baseline',fit_reg=False,sharey=False,sharex=False,
                    palette=dict_col_sour, scatter_kws={'alpha':0.3})
        ax1 = sg.fig.axes[0]
        hm1y=ax1.get_ylim()
        hm1x=ax1.get_xlim()
        ax1.plot(hm1x,[1,1],'k--')
        #ax1.set_ylim(hm1y)
        #ax1.set_xlim(hm1x)
        ax2 = sg.fig.axes[1]
        hm2x=ax2.get_xlim()
        hm2y=ax2.get_ylim()
        ax2.plot(hm2x,[1,1],'k--')
        #ax2.set_ylim(hm2y)
        #ax2.set_xlim(hm2x)
        for lh in sg._legend.legendHandles: 
            lh.set_alpha(1)
            lh._sizes = [50] 
        plt.show()

    return coh[['datetime','source','expt_no','scan_id','polarization','band','baseline','amp_coh','amp_incoh','coh2incoh','snr_coh','snr_incoh']].copy()


def bandpass_amplitude_consistency(data0,xmax=10,by_what='source'):

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

    med=np.median((data['amp_diff']))
    mad_abs=np.median(np.abs(data['amp_diff']))
    mad_rel=np.median(np.abs(data['rel_diff']))/0.67449
    rangey = plt.ylim()
    rangex = plt.xlim()
    plt.text(0.5*rangex[1], 0.9*rangey[1], "MED: %4.3f" % med , bbox=dict(facecolor='white', alpha=1.))
    plt.text(0.5*rangex[1], 0.8*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
    plt.text(0.5*rangex[1], 0.7*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))

    plt.show()

    sourceL = sorted(list(data.source.unique()))
    whatL = sorted(list(data[by_what].unique()))
    nplots=len(whatL)
    ncols=2
    nrows=int(np.ceil(nplots/ncols))
    fig, ax = plt.subplots(nrows,ncols,sharey='all',sharex='all',figsize=(ncols*7,nrows*5))

    for cou,what in enumerate(whatL):
        nbins = int(np.sqrt(np.shape(data[data[by_what]==what])[0]))
        bins = np.linspace(-xmax,xmax,nbins)
        nrowL = int(np.floor(cou/2))
        ncolL = cou%ncols
        ax[nrowL,ncolL].hist(data[data[by_what]==what]['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
        ax[nrowL,ncolL].plot(x,np.exp(-(x)**2/2)/np.sqrt(2.*np.pi),'k')
        ax[nrowL,ncolL].grid()
        ax[nrowL,ncolL].axvline(0,color='k')
        ax[nrowL,ncolL].set_xlabel('(LO-HI)/(thermal error)')
        ax[nrowL,ncolL].set_title(what)
        med=np.median((data[data[by_what]==what]['amp_diff']))
        mad_abs=np.median(np.abs(data[data[by_what]==what]['amp_diff']))
        mad_rel=np.median(np.abs(data[data[by_what]==what]['rel_diff']))/0.67449
        rangey = ax[nrowL,ncolL].get_ylim()
        rangex = ax[nrowL,ncolL].get_xlim()
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.9*rangey[1], "MED: %4.3f" % med , bbox=dict(facecolor='white', alpha=1.))
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.8*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.7*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))

    plt.show()
    return data


def bandpass_cphase_consistency(data0,xmax=10,by_what='source'):

    data_lo, data_hi = ut.match_frames(data0[data0.band=='lo'].copy(),data0[data0.band=='hi'].copy(),['scan_id','triangle','polarization'])
    data = data_lo.copy()
    data['cphase_lo'] = data_lo['cphase']
    data['cphase_hi'] = data_hi['cphase']
    data['sigma_lo'] = data_lo['sigmaCP']
    data['sigma_hi'] = data_hi['sigmaCP']
    data['sigma'] = np.sqrt(data['sigma_lo']**2 + data['sigma_hi']**2)
    data['cphase_diff'] = np.angle(np.exp(1j*(data['cphase_lo'] - data['cphase_hi'])*np.pi/180))*180./np.pi
    data['rel_diff'] = np.asarray(data['cphase_diff'])/np.asarray(data['sigma'])

    nbins = int(np.sqrt(np.shape(data)[0]))
    bins = np.linspace(-xmax,xmax,nbins)
    x=np.linspace(-xmax,xmax,128)
    plt.hist(data['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
    plt.grid()
    plt.plot(x,np.exp(-(x)**2/2)/np.sqrt(2.*np.pi),'k')
    plt.axvline(0,color='k')
    plt.xlabel('(LO-HI)/(thermal error)')
    plt.title('All data')
    mad_abs=np.median(np.abs(data['cphase_diff']))
    mad_rel=np.median(np.abs(data['rel_diff']))/0.67449
    rangey = plt.ylim()
    rangex = plt.xlim()
    plt.text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f deg" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
    plt.text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))

    plt.show()

    sourceL = sorted(list(data.source.unique()))
    whatL = sorted(list(data[by_what].unique()))
    nplots=len(whatL)
    ncols=2
    nrows=int(np.ceil(nplots/ncols))
    fig, ax = plt.subplots(nrows,ncols,sharey='all',sharex='all',figsize=(ncols*7,nrows*5))

    
    #for cou,sour in enumerate(sourceL):
    for cou,what in enumerate(whatL):
        nbins = int(np.sqrt(np.shape(data[data[by_what]==what])[0]))
        bins = np.linspace(-xmax,xmax,nbins)
        nrowL = int(np.floor(cou/2))
        ncolL = cou%ncols
        ax[nrowL,ncolL].hist(data[data[by_what]==what]['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
        ax[nrowL,ncolL].plot(x,np.exp(-(x)**2/2)/np.sqrt(2.*np.pi),'k')
        ax[nrowL,ncolL].grid()
        ax[nrowL,ncolL].axvline(0,color='k')
        ax[nrowL,ncolL].set_xlabel('(LO-HI)/(thermal error)')
        ax[nrowL,ncolL].set_title(what)
        mad_abs=np.median(np.abs(data[data[by_what]==what]['cphase_diff']))
        mad_rel=np.median(np.abs(data[data[by_what]==what]['rel_diff']))/0.67449
        rangey = ax[nrowL,ncolL].get_ylim()
        rangex = ax[nrowL,ncolL].get_xlim()
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f deg" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))
    plt.show()
    return data


def polar_cphase_consistency(data0,xmax=10,by_what='source'):

    data_rr, data_ll = ut.match_frames(data0[data0.polarization=='LL'].copy(),data0[data0.polarization=='RR'].copy(),['scan_id','triangle','band'])
    data = data_rr.copy()
    data['cphase_rr'] = data_rr['cphase']
    data['cphase_ll'] = data_ll['cphase']
    data['sigma_rr'] = data_rr['sigmaCP']
    data['sigma_ll'] = data_ll['sigmaCP']
    data['sigma'] = np.sqrt(data['sigma_ll']**2 + data['sigma_rr']**2)
    data['cphase_diff'] = np.angle(np.exp(1j*(data['cphase_rr'] - data['cphase_ll'])*np.pi/180))*180./np.pi
    data['rel_diff'] = np.asarray(data['cphase_diff'])/np.asarray(data['sigma'])

    nbins = int(np.sqrt(np.shape(data)[0]))
    bins = np.linspace(-xmax,xmax,nbins)
    x=np.linspace(-xmax,xmax,128)
    plt.hist(data['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
    plt.grid()
    plt.plot(x,np.exp(-(x)**2/2)/np.sqrt(2.*np.pi),'k')
    plt.axvline(0,color='k')
    plt.xlabel('(RR-LL)/(thermal error)')
    plt.title('All data')
    mad_abs=np.median(np.abs(data['cphase_diff']))
    mad_rel=np.median(np.abs(data['rel_diff']))/0.67449
    rangey = plt.ylim()
    rangex = plt.xlim()
    plt.text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f deg" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
    plt.text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))

    plt.show()

    sourceL = sorted(list(data.source.unique()))
    whatL = sorted(list(data[by_what].unique()))
    nplots=len(whatL)
    ncols=2
    nrows=int(np.ceil(nplots/ncols))
    fig, ax = plt.subplots(nrows,ncols,sharey='all',sharex='all',figsize=(ncols*7,nrows*5))

    
    #for cou,sour in enumerate(sourceL):
    for cou,what in enumerate(whatL):
        nbins = int(np.sqrt(np.shape(data[data[by_what]==what])[0]))
        bins = np.linspace(-xmax,xmax,nbins)
        nrowL = int(np.floor(cou/2))
        ncolL = cou%ncols
        ax[nrowL,ncolL].hist(data[data[by_what]==what]['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
        ax[nrowL,ncolL].plot(x,np.exp(-(x)**2/2)/np.sqrt(2.*np.pi),'k')
        ax[nrowL,ncolL].grid()
        ax[nrowL,ncolL].axvline(0,color='k')
        ax[nrowL,ncolL].set_xlabel('(RR-LL)/(thermal error)')
        ax[nrowL,ncolL].set_title(what)
        mad_abs=np.median(np.abs(data[data[by_what]==what]['cphase_diff']))
        mad_rel=np.median(np.abs(data[data[by_what]==what]['rel_diff']))/0.67449
        rangey = ax[nrowL,ncolL].get_ylim()
        rangex = ax[nrowL,ncolL].get_xlim()
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f deg" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))
    plt.show()
    return data



def bandpass_amplitude_rel_consistency(data0,xmax=2.,by_what='source'):

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

    med=np.median((data['amp_diff']))
    mad_abs=np.median(np.abs(data['amp_diff']))
    mad_rel=np.median(np.abs(data['rel_diff']))/0.67449
    rangey = plt.ylim()
    rangex = plt.xlim()
    plt.text(0.5*rangex[1], 0.9*rangey[1], "MED: %4.3f" % med , bbox=dict(facecolor='white', alpha=1.))
    plt.text(0.5*rangex[1], 0.8*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
    plt.text(0.5*rangex[1], 0.7*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))

    plt.show()

    sourceL = sorted(list(data.source.unique()))
    whatL = sorted(list(data[by_what].unique()))
    nplots=len(whatL)
    ncols=2
    nrows=int(np.ceil(nplots/ncols))
    fig, ax = plt.subplots(nrows,ncols,sharey='all',sharex='all',figsize=(ncols*7,nrows*5))

    for cou,what in enumerate(whatL):
    #for cou,sour in enumerate(sourceL):
        nbins = int(np.sqrt(np.shape(data[data[by_what]==what])[0]))
        bins = np.linspace(-xmax,xmax,nbins)
        nrowL = int(np.floor(cou/2))
        ncolL = cou%ncols
        ax[nrowL,ncolL].hist(data[data[by_what]==what]['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
        ax[nrowL,ncolL].grid()
        ax[nrowL,ncolL].axvline(0,color='k')
        ax[nrowL,ncolL].set_xlabel('2*(LO-HI)/(L0 + HI)')
        #ax[nrowL,ncolL].set_xlim([-xmax,xmax])
        ax[nrowL,ncolL].set_title(what)

        med=np.median((data[data[by_what]==what]['amp_diff']))
        mad_abs=np.median(np.abs(data[data[by_what]==what]['amp_diff']))
        mad_rel=np.median(np.abs(data[data[by_what]==what]['rel_diff']))/0.67449
        rangey = ax[nrowL,ncolL].get_ylim()
        rangex = ax[nrowL,ncolL].get_xlim()
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.9*rangey[1], "MED: %4.3f" % med , bbox=dict(facecolor='white', alpha=1.))
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.8*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.7*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))

    plt.show()
    return data


def polar_amplitude_consistency(data0,xmax=10,by_what='source'):

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

    med=np.median((data['amp_diff']))
    mad_abs=np.median(np.abs(data['amp_diff']))
    mad_rel=np.median(np.abs(data['rel_diff']))/0.67449
    rangey = plt.ylim()
    rangex = plt.xlim()
    plt.text(0.5*rangex[1], 0.9*rangey[1], "MED: %4.3f" % med , bbox=dict(facecolor='white', alpha=1.))
    plt.text(0.5*rangex[1], 0.8*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
    plt.text(0.5*rangex[1], 0.7*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))

    plt.show()

    sourceL = sorted(list(data.source.unique()))
    whatL = sorted(list(data[by_what].unique()))
    nplots=len(whatL)
    ncols=2
    nrows=int(np.ceil(nplots/ncols))
    fig, ax = plt.subplots(nrows,ncols,sharey='all',sharex='all',figsize=(ncols*7,nrows*5))

    for cou,what in enumerate(whatL):
    #for cou,sour in enumerate(sourceL):
        nbins = int(np.sqrt(np.shape(data[data[by_what]==what])[0]))
        bins = np.linspace(-xmax,xmax,nbins)
        nrowL = int(np.floor(cou/2))
        ncolL = cou%ncols
        ax[nrowL,ncolL].hist(data[data[by_what]==what]['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
        ax[nrowL,ncolL].plot(x,np.exp(-(x)**2/2)/np.sqrt(2.*np.pi),'k')
        ax[nrowL,ncolL].grid()
        ax[nrowL,ncolL].axvline(0,color='k')
        ax[nrowL,ncolL].set_xlabel('(RR-LL)/(thermal error)')
        ax[nrowL,ncolL].set_title(what)

        med=np.median((data[data[by_what]==what]['amp_diff']))
        mad_abs=np.median(np.abs(data[data[by_what]==what]['amp_diff']))
        mad_rel=np.median(np.abs(data[data[by_what]==what]['rel_diff']))/0.67449
        rangey = ax[nrowL,ncolL].get_ylim()
        rangex = ax[nrowL,ncolL].get_xlim()
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.9*rangey[1], "MED: %4.3f" % med , bbox=dict(facecolor='white', alpha=1.))
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.8*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.7*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))

        
    plt.show()
    return data


def polar_amplitude_rel_consistency(data0,xmax=2.,by_what='source'):

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

    med=np.median((data['amp_diff']))
    mad_abs=np.median(np.abs(data['amp_diff']))
    mad_rel=np.median(np.abs(data['rel_diff']))/0.67449
    rangey = plt.ylim()
    rangex = plt.xlim()
    plt.text(0.5*rangex[1], 0.9*rangey[1], "MED: %4.3f" % med , bbox=dict(facecolor='white', alpha=1.))
    plt.text(0.5*rangex[1], 0.8*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
    plt.text(0.5*rangex[1], 0.7*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))

    plt.show()

    sourceL = sorted(list(data.source.unique()))
    whatL = sorted(list(data[by_what].unique()))
    nplots=len(whatL)
    ncols=2
    nrows=int(np.ceil(nplots/ncols))
    fig, ax = plt.subplots(nrows,ncols,sharey='all',sharex='all',figsize=(ncols*7,nrows*5))

    for cou,what in enumerate(whatL):    
    #for cou,sour in enumerate(sourceL):
        nbins = int(np.sqrt(np.shape(data[data[by_what]==what])[0]))
        bins = np.linspace(-xmax,xmax,nbins)
        nrowL = int(np.floor(cou/2))
        ncolL = cou%ncols
        ax[nrowL,ncolL].hist(data[data[by_what]==what]['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
        ax[nrowL,ncolL].grid()
        ax[nrowL,ncolL].axvline(0,color='k')
        ax[nrowL,ncolL].set_xlabel('2*(RR-LL)/(RR+LL)')
        ax[nrowL,ncolL].set_title(what)

        med=np.median((data[data[by_what]==what]['amp_diff']))
        mad_abs=np.median(np.abs(data[data[by_what]==what]['amp_diff']))
        mad_rel=np.median(np.abs(data[data[by_what]==what]['rel_diff']))/0.67449
        rangey = ax[nrowL,ncolL].get_ylim()
        rangex = ax[nrowL,ncolL].get_xlim()
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.9*rangey[1], "MED: %4.3f" % med , bbox=dict(facecolor='white', alpha=1.))
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.8*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.7*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))

    plt.show()
    return data

def bandpass_lcamp_consistency(data0,xmax=10,by_what='source'):

    data_lo, data_hi = ut.match_frames(data0[data0.band=='lo'].copy(),data0[data0.band=='hi'].copy(),['scan_id','quadrangle','polarization'])
    data = data_lo.copy()
    data['lcamp_lo'] = data_lo['camp']
    data['lcamp_hi'] = data_hi['camp']
    data['sigma_lo'] = data_lo['sigmaCA']
    data['sigma_hi'] = data_hi['sigmaCA']
    data['sigma'] = np.sqrt(data['sigma_lo']**2 + data['sigma_hi']**2)
    data['lcamp_diff'] = data['lcamp_lo'] - data['lcamp_hi']
    data['rel_diff'] = np.asarray(data['lcamp_diff'])/np.asarray(data['sigma'])

    nbins = int(np.sqrt(np.shape(data)[0]))
    bins = np.linspace(-xmax,xmax,nbins)
    x=np.linspace(-xmax,xmax,128)
    plt.hist(data['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
    plt.grid()
    plt.plot(x,np.exp(-(x)**2/2)/np.sqrt(2.*np.pi),'k')
    plt.axvline(0,color='k')
    plt.xlabel('(LO-HI)/(thermal error)')
    plt.title('All data')
    mad_abs=np.median(np.abs(data['lcamp_diff']))
    mad_rel=np.median(np.abs(data['rel_diff']))/0.67449
    rangey = plt.ylim()
    rangex = plt.xlim()
    plt.text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
    plt.text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))

    plt.show()

    sourceL = sorted(list(data.source.unique()))
    whatL = sorted(list(data[by_what].unique()))
    nplots=len(whatL)
    ncols=2
    nrows=int(np.ceil(nplots/ncols))
    fig, ax = plt.subplots(nrows,ncols,sharey='all',sharex='all',figsize=(ncols*7,nrows*5))
    
    #for cou,sour in enumerate(sourceL):
    for cou,what in enumerate(whatL):
        nbins = int(np.sqrt(np.shape(data[data[by_what]==what])[0]))
        bins = np.linspace(-xmax,xmax,nbins)
        nrowL = int(np.floor(cou/2))
        ncolL = cou%ncols
        ax[nrowL,ncolL].hist(data[data[by_what]==what]['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
        ax[nrowL,ncolL].plot(x,np.exp(-(x)**2/2)/np.sqrt(2.*np.pi),'k')
        ax[nrowL,ncolL].grid()
        ax[nrowL,ncolL].axvline(0,color='k')
        ax[nrowL,ncolL].set_xlabel('(LO-HI)/(thermal error)')
        ax[nrowL,ncolL].set_title(what)
        mad_abs=np.median(np.abs(data[data[by_what]==what]['lcamp_diff']))
        mad_rel=np.median(np.abs(data[data[by_what]==what]['rel_diff']))/0.67449
        rangey = ax[nrowL,ncolL].get_ylim()
        rangex = ax[nrowL,ncolL].get_xlim()
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))
    plt.show()
    return data

def polar_lcamp_consistency(data0,xmax=10,by_what='source'):

    data_rr, data_ll = ut.match_frames(data0[data0.polarization=='LL'].copy(),data0[data0.polarization=='RR'].copy(),['scan_id','quadrangle','band'])
    data = data_rr.copy()
    data['lcamp_rr'] = data_rr['camp']
    data['lcamp_ll'] = data_ll['camp']
    data['sigma_rr'] = data_rr['sigmaCA']
    data['sigma_ll'] = data_ll['sigmaCA']
    data['sigma'] = np.sqrt(data['sigma_rr']**2 + data['sigma_ll']**2)
    data['lcamp_diff'] = data['lcamp_rr'] - data['lcamp_ll']
    data['rel_diff'] = np.asarray(data['lcamp_diff'])/np.asarray(data['sigma'])

    nbins = int(np.sqrt(np.shape(data)[0]))
    bins = np.linspace(-xmax,xmax,nbins)
    x=np.linspace(-xmax,xmax,128)
    plt.hist(data['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
    plt.grid()
    plt.plot(x,np.exp(-(x)**2/2)/np.sqrt(2.*np.pi),'k')
    plt.axvline(0,color='k')
    plt.xlabel('(RR-LL)/(thermal error)')
    plt.title('All data')
    mad_abs=np.median(np.abs(data['lcamp_diff']))
    mad_rel=np.median(np.abs(data['rel_diff']))/0.67449
    rangey = plt.ylim()
    rangex = plt.xlim()
    plt.text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
    plt.text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))

    plt.show()

    sourceL = sorted(list(data.source.unique()))
    whatL = sorted(list(data[by_what].unique()))
    nplots=len(whatL)
    ncols=2
    nrows=int(np.ceil(nplots/ncols))
    fig, ax = plt.subplots(nrows,ncols,sharey='all',sharex='all',figsize=(ncols*7,nrows*5))
    
    #for cou,sour in enumerate(sourceL):
    for cou,what in enumerate(whatL):
        nbins = int(np.sqrt(np.shape(data[data[by_what]==what])[0]))
        bins = np.linspace(-xmax,xmax,nbins)
        nrowL = int(np.floor(cou/2))
        ncolL = cou%ncols
        ax[nrowL,ncolL].hist(data[data[by_what]==what]['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
        ax[nrowL,ncolL].plot(x,np.exp(-(x)**2/2)/np.sqrt(2.*np.pi),'k')
        ax[nrowL,ncolL].grid()
        ax[nrowL,ncolL].axvline(0,color='k')
        ax[nrowL,ncolL].set_xlabel('(RR-LL)/(thermal error)')
        ax[nrowL,ncolL].set_title(what)
        mad_abs=np.median(np.abs(data[data[by_what]==what]['lcamp_diff']))
        mad_rel=np.median(np.abs(data[data[by_what]==what]['rel_diff']))/0.67449
        rangey = ax[nrowL,ncolL].get_ylim()
        rangex = ax[nrowL,ncolL].get_xlim()
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))
    plt.show()
    return data


def trivial_lcamp(data0,xmax=10,whichB='all',by_what='source'):

    data = cl.only_trivial_quadrangles_str(data0, whichB=whichB)
    data=data.copy()
    data['rel_diff'] = np.asarray(data['camp'])/np.asarray(data['sigmaCA'])

    nbins = int(np.sqrt(np.shape(data)[0]))
    bins = np.linspace(-xmax,xmax,nbins)
    x=np.linspace(-xmax,xmax,128)
    plt.hist(data['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
    plt.grid()
    plt.plot(x,np.exp(-(x)**2/2)/np.sqrt(2.*np.pi),'k')
    plt.axvline(0,color='k')
    plt.xlabel('(TCA)/(thermal error)')
    plt.title('All data')
    mad_abs=np.median(np.abs(data['camp']))
    mad_rel=np.median(np.abs(data['rel_diff']))/0.67449
    rangey = plt.ylim()
    rangex = plt.xlim()
    plt.text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
    plt.text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))

    plt.show()

    sourceL = sorted(list(data.source.unique()))
    whatL = sorted(list(data[by_what].unique()))
    nplots=len(whatL)
    ncols=2
    nrows=int(np.ceil(nplots/ncols))
    fig, ax = plt.subplots(nrows,ncols,sharey='all',sharex='all',figsize=(ncols*7,nrows*5))
    
    #for cou,sour in enumerate(sourceL):
    for cou,what in enumerate(whatL):
        nbins = int(np.sqrt(np.shape(data[data[by_what]==what])[0]))
        bins = np.linspace(-xmax,xmax,nbins)
        nrowL = int(np.floor(cou/2))
        ncolL = cou%ncols
        ax[nrowL,ncolL].hist(data[data[by_what]==what]['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
        ax[nrowL,ncolL].plot(x,np.exp(-(x)**2/2)/np.sqrt(2.*np.pi),'k')
        ax[nrowL,ncolL].grid()
        ax[nrowL,ncolL].axvline(0,color='k')
        ax[nrowL,ncolL].set_xlabel('(TCA)/(thermal error)')
        ax[nrowL,ncolL].set_title(what)
        mad_abs=np.median(np.abs(data[data[by_what]==what]['camp']))
        mad_rel=np.median(np.abs(data[data[by_what]==what]['rel_diff']))/0.67449
        rangey = ax[nrowL,ncolL].get_ylim()
        rangex = ax[nrowL,ncolL].get_xlim()
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))
    plt.show()
    return data


def trivial_cphase(data0,xmax=10,whichB='all',by_what='source'):

    data = cl.only_trivial_triangles(data0, whichB=whichB)
    data=data.copy()
    data['rel_diff'] = np.asarray(data['cphase'])/np.asarray(data['sigmaCP'])

    nbins = int(np.sqrt(np.shape(data)[0]))
    bins = np.linspace(-xmax,xmax,nbins)
    x=np.linspace(-xmax,xmax,128)
    plt.hist(data['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
    plt.grid()
    plt.plot(x,np.exp(-(x)**2/2)/np.sqrt(2.*np.pi),'k')
    plt.axvline(0,color='k')
    plt.xlabel('(TCP)/(thermal error)')
    plt.title('All data')
    mad_abs=np.median(np.abs(data['cphase']))
    mad_rel=np.median(np.abs(data['rel_diff']))/0.67449
    rangey = plt.ylim()
    rangex = plt.xlim()
    plt.text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f deg" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
    plt.text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))

    plt.show()

    sourceL = sorted(list(data.source.unique()))
    whatL = sorted(list(data[by_what].unique()))
    nplots=len(whatL)
    ncols=2
    nrows=int(np.ceil(nplots/ncols))
    fig, ax = plt.subplots(nrows,ncols,sharey='all',sharex='all',figsize=(ncols*7,nrows*5))

    
    #for cou,sour in enumerate(sourceL):
    for cou,what in enumerate(whatL):
        nbins = int(np.sqrt(np.shape(data[data[by_what]==what])[0]))
        bins = np.linspace(-xmax,xmax,nbins)
        nrowL = int(np.floor(cou/2))
        ncolL = cou%ncols
        ax[nrowL,ncolL].hist(data[data[by_what]==what]['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
        ax[nrowL,ncolL].plot(x,np.exp(-(x)**2/2)/np.sqrt(2.*np.pi),'k')
        ax[nrowL,ncolL].grid()
        ax[nrowL,ncolL].axvline(0,color='k')
        ax[nrowL,ncolL].set_xlabel('(TCP)/(thermal error)')
        ax[nrowL,ncolL].set_title(what)
        mad_abs=np.median(np.abs(data[data[by_what]==what]['cphase']))
        mad_rel=np.median(np.abs(data[data[by_what]==what]['rel_diff']))/0.67449
        rangey = ax[nrowL,ncolL].get_ylim()
        rangex = ax[nrowL,ncolL].get_xlim()
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f deg" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))
    plt.show()
    return data

############################
# PIPELINE TESTS
############################


def pipe_amp(pipe1,pipe2,xmax=10.,by_what='source',norm_thermal=True):
    if 'sigma' not in pipe1.columns:
        pipe1['sigma'] = pipe1['amp']/pipe1['snr']
    if 'sigma' not in pipe2.columns:
        pipe2['sigma'] = pipe2['amp']/pipe2['snr']
    p1, p2 = ut.match_frames(pipe1.copy(),pipe2.copy(),['scan_id','baseline','band','polarization'])
    data = p1.copy()
    data['amp_p1'] = p1['amp']
    data['amp_p2'] = p2['amp']
    data['sigma_p1'] = p1['sigma']
    data['sigma_p2'] = p2['sigma']
    data['sigma'] = np.sqrt(data['sigma_p1']**2 + data['sigma_p2']**2)
    data['amp_diff'] = np.asarray(data['amp_p1']) - np.asarray(data['amp_p2'])
    data['amp_mean'] = 0.5*(data['amp_p1']+data['amp_p2'])

    if norm_thermal==True:
        data['rel_diff'] = np.asarray(data['amp_diff'])/np.asarray(data['sigma'])
        lab='(P1-P2)/(thermal error)'
    else:
        data['rel_diff'] = data['amp_diff']/data['amp_mean']
        lab='0.5*(P1-P2)/(P1+P2)'

    nbins = int(np.sqrt(np.shape(data)[0]))
    bins = np.linspace(-xmax,xmax,nbins)
    x=np.linspace(-xmax,xmax,128)
    plt.hist(data['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
    plt.grid()
    plt.plot(x,np.exp(-(x)**2/2)/np.sqrt(2.*np.pi),'k')
    plt.axvline(0,color='k')
    plt.xlabel(lab)
    plt.title('All data')

    med=np.median((data['amp_diff']))
    mad_abs=np.median(np.abs(data['amp_diff']))
    mad_rel=np.median(np.abs(data['rel_diff']))/0.67449
    rangey = plt.ylim()
    rangex = plt.xlim()
    plt.text(0.5*rangex[1], 0.9*rangey[1], "MED: %4.3f" % med , bbox=dict(facecolor='white', alpha=1.))
    plt.text(0.5*rangex[1], 0.8*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
    plt.text(0.5*rangex[1], 0.7*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))

    plt.show()

    sourceL = sorted(list(data.source.unique()))
    whatL = sorted(list(data[by_what].unique()))
    nplots=len(whatL)
    ncols=2
    nrows=int(np.ceil(nplots/ncols))
    fig, ax = plt.subplots(nrows,ncols,sharey='all',sharex='all',figsize=(ncols*7,nrows*5))

    for cou,what in enumerate(whatL):
        nbins = int(np.sqrt(np.shape(data[data[by_what]==what])[0]))
        bins = np.linspace(-xmax,xmax,nbins)
        nrowL = int(np.floor(cou/2))
        ncolL = cou%ncols
        ax[nrowL,ncolL].hist(data[data[by_what]==what]['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
        ax[nrowL,ncolL].plot(x,np.exp(-(x)**2/2)/np.sqrt(2.*np.pi),'k')
        ax[nrowL,ncolL].grid()
        ax[nrowL,ncolL].axvline(0,color='k')
        ax[nrowL,ncolL].set_xlabel(lab)
        ax[nrowL,ncolL].set_title(what)
        med=np.median((data[data[by_what]==what]['amp_diff']))
        mad_abs=np.median(np.abs(data[data[by_what]==what]['amp_diff']))
        mad_rel=np.median(np.abs(data[data[by_what]==what]['rel_diff']))/0.67449
        rangey = ax[nrowL,ncolL].get_ylim()
        rangex = ax[nrowL,ncolL].get_xlim()
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.9*rangey[1], "MED: %4.3f" % med , bbox=dict(facecolor='white', alpha=1.))
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.8*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.7*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))

    plt.show()
    return data

def pipe_cphase(pipe1,pipe2,xmax=10.,by_what='source'):
    data_p1, data_p2 = ut.match_frames(pipe1.copy(),pipe2.copy(),['scan_id','triangle','band','polarization'])
    data = data_p1.copy()
    data['cphase_p1'] = data_p1['cphase']
    data['cphase_p2'] = data_p2['cphase']
    data['sigma_p1'] = data_p1['sigmaCP']
    data['sigma_p2'] = data_p2['sigmaCP']
    data['sigma'] = np.sqrt(data['sigma_p1']**2 + data['sigma_p2']**2)
    data['cphase_diff'] = np.angle(np.exp(1j*(data['cphase_p1'] - data['cphase_p2'])*np.pi/180))*180./np.pi
    data['rel_diff'] = np.asarray(data['cphase_diff'])/np.asarray(data['sigma'])

    nbins = int(np.sqrt(np.shape(data)[0]))
    bins = np.linspace(-xmax,xmax,nbins)
    x=np.linspace(-xmax,xmax,128)
    plt.hist(data['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
    plt.grid()
    plt.plot(x,np.exp(-(x)**2/2)/np.sqrt(2.*np.pi),'k')
    plt.axvline(0,color='k')
    plt.xlabel('(cp_P1-cp_P2)/(thermal error)')
    plt.title('All data')
    mad_abs=np.median(np.abs(data['cphase_diff']))
    mad_rel=np.median(np.abs(data['rel_diff']))/0.67449
    rangey = plt.ylim()
    rangex = plt.xlim()
    plt.text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f deg" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
    plt.text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))

    plt.show()

    sourceL = sorted(list(data.source.unique()))
    whatL = sorted(list(data[by_what].unique()))
    nplots=len(whatL)
    ncols=2
    nrows=int(np.ceil(nplots/ncols))
    fig, ax = plt.subplots(nrows,ncols,sharey='all',sharex='all',figsize=(ncols*7,nrows*5))

    
    #for cou,sour in enumerate(sourceL):
    for cou,what in enumerate(whatL):
        nbins = int(np.sqrt(np.shape(data[data[by_what]==what])[0]))
        bins = np.linspace(-xmax,xmax,nbins)
        nrowL = int(np.floor(cou/2))
        ncolL = cou%ncols
        ax[nrowL,ncolL].hist(data[data[by_what]==what]['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
        ax[nrowL,ncolL].plot(x,np.exp(-(x)**2/2)/np.sqrt(2.*np.pi),'k')
        ax[nrowL,ncolL].grid()
        ax[nrowL,ncolL].axvline(0,color='k')
        ax[nrowL,ncolL].set_xlabel('(cp_P1-cp_P2)/(thermal error)')
        ax[nrowL,ncolL].set_title(what)
        mad_abs=np.median(np.abs(data[data[by_what]==what]['cphase_diff']))
        mad_rel=np.median(np.abs(data[data[by_what]==what]['rel_diff']))/0.67449
        rangey = ax[nrowL,ncolL].get_ylim()
        rangex = ax[nrowL,ncolL].get_xlim()
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f deg" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))
    plt.show()
    return data

def pipe_lcamp(pipe1,pipe2,xmax=10.,by_what='source'):

    data_p1, data_p2 = ut.match_frames(pipe1.copy(),pipe2.copy(),['scan_id','quadrangle','polarization','band'])
    data = data_p1.copy()
    data['lcamp_p1'] = data_p1['camp']
    data['lcamp_p2'] = data_p2['camp']
    data['sigma_p1'] = data_p1['sigmaCA']
    data['sigma_p2'] = data_p2['sigmaCA']
    data['sigma'] = np.sqrt(data['sigma_p1']**2 + data['sigma_p2']**2)
    data['lcamp_diff'] = data['lcamp_p1'] - data['lcamp_p2']
    data['rel_diff'] = np.asarray(data['lcamp_diff'])/np.asarray(data['sigma'])

    nbins = int(np.sqrt(np.shape(data)[0]))
    bins = np.linspace(-xmax,xmax,nbins)
    x=np.linspace(-xmax,xmax,128)
    plt.hist(data['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
    plt.grid()
    plt.plot(x,np.exp(-(x)**2/2)/np.sqrt(2.*np.pi),'k')
    plt.axvline(0,color='k')
    plt.xlabel('(lca_P1 - lca_P2)/(thermal error)')
    plt.title('All data')
    mad_abs=np.median(np.abs(data['lcamp_diff']))
    mad_rel=np.median(np.abs(data['rel_diff']))/0.67449
    rangey = plt.ylim()
    rangex = plt.xlim()
    plt.text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
    plt.text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))
    plt.show()

    sourceL = sorted(list(data.source.unique()))
    whatL = sorted(list(data[by_what].unique()))
    nplots=len(whatL)
    ncols=2
    nrows=int(np.ceil(nplots/ncols))
    fig, ax = plt.subplots(nrows,ncols,sharey='all',sharex='all',figsize=(ncols*7,nrows*5))
    
    #for cou,sour in enumerate(sourceL):
    for cou,what in enumerate(whatL):
        nbins = int(np.sqrt(np.shape(data[data[by_what]==what])[0]))
        bins = np.linspace(-xmax,xmax,nbins)
        nrowL = int(np.floor(cou/2))
        ncolL = cou%ncols
        ax[nrowL,ncolL].hist(data[data[by_what]==what]['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
        ax[nrowL,ncolL].plot(x,np.exp(-(x)**2/2)/np.sqrt(2.*np.pi),'k')
        ax[nrowL,ncolL].grid()
        ax[nrowL,ncolL].axvline(0,color='k')
        ax[nrowL,ncolL].set_xlabel('(lca_P1 - lca_P2)/(thermal error)')
        ax[nrowL,ncolL].set_title(what)
        mad_abs=np.median(np.abs(data[data[by_what]==what]['lcamp_diff']))
        mad_rel=np.median(np.abs(data[data[by_what]==what]['rel_diff']))/0.67449
        rangey = ax[nrowL,ncolL].get_ylim()
        rangex = ax[nrowL,ncolL].get_xlim()
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
        ax[nrowL,ncolL].text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))
    plt.show()
    return data
