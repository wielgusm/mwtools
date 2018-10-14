from __future__ import print_function
from __future__ import division
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
     '3C273': 'lime',
     '3C279': 'orange',
     '3C454.3': 'cyan',
     '3C84': 'gold',
     'BLLAC': 'magenta',
     'CENA': 'darkgreen',
     'CTA102': 'yellow',
     'J0006-0623': 'lightblue', 'J0006-06': 'lightblue',
     'J0132-1654': 'olivedrab','J0132-16': 'olivedrab',
     'J1733-1304': 'salmon','J1733-13': 'salmon',
     'J1924-2914': 'saddlebrown','J1924-29': 'saddlebrown', '1921-293': 'saddlebrown',
     'M87': 'k',
     'NGC1052': 'dodgerblue',
     'OJ287': 'blueviolet',
     'SGRA': 'red',
     'CYGX-3': 'silver'}

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

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

palette_dict_rev = {k.split('-')[1]+'-'+k.split('-')[0]:v for k, v in palette_dict.items()}
#palette_dict = {**palette_dict, **palette_dict_rev}
palette_dict = merge_two_dicts(palette_dict, palette_dict_rev)

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
    try:
        plt.hist(foo[~foo.baseline.str.contains('A')].snr,bins=bins,histtype='step',linewidth=2,density=False,label='non-ALMA baselines')
        plt.hist(foo[foo.baseline.str.contains('A')].snr,bins=bins,histtype='step',linewidth=2,density=False,label='ALMA baselines')
    except AttributeError:
        plt.hist(foo[~foo.baseline.str.contains('A')].snr,bins=bins,histtype='step',linewidth=2,normed=False,label='non-ALMA baselines')
        plt.hist(foo[foo.baseline.str.contains('A')].snr,bins=bins,histtype='step',linewidth=2,normed=False,label='ALMA baselines')

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


def plot_amp_single_day(data,sour,expt=3601, bars_on=False,logscale=True,polarizations=['LL','RR'], bands=['lo','hi'],palette_dict=palette_dict):

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
    try:
        plt.hist(foo[~foo.baseline.str.contains('A')].snr,bins=bins,histtype='step',linewidth=2,density=False,label='non-ALMA baselines')
        plt.hist(foo[foo.baseline.str.contains('A')].snr,bins=bins,histtype='step',linewidth=2,density=False,label='ALMA baselines')
    except AttributeError:
        plt.hist(foo[~foo.baseline.str.contains('A')].snr,bins=bins,histtype='step',linewidth=2,normed=False,label='non-ALMA baselines')
        plt.hist(foo[foo.baseline.str.contains('A')].snr,bins=bins,histtype='step',linewidth=2,normed=False,label='ALMA baselines')

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


    fig, ax = plt.subplots(1,1,sharey='all',sharex='all',figsize=(1*7,1*5))

    #couC=0
    foo1=foo[foo.expt_no==expt]
    for base in list(foo1.baseline.unique()):

        foo2=foo1[foo1.baseline==base]
        if ('ALMA' in base) or (('SMA' in base)&('JCMT' not in base)):
            ax.errorbar(foo2.uvdist,foo2.amp,bars_on*foo2.sigma,fmt='o',mfc='none',ms=8,color=palette_dict[base],label=SMT2Zb[base.split('-')[0]]+SMT2Zb[base.split('-')[1]])
        else:
            ax.errorbar(foo2.uvdist,foo2.amp,bars_on*foo2.sigma,fmt='x',ms=5,color=palette_dict[base],label=SMT2Zb[base.split('-')[0]]+SMT2Zb[base.split('-')[1]])
    ax.legend(bbox_to_anchor=(-0.15, 1.52))
    ax.grid()
    ax.set_title(sour+' | '+str((expt)))
    ax.set_xlabel('UV distance')
    ax.set_ylabel('amplitudes')
    if logscale==True:
        ax.set_yscale("log", nonposy='clip')
    ax.set_ylim(ymin=0.1*np.min(foo.amp))
    ax.set_ylim(ymax=1.1*np.max(foo.amp))
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


def compare_coherence_time(coh0,incoh0,dict_col_sour=dict_col_sour,snr_cut=0, polarizations=['LL','RR'],bands=['lo','hi'],debias_coh=True,debias_inc=True):

    coh0 = coh0[list(map(lambda x: x in polarizations, coh0.polarization))]
    coh00 = coh0[list(map(lambda x: x in bands, coh0.band))].copy()    
    incoh0 = incoh0[list(map(lambda x: x in polarizations, incoh0.polarization))]
    incoh00 = incoh0[list(map(lambda x: x in bands, incoh0.band))].copy()
    coh,incoh = ut.match_frames(coh00[coh00.snr>snr_cut].copy(),incoh00[incoh00.snr>snr_cut].copy(),['scan_id','baseline','polarization','band'])
    #print(np.shape(coh),np.shape(incoh))
    if debias_coh==True:
        coh['amp_coh'] = np.sqrt(np.maximum(0,coh['amp']**2 - coh['sigma']**2))
    else: coh['amp_coh'] = coh['amp']
    if debias_inc==True:
        coh['amp_incoh'] = np.sqrt(np.maximum(0,incoh['amp']**2 - incoh['sigma']**2))
    else: coh['amp_incoh'] = incoh['amp']
    
    coh['sigma_coh'] = coh['sigma']
    coh['sigma_incoh'] = incoh['sigma']
    coh['coh2incoh'] = coh['amp_coh']/coh['amp_incoh']
    coh['snr_coh'] = coh['snr']
    coh['snr_incoh'] = incoh['snr']
    data=coh.copy()
    sns.set_style('whitegrid')
    baseL = sorted(list(data.baseline.unique()))
    ncol = int(np.ceil(len(baseL)/2.))
    num_base = (np.ceil((np.asarray(range(2*ncol))+0.1)/2.))
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
        try:
            ax2 = sg.fig.axes[1]
            hm2x=ax2.get_xlim()
            hm2y=ax2.get_ylim()
            ax2.plot(hm2x,[1,1],'k--')
            #ax2.set_ylim(hm2y)
            #ax2.set_xlim(hm2x)
        except IndexError: continue
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
    plt.text(rangex[1], 0., "MED: {} \nMAD: {}\nREL MAD: {} ".format(format(med,'.4g'),format(mad_abs,'.4g'),format(mad_rel,'.4g')), size=12,
         va="center", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))
    #plt.text(0.5*rangex[1], 0.9*rangey[1], "MED: %4.3f" % med , bbox=dict(facecolor='white', alpha=1.))
    #plt.text(0.5*rangex[1], 0.8*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
    #plt.text(0.5*rangex[1], 0.7*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))
    plt.show()
    if by_what!='':
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
            ax[nrowL,ncolL].text(rangex[1], 0., "MED: {} \nMAD: {}\nREL MAD: {} ".format(format(med,'.4g'),format(mad_abs,'.4g'),format(mad_rel,'.4g')), size=12,
            va="center", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))
        plt.show()
    return data

def bandpass_cphase_consistency_prep(data0):

    data_lo, data_hi = ut.match_frames(data0[data0.band=='lo'].copy(),data0[data0.band=='hi'].copy(),['scan_id','triangle','polarization'])
    data = data_lo.copy()
    data['cphase_lo'] = data_lo['cphase']
    data['cphase_hi'] = data_hi['cphase']
    data['sigma_lo'] = data_lo['sigmaCP']
    data['sigma_hi'] = data_hi['sigmaCP']
    data['sigma'] = np.sqrt(data['sigma_lo']**2 + data['sigma_hi']**2)
    data['cphase_diff'] = np.angle(np.exp(1j*(data['cphase_lo'] - data['cphase_hi'])*np.pi/180))*180./np.pi
    data['rel_diff'] = np.asarray(data['cphase_diff'])/np.asarray(data['sigma'])

    return data


def bandpass_cphase_consistency(data0,xmax=10,by_what='source'):

    data = bandpass_cphase_consistency_prep(data0)

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
    plt.text(rangex[1], 0., "MAD: {} deg \nREL MAD: {} ".format(format(mad_abs,'.4g'),format(mad_rel,'.4g')), size=12,
         va="center", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))
    #plt.text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f deg" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
    #plt.text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))
    plt.show()
    if by_what!='':
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
            ax[nrowL,ncolL].text(rangex[1], 0., "MAD: {} deg \nREL MAD: {} ".format(format(mad_abs,'.4g'),format(mad_rel,'.4g')), size=12,
            va="center", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))
            #ax[nrowL,ncolL].text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f deg" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
            #ax[nrowL,ncolL].text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))
        plt.show()
    return data

def polar_cphase_consistency_prep(data0):

    data_rr, data_ll = ut.match_frames(data0[data0.polarization=='LL'].copy(),data0[data0.polarization=='RR'].copy(),['scan_id','triangle','band'])
    data = data_rr.copy()
    data['cphase_rr'] = data_rr['cphase']
    data['cphase_ll'] = data_ll['cphase']
    data['sigma_rr'] = data_rr['sigmaCP']
    data['sigma_ll'] = data_ll['sigmaCP']
    data['sigma'] = np.sqrt(data['sigma_ll']**2 + data['sigma_rr']**2)
    data['cphase_diff'] = np.angle(np.exp(1j*(data['cphase_rr'] - data['cphase_ll'])*np.pi/180))*180./np.pi
    data['rel_diff'] = np.asarray(data['cphase_diff'])/np.asarray(data['sigma'])

    return data

def polar_cphase_consistency(data0,xmax=10,by_what='source'):

    data = polar_cphase_consistency_prep(data0)

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
    plt.text(rangex[1], 0., "MAD: {} deg \nREL MAD: {} ".format(format(mad_abs,'.4g'),format(mad_rel,'.4g')), size=12,
         va="center", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))

    #plt.text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f deg" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
    #plt.text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))

    plt.show()
    if by_what!='':
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
            ax[nrowL,ncolL].text(rangex[1], 0., "MAD: {} deg \nREL MAD: {} ".format(format(mad_abs,'.4g'),format(mad_rel,'.4g')), size=12,
            va="center", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))

            #ax[nrowL,ncolL].text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f deg" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
            #ax[nrowL,ncolL].text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))
        plt.show()
    return data


def bandpass_amplitude_rel_consistency_prep(data0):

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

    return data


def bandpass_amplitude_rel_consistency(data0,xmax=2.,by_what='source'):

    data = bandpass_amplitude_rel_consistency_prep(data0)

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
    plt.text(rangex[1], 0., "MED: {} \nMAD: {}\nREL MAD: {} ".format(format(med,'.4g'),format(mad_abs,'.4g'),format(mad_rel,'.4g')), size=12,
         va="center", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))
    #plt.text(0.5*rangex[1], 0.9*rangey[1], "MED: %4.3f" % med , bbox=dict(facecolor='white', alpha=1.))
    #plt.text(0.5*rangex[1], 0.8*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
    #plt.text(0.5*rangex[1], 0.7*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))

    plt.show()
    if by_what!='':
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
            ax[nrowL,ncolL].text(rangex[1], 0., "MED: {} \nMAD: {}\nREL MAD: {} ".format(format(med,'.4g'),format(mad_abs,'.4g'),format(mad_rel,'.4g')), size=12,
            va="center", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))
            #ax[nrowL,ncolL].text(0.5*rangex[1], 0.9*rangey[1], "MED: %4.3f" % med , bbox=dict(facecolor='white', alpha=1.))
            #ax[nrowL,ncolL].text(0.5*rangex[1], 0.8*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
            #ax[nrowL,ncolL].text(0.5*rangex[1], 0.7*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))

        plt.show()
    return data


def polar_amplitude_consistency_prep(data0):

    data_rr, data_ll = ut.match_frames(data0[data0.polarization=='LL'].copy(),data0[data0.polarization=='RR'].copy(),['scan_id','baseline','band'])
    data = data_rr.copy()
    data['amp_rr'] = data_rr['amp']
    data['amp_ll'] = data_ll['amp']
    data['sigma_rr'] = data_rr['sigma']
    data['sigma_ll'] = data_ll['sigma']
    data['sigma'] = np.sqrt(data['sigma_rr']**2 + data['sigma_ll']**2)
    data['amp_diff'] = data['amp_rr'] - data['amp_ll']
    data['rel_diff'] = data['amp_diff']/data['sigma']
    return data

def polar_amplitude_consistency(data0,xmax=10,by_what='source'):

    data = polar_amplitude_consistency_prep(data0)

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
    plt.text(rangex[1], 0., "MED: {} \nMAD: {}\nREL MAD: {} ".format(format(med,'.4g'),format(mad_abs,'.4g'),format(mad_rel,'.4g')), size=12,
         va="center", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))
    #plt.text(0.5*rangex[1], 0.9*rangey[1], "MED: %4.3f" % med , bbox=dict(facecolor='white', alpha=1.))
    #plt.text(0.5*rangex[1], 0.8*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
    #plt.text(0.5*rangex[1], 0.7*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))

    plt.show()
    if by_what!='':
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
            ax[nrowL,ncolL].text(rangex[1], 0., "MED: {} \nMAD: {}\nREL MAD: {} ".format(format(med,'.4g'),format(mad_abs,'.4g'),format(mad_rel,'.4g')), size=12,
            va="center", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))
            #ax[nrowL,ncolL].text(0.5*rangex[1], 0.9*rangey[1], "MED: %4.3f" % med , bbox=dict(facecolor='white', alpha=1.))
            #ax[nrowL,ncolL].text(0.5*rangex[1], 0.8*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
            #ax[nrowL,ncolL].text(0.5*rangex[1], 0.7*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))
        plt.show()
    return data


def polar_amplitude_rel_consistency_prep(data0):

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
    return data

def polar_amplitude_rel_consistency(data0,xmax=2.,by_what='source'):

    data = polar_amplitude_rel_consistency_prep(data0)

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
    plt.text(rangex[1], 0., "MED: {} \nMAD: {}\nREL MAD: {} ".format(format(med,'.4g'),format(mad_abs,'.4g'),format(mad_rel,'.4g')), size=12,
         va="center", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))
    #plt.text(0.5*rangex[1], 0.9*rangey[1], "MED: %4.3f" % med , bbox=dict(facecolor='white', alpha=1.))
    #plt.text(0.5*rangex[1], 0.8*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
    #plt.text(0.5*rangex[1], 0.7*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))

    plt.show()
    if by_what!='':
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
            ax[nrowL,ncolL].text(rangex[1], 0., "MED: {} \nMAD: {}\nREL MAD: {} ".format(format(med,'.4g'),format(mad_abs,'.4g'),format(mad_rel,'.4g')), size=12,
            va="center", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))
            #ax[nrowL,ncolL].text(0.5*rangex[1], 0.9*rangey[1], "MED: %4.3f" % med , bbox=dict(facecolor='white', alpha=1.))
            #ax[nrowL,ncolL].text(0.5*rangex[1], 0.8*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
            #ax[nrowL,ncolL].text(0.5*rangex[1], 0.7*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))
        plt.show()
    return data

def bandpass_lcamp_consistency_prep(data0):

    data_lo, data_hi = ut.match_frames(data0[data0.band=='lo'].copy(),data0[data0.band=='hi'].copy(),['scan_id','quadrangle','polarization'])
    data = data_lo.copy()
    data['lcamp_lo'] = data_lo['camp']
    data['lcamp_hi'] = data_hi['camp']
    data['sigma_lo'] = data_lo['sigmaCA']
    data['sigma_hi'] = data_hi['sigmaCA']
    data['sigma'] = np.sqrt(data['sigma_lo']**2 + data['sigma_hi']**2)
    data['lcamp_diff'] = data['lcamp_lo'] - data['lcamp_hi']
    data['rel_diff'] = np.asarray(data['lcamp_diff'])/np.asarray(data['sigma'])
    data.dropna(subset=['rel_diff'],inplace=True)

    return data

def bandpass_lcamp_consistency(data0,xmax=10,by_what='source'):

    data= bandpass_lcamp_consistency_prep(data0)

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
    plt.text(rangex[1], 0., "MAD: {}\nREL MAD: {} ".format(format(mad_abs,'.4g'),format(mad_rel,'.4g')), size=12,
         va="center", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))

    #plt.text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
    #plt.text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))

    plt.show()
    if by_what!='':
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
            ax[nrowL,ncolL].text(rangex[1], 0., "MAD: {}\nREL MAD: {} ".format(format(mad_abs,'.4g'),format(mad_rel,'.4g')), size=12,
            va="center", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))
            #ax[nrowL,ncolL].text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
            #ax[nrowL,ncolL].text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))
        plt.show()
    return data

def polar_lcamp_consistency_prep(data0):

    data_rr, data_ll = ut.match_frames(data0[data0.polarization=='LL'].copy(),data0[data0.polarization=='RR'].copy(),['scan_id','quadrangle','band'])
    data = data_rr.copy()
    data['lcamp_rr'] = data_rr['camp']
    data['lcamp_ll'] = data_ll['camp']
    data['sigma_rr'] = data_rr['sigmaCA']
    data['sigma_ll'] = data_ll['sigmaCA']
    data['sigma'] = np.sqrt(data['sigma_rr']**2 + data['sigma_ll']**2)
    data['lcamp_diff'] = data['lcamp_rr'] - data['lcamp_ll']
    data['rel_diff'] = np.asarray(data['lcamp_diff'])/np.asarray(data['sigma'])
    data.dropna(subset=['rel_diff'],inplace=True)
    nbins = int(np.sqrt(np.shape(data)[0]))
    bins = np.linspace(-xmax,xmax,nbins)
    x=np.linspace(-xmax,xmax,128)
    return data

def polar_lcamp_consistency(data0,xmax=10,by_what='source'):

    data = polar_lcamp_consistency_prep(data0)

    data_rr, data_ll = ut.match_frames(data0[data0.polarization=='LL'].copy(),data0[data0.polarization=='RR'].copy(),['scan_id','quadrangle','band'])
    data = data_rr.copy()
    data['lcamp_rr'] = data_rr['camp']
    data['lcamp_ll'] = data_ll['camp']
    data['sigma_rr'] = data_rr['sigmaCA']
    data['sigma_ll'] = data_ll['sigmaCA']
    data['sigma'] = np.sqrt(data['sigma_rr']**2 + data['sigma_ll']**2)
    data['lcamp_diff'] = data['lcamp_rr'] - data['lcamp_ll']
    data['rel_diff'] = np.asarray(data['lcamp_diff'])/np.asarray(data['sigma'])
    data.dropna(subset=['rel_diff'],inplace=True)
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
    plt.text(rangex[1], 0., "MAD: {}\nREL MAD: {} ".format(format(mad_abs,'.4g'),format(mad_rel,'.4g')), size=12,
         va="center", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))
    #plt.text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
    #plt.text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))
    plt.show()
    if by_what!='':
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
            ax[nrowL,ncolL].text(rangex[1], 0., "MAD: {}\nREL MAD: {} ".format(format(mad_abs,'.4g'),format(mad_rel,'.4g')), size=12,
            va="center", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))
            #ax[nrowL,ncolL].text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
            #ax[nrowL,ncolL].text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))
        plt.show()
    return data


def lcamp_prep(data,add_sys=None):

    data['rel_diff'] = np.asarray(data['camp'])/np.asarray(data['sigmaCA'])
    if add_sys is not None:
        data['rel_diff_sys'] = np.asarray(data['camp'])/np.sqrt(np.asarray(data['sigmaCA'])**2 +add_sys**2)
    return data

def trivial_lcamp(data0,xmax=10,whichB='all',by_what='source',est_sys=False):

    data = lcamp_prep(data0)

    data = cl.only_trivial_quadrangles_str(data0, whichB=whichB)
    data=data.copy()
    data['rel_diff'] = np.asarray(data['camp'])/np.asarray(data['sigmaCA'])

    nbins = int(np.sqrt(np.shape(data)[0]))
    bins = np.linspace(-xmax,xmax,nbins)
    x=np.linspace(-xmax,xmax,128)
    data.dropna(subset=['rel_diff'],inplace=True)
    plt.hist(data['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
    if est_sys:
        s0 = get_systematic(data,'camp','sigmaCA')
        #print('S0: ',s0)
        data['corrected'] = np.asarray(data['camp'])/np.sqrt(np.asarray(data['sigmaCA'])**2 + s0**2 )
        plt.hist(data['corrected'],bins=bins,histtype='step',linewidth=2,density=True)

    plt.grid()
    plt.plot(x,np.exp(-(x)**2/2)/np.sqrt(2.*np.pi),'k')
    plt.axvline(0,color='k')
    plt.xlabel('(TCA)/(thermal error)')
    plt.title('All data')
    mad_abs=np.median(np.abs(data['camp']))
    mad_rel=np.median(np.abs(data['rel_diff']))/0.67449
    rangey = plt.ylim()
    rangex = plt.xlim()
    #plt.text(rangex[1], 0., "MAD: {}\nREL MAD: {} ".format(format(mad_abs,'.4g'),format(mad_rel,'.4g')), size=12,
    #     va="center", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))
    if est_sys:
        plt.text(rangex[1], 0., "MAD: {} deg \nREL MAD: {} \nSYS ERR: {}".format(format(mad_abs,'.4g'),format(mad_rel,'.4g'),format(s0,'.4g')), size=12,
        va="center", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))
    else:
        plt.text(rangex[1], 0., "MAD: {} deg \nREL MAD: {} ".format(format(mad_abs,'.4g'),format(mad_rel,'.4g')), size=12,
        va="center", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))

    #plt.text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
    #plt.text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))

    plt.show()
    if by_what!='':
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
            if est_sys:
                s0 = get_systematic(data[data[by_what]==what],'camp','sigmaCA')
                data[data[by_what]==what]['corrected'] = np.asarray(data[data[by_what]==what]['camp'])/np.sqrt(np.asarray(data[data[by_what]==what]['sigmaCA'])**2 + s0**2 )
                ax[nrowL,ncolL].hist(data[data[by_what]==what]['corrected'],bins=bins,histtype='step',linewidth=2,density=True)
            ax[nrowL,ncolL].plot(x,np.exp(-(x)**2/2)/np.sqrt(2.*np.pi),'k')
            ax[nrowL,ncolL].grid()
            ax[nrowL,ncolL].axvline(0,color='k')
            ax[nrowL,ncolL].set_xlabel('(TCA)/(thermal error)')
            ax[nrowL,ncolL].set_title(what)
            mad_abs=np.median(np.abs(data[data[by_what]==what]['camp']))
            mad_rel=np.median(np.abs(data[data[by_what]==what]['rel_diff']))/0.67449
            rangey = ax[nrowL,ncolL].get_ylim()
            rangex = ax[nrowL,ncolL].get_xlim()
            #ax[nrowL,ncolL].text(rangex[1], 0., "MAD: {}\nREL MAD: {} ".format(format(mad_abs,'.4g'),format(mad_rel,'.4g')), size=12,
            #va="center", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))
            if est_sys:
                ax[nrowL,ncolL].text(rangex[1], 0., "MAD: {} deg\nREL MAD: {} \nSYS ERR: {}".format(format(mad_abs,'.4g'),format(mad_rel,'.4g'),format(s0,'.4g')), size=12,
                va="center", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))
            else:
                ax[nrowL,ncolL].text(rangex[1], 0., "MAD: {} deg\nREL MAD: {} ".format(format(mad_abs,'.4g'),format(mad_rel,'.4g')), size=12,
                va="center", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))

            #ax[nrowL,ncolL].text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
            #ax[nrowL,ncolL].text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))
        plt.show()
    return data

def get_systematic(data,absolute,error):
    import scipy.optimize as so
    absolut = np.asarray(data[absolute])
    err = np.asarray(data[error])
    m0 = np.median(np.abs(absolut/np.sqrt(err**2))/0.67449)
    if m0>1.:
        fun0 = lambda x: np.median( np.abs(absolut/np.sqrt(err**2 + x**2)) )/0.67449 -1.
        #print(fun0(0),fun0(10))
        s0 = so.brentq(fun0, 0, 100)
    else: s0=0.
    return s0


def cphase_prep(data0,add_sys=None):

    data['rel_diff'] = np.asarray(data['cphase'])/np.asarray(data['sigmaCP'])
    if add_sys is not None:
        data['rel_diff_sys'] = np.asarray(data['cphase'])/np.sqrt(np.asarray(data['sigmaCP'])**2 +add_sys**2)
    return data

def trivial_cphase(data0,xmax=10,whichB='all',by_what='source',est_sys=False,add_sys=None):

    data = cl.only_trivial_triangles(data0, whichB=whichB)
    data=data.copy()
    #data['rel_diff'] = np.asarray(data['cphase'])/np.asarray(data['sigmaCP'])
    if add_sys is not None:
        data['rel_diff_sys'] = np.asarray(data['cphase'])/np.asarray(data['sigmaCP']**2 +add_sys**2)

    nbins = int(np.sqrt(np.shape(data)[0]))
    bins = np.linspace(-xmax,xmax,nbins)
    x=np.linspace(-xmax,xmax,128)
    plt.hist(data['rel_diff'],bins=bins,histtype='step',linewidth=2,density=True)
    if est_sys:
        s0 = get_systematic(data,'cphase','sigmaCP')
        #print('S0: ',s0)
        data['corrected'] = np.asarray(data['cphase'])/np.sqrt(np.asarray(data['sigmaCP'])**2 + s0**2 )
        plt.hist(data['corrected'],bins=bins,histtype='step',linewidth=2,density=True)
    plt.grid()
    plt.plot(x,np.exp(-(x)**2/2)/np.sqrt(2.*np.pi),'k')
    plt.axvline(0,color='k')
    plt.xlabel('(TCP)/(thermal error)')
    plt.title('All data')
    mad_abs=np.median(np.abs(data['cphase']))
    mad_rel=np.median(np.abs(data['rel_diff']))/0.67449
    rangey = plt.ylim()
    rangex = plt.xlim()
    if est_sys:
        plt.text(rangex[1], 0., "MAD: {} deg \nREL MAD: {} \nSYS ERR: {}".format(format(mad_abs,'.4g'),format(mad_rel,'.4g'),format(s0,'.4g')), size=12,
        va="center", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))
    else:
        plt.text(rangex[1], 0., "MAD: {} deg \nREL MAD: {} ".format(format(mad_abs,'.4g'),format(mad_rel,'.4g')), size=12,
        va="center", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))
    #plt.text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f deg" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
    #plt.text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))

    plt.show()
    if by_what!='':
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
            if est_sys:
                #data= data[data[by_what]==what].copy()
                s0 = get_systematic(data[data[by_what]==what],'cphase','sigmaCP')
                data[data[by_what]==what]['corrected'] = np.asarray(data[data[by_what]==what]['cphase'])/np.sqrt(np.asarray(data[data[by_what]==what]['sigmaCP'])**2 + s0**2 )
                ax[nrowL,ncolL].hist(data[data[by_what]==what]['corrected'],bins=bins,histtype='step',linewidth=2,density=True)
            ax[nrowL,ncolL].plot(x,np.exp(-(x)**2/2)/np.sqrt(2.*np.pi),'k')
            ax[nrowL,ncolL].grid()
            ax[nrowL,ncolL].axvline(0,color='k')
            ax[nrowL,ncolL].set_xlabel('(TCP)/(thermal error)')
            ax[nrowL,ncolL].set_title(what)
            mad_abs=np.median(np.abs(data[data[by_what]==what]['cphase']))
            mad_rel=np.median(np.abs(data[data[by_what]==what]['rel_diff']))/0.67449
            rangey = ax[nrowL,ncolL].get_ylim()
            rangex = ax[nrowL,ncolL].get_xlim()
            if est_sys:
                ax[nrowL,ncolL].text(rangex[1], 0., "MAD: {} deg\nREL MAD: {} \nSYS ERR: {}".format(format(mad_abs,'.4g'),format(mad_rel,'.4g'),format(s0,'.4g')), size=12,
                va="center", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))
            else:
                ax[nrowL,ncolL].text(rangex[1], 0., "MAD: {} deg\nREL MAD: {} ".format(format(mad_abs,'.4g'),format(mad_rel,'.4g')), size=12,
                va="center", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))

            #ax[nrowL,ncolL].text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f deg" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
            #ax[nrowL,ncolL].text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))
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
    if norm_thermal==True:
        plt.plot(x,np.exp(-(x)**2/2)/np.sqrt(2.*np.pi),'k')
    plt.axvline(0,color='k')
    plt.xlabel(lab)
    plt.title('All data')

    med=np.median((data['amp_diff']))
    mad_abs=np.median(np.abs(data['amp_diff']))
    mad_rel=np.median(np.abs(data['rel_diff']))/0.67449
    rangey = plt.ylim()
    rangex = plt.xlim()
    plt.text(xmax, 0.98*rangey[1], "MED: {} \nMAD: {}\nREL MAD: {} ".format(format(med,'.4g'),format(mad_abs,'.4g'),format(mad_rel,'.4g')), size=12,
         va="top", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))    #plt.text(0.5*rangex[1], 0.9*rangey[1], "MED: %4.3f" % med , bbox=dict(facecolor='white', alpha=1.))
    #plt.text(0.5*rangex[1], 0.8*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
    #plt.text(0.5*rangex[1], 0.7*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))

    plt.show()
    print('Comparing {} matched data points'.format(np.shape(p1)[0]))
    print("Median absolute error in % of amplitude: {}".format(100.*np.median(np.abs(data['amp_diff']/data['amp_mean']))))
    print("90th percentile of absolute error in % of amplitude: {}".format(100.*np.percentile(np.abs(data['amp_diff']/data['amp_mean']),90)))
    if by_what!='':
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
            if norm_thermal==True:
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
            ax[nrowL,ncolL].text(rangex[1], 0., "MED: {} \nMAD: {} \nREL MAD: {} ".format(format(med,'.4g'),format(mad_abs,'.4g'),format(mad_rel,'.4g')), size=12,
            va="center", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))
            #ax[nrowL,ncolL].text(0.5*rangex[1], 0.9*rangey[1], "MED: %4.3f" % med , bbox=dict(facecolor='white', alpha=1.))
            #ax[nrowL,ncolL].text(0.5*rangex[1], 0.8*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
            #ax[nrowL,ncolL].text(0.5*rangex[1], 0.7*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))

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
    med=np.median((data['cphase_diff']))
    mad_abs=np.median(np.abs(data['cphase_diff']))
    mad_rel=np.median(np.abs(data['rel_diff']))/0.67449
    rangey = plt.ylim()
    rangex = plt.xlim()
    plt.text(rangex[1], 0., "MED: {} \nMAD: {} deg\nREL MAD: {} ".format(format(med,'.4g'),format(mad_abs,'.4g'),format(mad_rel,'.4g')), size=12,
         va="center", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))
    #plt.text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f deg" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
    #plt.text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))

    plt.show()
    print('Comparing {} scans'.format(np.shape(data_p1)[0]))
    print("Median absolute error in deg: {}".format(np.median(np.abs(data['cphase_diff']))))
    print("90th percentile of absolute error in deg: {}".format(np.percentile(np.abs(data['cphase_diff']),90)))
    if by_what!='':
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
            med=np.median((data[data[by_what]==what]['cphase_diff']))
            mad_abs=np.median(np.abs(data[data[by_what]==what]['cphase_diff']))
            mad_rel=np.median(np.abs(data[data[by_what]==what]['rel_diff']))/0.67449
            rangey = ax[nrowL,ncolL].get_ylim()
            rangex = ax[nrowL,ncolL].get_xlim()
            ax[nrowL,ncolL].text(rangex[1], 0., "MED: {} \nMAD: {}\nREL MAD: {} ".format(format(med,'.4g'),format(mad_abs,'.4g'),format(mad_rel,'.4g')), size=12,
            va="bottom", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))
        
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
    med=np.median((data['lcamp_diff']))
    mad_abs=np.median(np.abs(data['lcamp_diff']))
    mad_rel=np.median(np.abs(data['rel_diff']))/0.67449
    rangey = plt.ylim()
    rangex = plt.xlim()
    plt.text(0.98*xmax, 0.98*rangey[1], "MED: {} \nMAD: {}\nREL MAD: {} ".format(format(med,'.4g'),format(mad_abs,'.4g'),format(mad_rel,'.4g')), size=12,
         va="top", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))

    #plt.text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
    #plt.text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))
    plt.show()
    print('Comparing {} scans'.format(np.shape(data_p1)[0]))
    print("Median absolute error: {}".format(np.median(np.abs(data['lcamp_diff']))))
    print("90th percentile of absolute error: {}".format(np.percentile(np.abs(data['lcamp_diff']),90)))

    if by_what!='':
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
            med=np.median((data[data[by_what]==what]['lcamp_diff']))
            mad_abs=np.median(np.abs(data[data[by_what]==what]['lcamp_diff']))
            mad_rel=np.median(np.abs(data[data[by_what]==what]['rel_diff']))/0.67449
            rangey = ax[nrowL,ncolL].get_ylim()
            rangex = ax[nrowL,ncolL].get_xlim()
            ax[nrowL,ncolL].text(rangex[1], 0., "MED: {} \nMAD: {}\nREL MAD: {} ".format(format(med,'.4g'),format(mad_abs,'.4g'),format(mad_rel,'.4g')), size=12,
            va="center", ha="right", multialignment="left",bbox=dict(facecolor='white',alpha=0.8))
            #ax[nrowL,ncolL].text(0.5*rangex[1], 0.9*rangey[1], "MAD: %4.3f" % mad_abs , bbox=dict(facecolor='white', alpha=1.))
            #ax[nrowL,ncolL].text(0.5*rangex[1], 0.8*rangey[1], "REL MAD: %4.3f" % mad_rel , bbox=dict(facecolor='white', alpha=1.))
        plt.show()
    return data


def plot_polgains(data,base):
    
    coldic=dict_col_sour
    foo = data[data['baseline']==base]
    
    ####PLOT AMP RATIOS
    plt.figure(figsize=(14,8))
    plt.grid()
    plt.axhline(1,linestyle='--', color='gray')
    for source in list(foo.source.unique()):
        foo2 = foo[(foo.source==source)&(foo.AmpRatioErr<2.)]
        plt.errorbar(foo2.scan_id, foo2.AmpRatio,yerr=foo2.AmpRatioErr,fmt='o',capsize=5,label=source,c=coldic[source])
    plt.grid()   
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.))
    plt.xlim([-10,510])
    plt.ylabel('R/L amplitude',fontsize=15)
    plt.xlabel('scan id',fontsize=15)
    plt.title(base+ ' amp ratio',fontsize=16)
    plt.grid()
    plt.show()
    print('Median amp ratio: ', np.median(foo.AmpRatio))
    print('Median abs deviation from 1: ', np.median(np.abs(foo.AmpRatio-1.)))

    ####PLOT PHASE OFFSETS
    plt.figure(figsize=(14,8))
    plt.grid()
    plt.axhline(0,linestyle='--', color='gray')
    for source in list(foo.source.unique()):
        foo2 = foo[(foo.source==source)&(foo.RLphaseErr<360.)]
        plt.errorbar(foo2.scan_id, foo2.RLphase,yerr=foo2.RLphaseErr,fmt='o',capsize=5,label=source,c=coldic[source])
    plt.grid()
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.))
    #plt.axis([-10,510,-100,100])
    plt.xlim([-10,510])
    plt.ylabel('R-L phase [deg]',fontsize=15)
    plt.xlabel('scan id',fontsize=15)
    plt.title(base+ ' phase diff',fontsize=16)
    plt.grid()
    plt.show()
    print('Median phase diff [deg]: ', np.median(foo.RLphase))
    print('Median abs deviation from 0 [deg]: ', np.median(np.abs(foo.RLphase-0.)))
    
    
def plot_all_polgains(data):
    
    sources=list(data.source.unique())
    baseL=sorted(list(data.baseline.unique()))
    baseL = [x for x in baseL if 'J' not in x] #JCMT is singlepol
    #data=data[data.source!='1055+018'] #this one has really crappy gains
    
    for base in baseL:
        print('\n')
        print('\n')
        print('=====================================')
        print('=====================================')
        print('=================='+base+'=================')
        print('=====================================')
        print('=====================================')
        plot_polgains(data,base)


dict_scan_id={'094-2231': 0,
 '094-2242': 1,
 '094-2253': 2,
 '094-2304': 3,
 '094-2319': 4,
 '094-2330': 5,
 '094-2344': 6,
 '094-2355': 7,
 '095-0009': 8,
 '095-0020': 9,
 '095-0031': 10,
 '095-0046': 11,
 '095-0100': 12,
 '095-0110': 13,
 '095-0125': 14,
 '095-0138': 15,
 '095-0149': 16,
 '095-0159': 17,
 '095-0214': 18,
 '095-0225': 19,
 '095-0236': 20,
 '095-0251': 21,
 '095-0258': 22,
 '095-0310': 23,
 '095-0325': 24,
 '095-0332': 25,
 '095-0345': 26,
 '095-0359': 27,
 '095-0408': 28,
 '095-0415': 29,
 '095-0428': 30,
 '095-0435': 31,
 '095-0450': 32,
 '095-0500': 33,
 '095-0511': 34,
 '095-0519': 35,
 '095-0530': 36,
 '095-0538': 37,
 '095-0549': 38,
 '095-0557': 39,
 '095-0612': 40,
 '095-0620': 41,
 '095-0631': 42,
 '095-0639': 43,
 '095-0650': 44,
 '095-0658': 45,
 '095-0709': 46,
 '095-0728': 47,
 '095-0736': 48,
 '095-0747': 49,
 '095-0755': 50,
 '095-0806': 51,
 '095-0814': 52,
 '095-0829': 53,
 '095-0839': 54,
 '095-0849': 55,
 '095-0859': 56,
 '095-0908': 57,
 '095-0923': 58,
 '095-0935': 59,
 '095-0942': 60,
 '095-0954': 61,
 '095-1005': 62,
 '095-1012': 63,
 '095-1025': 64,
 '095-1036': 65,
 '095-1045': 66,
 '095-1052': 67,
 '095-1105': 68,
 '095-1118': 69,
 '095-1129': 70,
 '095-1136': 71,
 '095-1148': 72,
 '095-1156': 73,
 '095-1209': 74,
 '095-1223': 75,
 '095-1230': 76,
 '095-1243': 77,
 '095-1257': 78,
 '095-1310': 79,
 '095-1324': 80,
 '095-1331': 81,
 '095-1344': 82,
 '095-1355': 83,
 '095-1402': 84,
 '095-1419': 85,
 '095-1431': 86,
 '095-1438': 87,
 '095-1449': 88,
 '095-1458': 89,
 '095-1505': 90,
 '095-1513': 91,
 '095-1526': 92,
 '095-1533': 93,
 '095-1547': 94,
 '095-1558': 95,
 '095-1605': 96,
 '095-1612': 97,
 '095-1619': 98,
 '095-1628': 99,
 '095-1635': 100,
 '095-1642': 101,
 '095-1649': 102,
 '095-1656': 103,
 '095-1703': 104,
 '096-0046': 105,
 '096-0052': 106,
 '096-0104': 107,
 '096-0110': 108,
 '096-0122': 109,
 '096-0128': 110,
 '096-0140': 111,
 '096-0146': 112,
 '096-0158': 113,
 '096-0204': 114,
 '096-0218': 115,
 '096-0224': 116,
 '096-0236': 117,
 '096-0242': 118,
 '096-0254': 119,
 '096-0300': 120,
 '096-0312': 121,
 '096-0318': 122,
 '096-0332': 123,
 '096-0338': 124,
 '096-0350': 125,
 '096-0356': 126,
 '096-0408': 127,
 '096-0414': 128,
 '096-0426': 129,
 '096-0432': 130,
 '096-0446': 131,
 '096-0452': 132,
 '096-0502': 133,
 '096-0508': 134,
 '096-0518': 135,
 '096-0524': 136,
 '096-0534': 137,
 '096-0540': 138,
 '096-0554': 139,
 '096-0600': 140,
 '096-0610': 141,
 '096-0616': 142,
 '096-0626': 143,
 '096-0632': 144,
 '096-0642': 145,
 '096-0648': 146,
 '096-0702': 147,
 '096-0708': 148,
 '096-0718': 149,
 '096-0724': 150,
 '096-0734': 151,
 '096-0740': 152,
 '096-0750': 153,
 '096-0756': 154,
 '096-0817': 155,
 '096-0824': 156,
 '096-0835': 157,
 '096-0847': 158,
 '096-0854': 159,
 '096-0905': 160,
 '096-0918': 161,
 '096-0926': 162,
 '096-0938': 163,
 '096-0949': 164,
 '096-1000': 165,
 '096-1012': 166,
 '096-1019': 167,
 '096-1030': 168,
 '096-1041': 169,
 '096-1048': 170,
 '096-1059': 171,
 '096-1110': 172,
 '096-1121': 173,
 '096-1132': 174,
 '096-1144': 175,
 '096-1151': 176,
 '096-1202': 177,
 '096-1214': 178,
 '096-1221': 179,
 '096-1237': 180,
 '096-1248': 181,
 '096-1300': 182,
 '096-1307': 183,
 '096-1318': 184,
 '096-1330': 185,
 '096-1337': 186,
 '096-1353': 187,
 '096-1404': 188,
 '096-1415': 189,
 '096-1425': 190,
 '096-1437': 191,
 '096-1444': 192,
 '096-1453': 193,
 '096-1505': 194,
 '096-1512': 195,
 '096-1522': 196,
 '096-1533': 197,
 '096-1541': 198,
 '096-1551': 199,
 '096-1601': 200,
 '096-1611': 201,
 '097-0401': 202,
 '097-0414': 203,
 '097-0423': 204,
 '097-0433': 205,
 '097-0446': 206,
 '097-0455': 207,
 '097-0506': 208,
 '097-0518': 209,
 '097-0529': 210,
 '097-0541': 211,
 '097-0553': 212,
 '097-0602': 213,
 '097-0612': 214,
 '097-0625': 215,
 '097-0634': 216,
 '097-0646': 217,
 '097-0700': 218,
 '097-0709': 219,
 '097-0720': 220,
 '097-0729': 221,
 '097-0747': 222,
 '097-0804': 223,
 '097-0812': 224,
 '097-0828': 225,
 '097-0836': 226,
 '097-0848': 227,
 '097-0858': 228,
 '097-0904': 229,
 '097-0913': 230,
 '097-0926': 231,
 '097-0937': 232,
 '097-0950': 233,
 '097-0958': 234,
 '097-1011': 235,
 '097-1019': 236,
 '097-1029': 237,
 '097-1035': 238,
 '097-1046': 239,
 '097-1057': 240,
 '097-1108': 241,
 '097-1117': 242,
 '097-1130': 243,
 '097-1136': 244,
 '097-1146': 245,
 '097-1158': 246,
 '097-1204': 247,
 '097-1220': 248,
 '097-1231': 249,
 '097-1237': 250,
 '097-1250': 251,
 '097-1303': 252,
 '097-1309': 253,
 '097-1326': 254,
 '097-1337': 255,
 '097-1343': 256,
 '097-1356': 257,
 '097-1407': 258,
 '097-1420': 259,
 '097-1428': 260,
 '097-1439': 261,
 '097-1450': 262,
 '097-1501': 263,
 '097-1507': 264,
 '097-1518': 265,
 '097-1531': 266,
 '097-1539': 267,
 '097-1547': 268,
 '097-1555': 269,
 '097-1608': 270,
 '097-1618': 271,
 '097-1626': 272,
 '097-1636': 273,
 '097-1646': 274,
 '097-1654': 275,
 '097-1704': 276,
 '097-1717': 277,
 '097-1725': 278,
 '097-1735': 279,
 '097-1745': 280,
 '097-1753': 281,
 '097-1803': 282,
 '097-1813': 283,
 '097-1821': 284,
 '097-1834': 285,
 '097-1844': 286,
 '097-1854': 287,
 '097-1904': 288,
 '097-1914': 289,
 '097-1924': 290,
 '097-1937': 291,
 '097-1946': 292,
 '097-1953': 293,
 '097-2000': 294,
 '097-2009': 295,
 '097-2018': 296,
 '097-2027': 297,
 '097-2039': 298,
 '099-2317': 299,
 '099-2328': 300,
 '099-2337': 301,
 '099-2348': 302,
 '099-2359': 303,
 '100-0012': 304,
 '100-0023': 305,
 '100-0035': 306,
 '100-0046': 307,
 '100-0057': 308,
 '100-0108': 309,
 '100-0123': 310,
 '100-0134': 311,
 '100-0145': 312,
 '100-0159': 313,
 '100-0209': 314,
 '100-0221': 315,
 '100-0232': 316,
 '100-0243': 317,
 '100-0252': 318,
 '100-0301': 319,
 '100-0309': 320,
 '100-0321': 321,
 '100-0330': 322,
 '100-0340': 323,
 '100-0359': 324,
 '100-0407': 325,
 '100-0416': 326,
 '100-0426': 327,
 '100-0437': 328,
 '100-0445': 329,
 '100-0453': 330,
 '100-0504': 331,
 '100-0517': 332,
 '100-0525': 333,
 '100-0533': 334,
 '100-0544': 335,
 '100-0555': 336,
 '100-0603': 337,
 '100-0611': 338,
 '100-0631': 339,
 '100-0639': 340,
 '100-0649': 341,
 '100-0657': 342,
 '100-0707': 343,
 '100-0715': 344,
 '100-0725': 345,
 '100-0738': 346,
 '100-0746': 347,
 '100-0756': 348,
 '100-0804': 349,
 '100-0814': 350,
 '100-0822': 351,
 '100-0830': 352,
 '100-0843': 353,
 '100-0853': 354,
 '100-0901': 355,
 '100-0909': 356,
 '100-0917': 357,
 '100-0925': 358,
 '100-0933': 359,
 '100-0941': 360,
 '100-0949': 361,
 '100-0957': 362,
 '100-1012': 363,
 '100-1021': 364,
 '100-1032': 365,
 '100-1043': 366,
 '100-1052': 367,
 '100-1101': 368,
 '100-1114': 369,
 '100-1123': 370,
 '100-1141': 371,
 '100-1150': 372,
 '100-1203': 373,
 '100-1216': 374,
 '100-1229': 375,
 '100-1241': 376,
 '100-1254': 377,
 '100-1307': 378,
 '100-1316': 379,
 '100-1329': 380,
 '100-1342': 381,
 '100-1354': 382,
 '100-1407': 383,
 '100-1420': 384,
 '100-1429': 385,
 '100-1442': 386,
 '100-1450': 387,
 '100-1459': 388,
 '100-1506': 389,
 '100-2216': 390,
 '100-2225': 391,
 '100-2234': 392,
 '100-2243': 393,
 '100-2252': 394,
 '100-2304': 395,
 '100-2316': 396,
 '100-2325': 397,
 '100-2334': 398,
 '100-2343': 399,
 '100-2352': 400,
 '101-0001': 401,
 '101-0013': 402,
 '101-0032': 403,
 '101-0041': 404,
 '101-0050': 405,
 '101-0102': 406,
 '101-0115': 407,
 '101-0124': 408,
 '101-0138': 409,
 '101-0150': 410,
 '101-0159': 411,
 '101-0212': 412,
 '101-0224': 413,
 '101-0233': 414,
 '101-0248': 415,
 '101-0300': 416,
 '101-0309': 417,
 '101-0322': 418,
 '101-0334': 419,
 '101-0343': 420,
 '101-0358': 421,
 '101-0410': 422,
 '101-0419': 423,
 '101-0432': 424,
 '101-0444': 425,
 '101-0453': 426,
 '101-0515': 427,
 '101-0524': 428,
 '101-0535': 429,
 '101-0546': 430,
 '101-0555': 431,
 '101-0606': 432,
 '101-0620': 433,
 '101-0629': 434,
 '101-0640': 435,
 '101-0651': 436,
 '101-0700': 437,
 '101-0711': 438,
 '101-0725': 439,
 '101-0736': 440,
 '101-0745': 441,
 '101-0754': 442,
 '101-0803': 443,
 '101-0812': 444,
 '101-0821': 445,
 '101-0830': 446,
 '101-0839': 447,
 '101-0900': 448,
 '101-0913': 449,
 '101-0920': 450,
 '101-0933': 451,
 '101-0947': 452,
 '101-0954': 453,
 '101-1007': 454,
 '101-1018': 455,
 '101-1036': 456,
 '101-1043': 457,
 '101-1053': 458,
 '101-1106': 459,
 '101-1120': 460,
 '101-1127': 461,
 '101-1139': 462,
 '101-1152': 463,
 '101-1208': 464,
 '101-1215': 465,
 '101-1227': 466,
 '101-1241': 467,
 '101-1248': 468,
 '101-1259': 469,
 '101-1313': 470,
 '101-1324': 471,
 '101-1335': 472,
 '101-1348': 473,
 '101-1355': 474,
 '101-1406': 475,
 '101-1419': 476,
 '101-1431': 477,
 '101-1442': 478,
 '101-1453': 479,
 '101-1505': 480,
 '101-1512': 481,
 '101-1518': 482}