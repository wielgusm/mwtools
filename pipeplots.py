import numpy as np
import pandas as pd
import seaborn as sns
from eat.io import hops, util
from eat.hops import util as hu
import matplotlib.pyplot as plt
from eat.inspect import utils as ut

def plot_amp_days(data,sour, bars_on=False, only_parallel=True,logscale=True):

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
    
    if only_parallel==True:
        foo=data[(data.source==sour)&(data.polarization.str[0]==data.polarization.str[1])].copy()
    else:
        foo=data[(data.source==sour)].copy()
    foo['baseline']=list(map(lambda x: Z2SMT[x[0]]+'-'+Z2SMT[x[1]],foo.baseline))
    exptL=list(foo.expt_no.unique())
    nplots=(len(exptL)+1)
    ncols=2
    nrows=np.maximum(2,int(np.ceil(nplots/ncols)))

    if 'uvdist' not in foo.columns:
        foo['uvdist'] = np.sqrt(foo['u']**2 + foo['v']**2)
    exptD = dict(zip(range(nplots),['all']+exptL))

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
                ax[couR,couC].set_title(sour+' | '+str(exptD[couP]))
                ax[couR,couC].set_xlabel('UT time')
                ax[couR,couC].set_ylabel('amplitudes')
                if logscale==True:
                    ax[couR,couC].set_yscale("log", nonposy='clip')
                ax[couR,couC].set_ylim(ymin=0.1*np.min(foo.amp))
                ax[couR,couC].set_ylim(ymax=2.*np.max(foo.amp))
                couP+=1
    plt.tight_layout()
    
    nscan=np.shape(foo)[0]
    nbase = len(foo.baseline.unique())
    print(sour)
    print("{} detections on {} baselines".format(nscan,nbase))
    print("median snr {}".format(np.median(foo.snr)))
    print("=========================================")
    plt.show()