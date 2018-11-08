import numpy as np
import pandas as pd
import sys
import os
#sys.path.append('/Volumes/DATAPEN/Shared/EHT/EHTIM/eht-imaging/')
#sys.path.append('/Volumes/DATAPEN/Shared/EHT/EHTIM/eht-imaging-workshop/eht-imaging/')
import ehtim as eh

def prepare_data(paths,init_avg='no',flag_snr='no',flag_anomal='no',sys_noise=0.0,order_stations=True, final_avg='no',flag_zero=False):

    #cou = 0
    obsL = [eh.obsdata.load_uvfits(pathf) for pathf in paths]
    
    print('initial data in obsL')
    for x in range(len(obsL)):
        print(np.shape(obsL[x].data))

    cou=0
    for pathf in paths:       
        # make parameters uniform and shift time
        if cou>0:
            obsL[cou].mjd = obsL[0].mjd
            obsL[cou].rf = obsL[0].rf
            obsL[cou].ra = obsL[0].ra
            obsL[cou].dec = obsL[0].dec
            obsL[cou].bw = obsL[0].bw
            for j in range(len(obsL[cou].data)):
                obsL[cou].data[j][0] = obsL[cou].data[j][0] + float(cou)*1e-6
        cou+=1

    for cou in range(len(obsL)):
        # filtration
        if init_avg!='no':
            print("Coherently averaging the data...")
            obsL[cou] = obsL[cou].avg_coherent(init_avg)

    print('data in obsL after avg')
    for x in range(len(obsL)):
        print(np.shape(obsL[x].data))

    for cou in range(len(obsL)):
        if flag_snr!='no':
            print("\nFlagging low-snr points...")
            obsL[cou] = obsL[cou].flag_low_snr(flag_snr)

    print('data in obsL after snr cut')
    for x in range(len(obsL)):
        print(np.shape(obsL[x].data))

    for cou in range(len(obsL)):
        if flag_anomal!='no':
            print("Flagging anomalous amplitudes...")
            obsL[cou] = obsL[cou].flag_anomalous('amp',max_diff_seconds=1000.0,robust_nsigma_cut=flag_anomal)

    for cou in range(len(obsL)):
        # Order stations
        if order_stations==True:
            obsL[cou].tarr = obsL[cou].tarr[obsL[cou].tarr['site']!='SR']
            obsL[cou].reorder_tarr_snr()
    
    for cou in range(len(obsL)):
        # Add systematic noise for leakage (reminder: this must be done *after* any averaging)
        for d in obsL[cou].data:
            d[-4] = (d[-4]**2 + np.abs(sys_noise*d[-8])**2)**0.5
            d[-3] = d[-4]
            d[-2] = d[-4]
            d[-1] = d[-4]

    print('final data in obsL')
    for x in range(len(obsL)):
        print(np.shape(obsL[x].data))

    obs = eh.obsdata.merge_obs(obsL)
    
    # Flag zero baselines 
    if flag_zero==True:
        obs = obs.flag_uvdist(uv_min = 0.1e9)

    if final_avg!='no':
        obs = obs.avg_coherent(final_avg)

    return obs,obsL


    # Helper function to repeat imaging with and without blurring to assure good convergence
def converge():
    for repeat in range(3):
        imgr.init_next = imgr.out_last().blur_circ(res)
        imgr.make_image_I(show_updates=False)
        for repeat2 in range(3):
            imgr.init_next = imgr.out_last()
            imgr.make_image_I(show_updates=False)




def prepare_obs(pathtodata, t_avg=0., flag_outl=5.,snr_cut=1.,flag_zbl=False,shift_time=0,sys_noise=0.0):

    obs = eh.obsdata.load_uvfits(pathtodata)

    # Flag problematic data
    if flag_outl!='no':
        print("Flagging anomalous amplitudes...")
        obs = obs.flag_anomalous('amp',max_diff_seconds=1200.0,robust_nsigma_cut=flag_outl)

    # Do additional averaging
    if t_avg > 0:
        print("Coherently averaging the data...")
        obs = obs.avg_coherent(t_avg)
        # Flag problematic data again
        if flag_outl:
            print("Flagging anomalous amplitudes once again...")
            obs = obs.flag_anomalous('amp',max_diff_seconds=1200.0,robust_nsigma_cut=flag_outl)

    # Drop low-snr points
    if snr_cut > 0:
        print("\nFlagging low-snr points...")
        obs = obs.flag_low_snr(snr_cut)

    # Flag zero baselines 
    if flag_zbl==True:
        obs = obs.flag_uvdist(uv_min = 0.1e9)

    if sys_noise!='no':
        # Add systematic noise for leakage (reminder: this must be done *after* any averaging)
        for d in obs.data:
            d[-4] = (d[-4]**2 + np.abs(sys_noise*d[-8])**2)**0.5
            d[-3] = d[-4]
            d[-2] = d[-4]
            d[-1] = d[-4]

    for j in range(len(obs.data)):
        obs.data[j][0] = obs.data[j][0] + shift_time

    return obs

# Helper function to repeat imaging with and without blurring to assure good convergence
def converge():
    for repeat in range(3):
        imgr.init_next = imgr.out_last().blur_circ(res)
        imgr.make_image_I(show_updates=False)
        for repeat2 in range(3):
            imgr.init_next = imgr.out_last()
            imgr.make_image_I(show_updates=False)

            
def imaging1(obs,fov=160*eh.RADPERUAS,prior_fwhm = 80*eh.RADPERUAS,npix = 128,zbl = 0.9):

    #zbl  = 0.9                   # Total compact flux density (Jy)
    flag_zbl = False             # Option to flag zero baselines
    sys_noise = 0.05             # Systematic noise added to visibilities to account for (e.g.,) leakage
    fit_amps  = True             # Whether or not to include visibility amplitudes in the imaging
    flag_amps = False            # Whether to flag anomalous amplitudes
    #snr_cut   = 0                # SNR cutoff
    #t_avg     = 180              # coherent averaging time (seconds)
    #npix = 128                   # number of pixels across the reconstructed image
    #fov = 160*eh.RADPERUAS       # field of view of the reconstructed image
    #prior_fwhm = 80*eh.RADPERUAS # Gaussian prior size
    LZ_gauss = 40*eh.RADPERUAS   # Gaussian FWHM for self-calibration of the LMT-SMT baseline
    systematic_noise = {'AA':0.1, 'AP':0.1, 'AZ':0.1, 'LM':0.3, 'PV':0.2, 'SM':0.1, 'JC':0.1} # systematic noise on a priori amplitudes
    reg_term = {'simple':10, 'tv2':10} # Image regularization parameters
    if fit_amps:
        data_term={'amp':20, 'cphase':100, 'logcamp':100}
    else:
        data_term={'cphase':100, 'logcamp':100}

    res = obs.res() # nominal array resolution, 1/longest baseline


    # Helper function to repeat imaging with and without blurring to assure good convergence
    def converge():
        for repeat in range(3):
            imgr.init_next = imgr.out_last().blur_circ(res)
            imgr.make_image_I(show_updates=False)
            for repeat2 in range(3):
                imgr.init_next = imgr.out_last()
                imgr.make_image_I(show_updates=False)

    # Make a Gaussian prior
    gaussprior    = eh.image.make_square(obs, npix, fov).add_gauss(zbl, (prior_fwhm, prior_fwhm, 0, 0, 0))
    gausspriorLMT = eh.image.make_square(obs, npix, fov).add_gauss(zbl, (LZ_gauss, LZ_gauss, 0, 0, 0))

    # Self calibrate the LMT to a Gaussian model 
    #print("Self-calibrating the LMT to a Gaussian model...")
    #for repeat in range(3):
    #    caltab = eh.self_cal.self_cal(obs.flag_uvdist(uv_max=2e9), gausspriorLMT, sites=['LM','LM'], method='amp', ttype='nfft', processes=4, caltable=True, gain_tol=1.0)
    #    obs = caltab.applycal(obs, interp='nearest', extrapolate=True)

    #eh.comp_plots.plotall_obs_im_compare(obs, gausspriorLMT, 'uvdist', 'amp')

    # Make an image -- with visibility amplitudes
    print("Imaging...")
    imgr = eh.imager.Imager(obs, gaussprior, prior_im=gaussprior, data_term=data_term, maxit=200, clipfloor=-1., norm_reg=True, systematic_noise=systematic_noise, reg_term = reg_term, ttype='nfft')
    imgr.make_image_I(show_updates=True)
    converge()

    # Check the closure phase chi^2 after dropping low-snr points
    print("Closure phase chi^2 after flagging low-snr points:",obs.flag_low_snr(3).chisq(imgr.out_last(),dtype='cphase',ttype='nfft'))

    # Store this image for later reference
    im1 = imgr.out_last().copy()

    # Self calibrate to the previous model (phase-only)
    im = imgr.out_last()
    obs_sc = eh.self_cal.self_cal(obs, im, method='phase', ttype='nfft')
    obs_sc = eh.self_cal.self_cal(obs_sc, im, method='phase', ttype='nfft')

    # Make an image -- now with complex visibilities
    imgr = eh.imager.Imager(obs_sc, gaussprior, prior_im=gaussprior, data_term={'vis':20, 'cphase':100, 'logcamp':100}, maxit=200, clipfloor=-1., norm_reg=True, systematic_noise=systematic_noise, reg_term = reg_term, ttype='nfft')
    imgr.make_image_I(show_updates=True)
    converge()

    # Store this image for later reference
    im2 = imgr.out_last().copy()

    # Self calibrate to the previous model (amplitude and phase)
    im = imgr.out_last()
    obs_sc = eh.self_cal.self_cal(obs_sc, im, method='phase', ttype='nfft')
    obs_sc = eh.self_cal.self_cal(obs_sc, im, method='phase', ttype='nfft')
    obs_sc = eh.self_cal.self_cal(obs_sc, im, method='amp', ttype='nfft')
    obs_sc = eh.self_cal.self_cal(obs_sc, im, method='amp', ttype='nfft')
    eh.comp_plots.plotall_obs_im_compare(obs_sc,im,'uvdist','amp')
    eh.comp_plots.plotall_obs_im_compare(obs_sc.flag_low_snr(5),im,'uvdist','amp')

    # Make an image -- now with complex visibilities; common systematic noise
    imgr = eh.imager.Imager(obs_sc, gaussprior, prior_im=gaussprior, data_term={'vis':20, 'cphase':100, 'logcamp':100}, maxit=200, clipfloor=-1., norm_reg=True, systematic_noise=0.05, reg_term = reg_term, ttype='nfft')
    imgr.make_image_I(show_updates=True)
    converge()

    # This is the final image
    im3 = imgr.out_last().copy()
    #im3.display(export_pdf='M87_' + expt + '_' + band + '_' + pol + '.pdf')
    #im3.save_fits('M87_' + expt + '_' + band + '_' + pol + '.fits')

    #obs_sc.save_uvfits('M87_maciek_Wednesday_3601_LL_lo.uvfits')
    #im3.save_fits('M87_maciek_Wednesday_3601_LL_lo.fits') 

    for dtype in ['vis', 'amp', 'cphase','logcamp']:
        chisq_nfft = obs_sc.chisq(im3, dtype=dtype)
        print("\n\n")
        print(dtype, chisq_nfft)

    #obs_sc.save_uvfits(name+'.uvfits')
    #im3.save_fits(name+'.fits')

    return im3, obs_sc


def shift_time(obs,shift_time=0,mjd='no',rf='no',ra='no',dec='no',bw='no'):

    if mjd!='no':
        obs.mjd = mjd
    if rf!='no':
        obs.rf = rf
    if ra!='no':
        obs.ra = ra
    if dec!='no':
        obs.dec = dec
    if bw!='no':
        obs.bw =bw
    for j in range(len(obs.data)):
        obs.data[j][0] = obs.data[j][0] + shift_time

    return obs



def load_data(pathtodata,date, merge=True, pol=None, band=None):
    AVGTIME = 180
    SNRCUT = 1
    UVMIN = 1.e8
    SOURCE='M87'
    #print(pathtodata)
    # Load R and L data
    obsLlo = eh.obsdata.load_uvfits(pathtodata + 'lo/hops_' + str(date) + '_' + SOURCE + '.LL+netcal.uvfits')
    obsRlo = eh.obsdata.load_uvfits(pathtodata + 'lo/hops_' + str(date) + '_' + SOURCE + '.RR+netcal.uvfits')
    obsLhi = eh.obsdata.load_uvfits(pathtodata + 'hi/hops_' + str(date) + '_' + SOURCE + '.LL+netcal.uvfits')
    obsRhi = eh.obsdata.load_uvfits(pathtodata + 'hi/hops_' + str(date) + '_' + SOURCE + '.RR+netcal.uvfits')

    # Noise rescale factors
    facRlo = obsRlo.estimate_noise_rescale_factor(max_diff_sec=300.)
    obsRlo = obsRlo.rescale_noise(noise_rescale_factor=facRlo)

    facLlo = obsLlo.estimate_noise_rescale_factor(max_diff_sec=300.)
    obsLlo = obsLlo.rescale_noise(noise_rescale_factor=facLlo)

    facRhi = obsRhi.estimate_noise_rescale_factor(max_diff_sec=300.)
    obsRhi = obsRhi.rescale_noise(noise_rescale_factor=facRhi)

    facLhi = obsLhi.estimate_noise_rescale_factor(max_diff_sec=300.)
    obsLhi = obsLhi.rescale_noise(noise_rescale_factor=facLhi)

    # Average and flag
    #obsRlo = obsRlo.avg_coherent_old(AVGTIME)
    obsRlo = obsRlo.avg_coherent(AVGTIME)
    obsRlo = obsRlo.flag_low_snr(SNRCUT)
    obsRlo = obsRlo.flag_anomalous(field='amp', max_diff_seconds=300)
    obsRlo = obsRlo.flag_uvdist(uv_min = UVMIN)

    #obsLlo = obsLlo.avg_coherent_old(AVGTIME)
    obsLlo = obsLlo.avg_coherent(AVGTIME)
    obsLlo = obsLlo.flag_low_snr(SNRCUT)
    obsLlo = obsLlo.flag_anomalous(field='amp', max_diff_seconds=300)
    obsLlo = obsLlo.flag_uvdist(uv_min = UVMIN)

    #obsRhi = obsRhi.avg_coherent_old(AVGTIME)
    obsRhi = obsRhi.avg_coherent(AVGTIME)
    obsRhi = obsRhi.flag_low_snr(SNRCUT)
    obsRhi = obsRhi.flag_anomalous(field='amp', max_diff_seconds=300)
    obsRhi = obsRhi.flag_uvdist(uv_min = UVMIN)

    #obsLhi = obsLhi.avg_coherent_old(AVGTIME)
    obsLhi = obsLhi.avg_coherent(AVGTIME)
    obsLhi = obsLhi.flag_low_snr(SNRCUT)
    obsLhi = obsLhi.flag_anomalous(field='amp', max_diff_seconds=300)
    obsLhi = obsLhi.flag_uvdist(uv_min = UVMIN)

    # Merge the L and R data
    for j in range(len(obsRlo.data)):
        obsRlo.data[j][0] = obsRlo.data[j][0] + 1e-6
    for j in range(len(obsRhi.data)):
        obsRhi.data[j][0] = obsRhi.data[j][0] + 2e-6
    for j in range(len(obsLhi.data)):
        obsLhi.data[j][0] = obsLhi.data[j][0] + 3e-6

    # to join the data they must have the same parameters
    obsLhi.mjd = obsLlo.mjd
    obsRhi.mjd = obsLlo.mjd
    obsRlo.mjd = obsLlo.mjd
    obsLhi.rf = obsLlo.rf
    obsRhi.rf = obsLlo.rf
    obsRlo.rf = obsLlo.rf
    obsLhi.ra = obsLlo.ra
    obsRhi.ra = obsLlo.ra
    obsRlo.ra = obsLlo.ra
    obsLhi.dec = obsLlo.dec
    obsRhi.dec = obsLlo.dec
    obsRlo.dec = obsLlo.dec
    obsLhi.bw = obsLlo.bw
    obsRhi.bw = obsLlo.bw
    obsRlo.bw = obsLlo.bw


    if merge:
        obs = eh.obsdata.merge_obs([obsLlo, obsRlo, obsLhi, obsRhi])
    elif pol is None and band is None:
        # merge
        if date==3597:
            obs = obsRlo.copy()
        if date==3598:
            obs = obsRlo.copy()
        if date==3599:
            obs = obsRlo.copy()
        if date==3600:
            obs = obsLlo.copy()
        if date==3601:
            obs = obsLlo.copy()
    else:
        if pol=='R' and band=='lo':
            obs = obsRlo.copy()
        if pol=='L' and band=='lo':
            obs = obsLlo.copy()
        if pol=='R' and band=='hi':
            obs = obsRhi.copy()
        if pol=='L' and band=='hi':
            obs = obsLhi.copy()

    return obs

'''
def imaging_Andrew_Thu():

    #def make_image(zbl_frac):
for zbl_frac in  ZBL_FRACS:
    zbl = zbl_tot * zbl_frac

    # load the observation
    obs = load_data(REFDATE)
    obs_static = obs.copy()

    # Make a Gaussian prior
    npix = 64
    fov = 200*eh.RADPERUAS
    prior_fwhm = 100*eh.RADPERUAS # Gaussian size in microarcssec
    emptyprior = eh.image.make_square(obs, npix, fov)
    gaussprior = emptyprior.add_gauss(zbl, (prior_fwhm, prior_fwhm, -np.pi/4, 0, 0))

    # Drop the LMT
    obs = obs_static.flag_sites('LM')

    #Figure out the beam
    beamparams = obs.fit_beam() # fitted beam parameters (fwhm_maj, fwhm_min, theta) in radians
    res = obs.res() # nominal array resolution, 1/longest baseline

    # Make an image, not using LMT
    obs.reorder_tarr_snr()
    imgr = eh.imager.Imager(obs, gaussprior, prior_im=gaussprior, 
                            clipfloor=0,
                            stop=1.e-10,
                            data_term={'amp':100, 'cphase':100, 'logcamp':100}, 
                            maxit=200, 
                            flux=zbl , 
                            systematic_noise=SYSTEMATIC_NOISE, 
                            norm_reg=True, 
                            reg_term={'gs':1, 'tv2':10, 'l1':5, 'flux':5}, 
                            ttype='nfft')

    for repeat in range(MAJ_ITER):
        for repeat2 in range(MIN_ITER):
            imgr.make_image_I(show_updates=show_updates, update_interval=update_interval)
            imgr.init_next = imgr.out_last()
        imgr.init_next = imgr.out_last().blur_circ(res)

    # save the reference image
    outref_v1 = imgr.out_last().copy()
    outref_v1.save_fits(pathtooutput + SCRIPTNAME +str(REFDATE)+'REF.fits')

    # now image all days with the reference as initial self-cal
    for date in ALLDAYS:
        # load the observation
        obs = load_data(date)
        obs_static = obs.copy()

        # self calilbrate to the reference image
        obs =  eh.self_cal.self_cal(obs_static, outref_v1, ttype='nfft', processes=PROCESSES, gain_tol=GAINTOL,msgtype='casa')
        obs =  eh.self_cal.self_cal(obs, outref_v1, ttype='nfft', processes=PROCESSES, gain_tol=GAINTOL,msgtype='casa')

        # coherently average again
        # obs = obs.avg_coherent_old(5*60)

        #Figure out the beam
        beamparams = obs.fit_beam() # fitted beam parameters (fwhm_maj, fwhm_min, theta) in radians
        res = obs.res() # nominal array resolution, 1/longest baseline

        # Make an image
        obs.reorder_tarr_snr()
        imgr = eh.imager.Imager(obs, outref_v1.blur_circ(res), outref_v1.blur_circ(res), 
                                clipfloor=0,
                                stop=STOP,
                                data_term={'cphase':100000, 'logcamp':50000, 'amp':10000}, 
                                reg_term={'gs':1, 'tv2':200, 'tv':100, 'l1':50, 'flux':5}, 
                                norm_reg=True, 
                                maxit=500, 
                                flux=zbl , 
                                systematic_noise=SYSTEMATIC_NOISE, 
                                ttype='nfft')

        for repeat in range(MAJ_ITER):
            for repeat2 in range(MIN_ITER):
                imgr.make_image_I(show_updates=show_updates, update_interval=update_interval)
                imgr.init_next = imgr.out_last()
            imgr.init_next = imgr.out_last().blur_circ(0.5*res)
        out_v1 = imgr.out_last().copy()

        # Self calibrate and reimage
        obs = eh.self_cal.self_cal(obs_static, out_v1, ttype='nfft', processes=PROCESSES, gain_tol=GAINTOL)
        obs = eh.self_cal.self_cal(obs, out_v1, ttype='nfft', processes=PROCESSES, gain_tol=GAINTOL)

        imgr.init_next = out_v1.blur_circ(res)
        imgr.maxit_next = 100
        imgr.dat_term_next={'cphase':100000, 'logcamp':50000, 'amp':10000}
        imgr.reg_term_next={'gs':1, 'tv2':200, 'tv':100, 'l1':50}
        imgr.obs_next = obs
        for repeat in range(MAJ_ITER):
            for repeat2 in range(MIN_ITER):
                imgr.make_image_I(show_updates=show_updates, update_interval=update_interval)
                imgr.init_next = imgr.out_last()
            imgr.init_next = imgr.out_last().blur_circ(0.33*res)
        out_v2 = imgr.out_last().copy()

        # Make a caltable
        obs_tmp = obs_static.copy()
        for i in range(3): 
            ct = eh.self_cal.self_cal(obs_tmp, out_v2, 
                                      method='amp', ttype='nfft', 
                                      caltable=True, gain_tol=.5,
                                      processes=PROCESSES)
            ct = ct.pad_scans()
            obs_tmp =  ct.applycal(obs_tmp,interp='nearest',extrapolate=True) #apply caltable
            if np.any(np.isnan(obs_tmp.data['vis'])):
                print "Warning: NaN in applycal vis table!"
                break
            if i>0:
                ct_out = ct_out.merge([ct])
            else:
                ct_out = ct

        # make a final selfcaled data set
        #obs_sc_out = ct_out.applycal(obs_static)
        obs_sc_out = eh.self_cal.self_cal(obs_tmp, out_v2, method='both',ttype='nfft',processes=PROCESSES, gain_tol=GAINTOL) # re self-calibrate
        #obs_sc_out = eh.self_cal.self_cal(obs_static, out_v2, method='amp',ttype='nfft',processes=PROCESSES, gain_tol=GAINTOL) # re self-calibrate

        zblstr = '%0.2f'%zbl_frac
        ct_out.save_txt(obs_sc_out, datadir=pathtooutput + SCRIPTNAME +str(date)+str(pol)+str(band)+'_'+zblstr+'caltable')
        ct_out.plot_gains('all', yscale='log', export_pdf=pathtooutput + SCRIPTNAME +str(date)+str(pol)+str(band)+'_'+zblstr+'_gains.pdf',rangey=[.1,10],show=False)
        
        fitsfile = pathtooutput + SCRIPTNAME +str(date)+str(pol)+str(band)+'_'+zblstr+'.fits'
        uvfitsfile = pathtooutput + SCRIPTNAME +str(date)+str(pol)+str(band)+'_'+zblstr+'.uvfits'
        out_v2.save_fits(fitsfile)
        obs_sc_out.save_uvfits(uvfitsfile)

        subprocess.call(['python', imgsum, fitsfile, uvfitsfile,'-o',pathtooutput])
'''



def imaging_Ramesh(obs):


    ###################################################
    # ramesh: July 15, 2018
    #
    # general purpose singleframe script, based on a script developed by Michael J
    #
    ###################################################


    import numpy as np
    import ehtim as eh
    import matplotlib.pyplot as plt
    import os


    ##################################################
    # Tunable parameters

    name = 'M87'               # Name of saved files
    band = 'lo'                  # Observing band; 'lo' or 'hi'
    expt = '3601'                # (3601) HOPS Experiment code
    pol  = 'L'                   # (L) Polarization to use; 'L' or 'R'

    name_folder = '../../../' + name + '/er4v2/data/'  # Name of data folder
    name_savefile = name + '_' + expt + '_' + band + '_' + pol
    pathtodata = name_folder + band + '/'
    #pathtooutput = './'

    zbl  = 0.8                   # Total compact flux density (Jy)
    zbl_tot = 1.2                # Total compact + extended flux density (Jy)
    flag_zbl = False             # Option to flag zero baselines

    fit_amps  = True             # Whether or not to include visibility amplitudes in the imaging
    flag_amps = False            # Whether to flag anomalous amplitudes

    rescale_noise = False         # Whether or not to rescale noise
    snr_cut   = 0                # SNR cutoff
    clip_floor = -1.             # use 1.e-10 to clip out zero intensity regions
    t_avg     = 180              # coherent averaging time (seconds)
    npix = 128                   # number of pixels across the reconstructed image

    maxit = 200

    amp_cal_all = False          # Whether to calibrate amplitudes of all stations
    amp_cal_LMT = True           # Whether to calibrate amplitudes of LMT only

    fov = 160*eh.RADPERUAS       # field of view of the reconstructed image

    prior_fwhm_maj = 80*eh.RADPERUAS # major axis of Gaussian prior size
    prior_fwhm_min = 80*eh.RADPERUAS # minor axis of Gaussian prior size
    prior_fwhm_angle = -0.6 # major axis angle of Gaussian prior size
    prior_xshift = 0.*eh.RADPERUAS # Gaussian prior x-center
    prior_yshift = 0.*eh.RADPERUAS # Gaussian prior y-center

    prior_gauss_maj = 40*eh.RADPERUAS # major axis of Gaussian regularizer
    prior_gauss_min = 40*eh.RADPERUAS # minor axis of Gaussian regularizer
    prior_gauss_pa = 0                # pa of Gaussian regularizer

    systematic_noise = {'AA':0.1, 'AP':0.1, 'AZ':0.1, 'LM':2., 'PV':0.1, 'SM':0.1, 'JC':0.1, 'SP':0.1} # systematic noise on a priori amplitudes
    sys_noise = 0.0             #  (0.05) Systematic noise added to visibilities to account for (e.g.,) leakage

    #reg_term = {'simple':1, 'tv2':10, 'l1':10, 'compact2':1.e24} # Image regularization parameters
    reg_term = {'simple':10, 'tv2':10, 'l1':10, 'rgauss':1.e5} # Image regularization parameters

    if fit_amps:
        data_term={'amp':10, 'cphase':100, 'logcamp':100}
    else:
        data_term={'cphase':100, 'logcamp':100}

    data_term_vis={'vis':20, 'cphase':100, 'logcamp':100}

    # end tunable parameters
    ##################################################

    # Load the data and rescale flux and noise

    obs_orig = obs.copy()

    res = obs.res() # nominal array resolution, 1/longest baseline


    # Look at all the observations

    obs.plotall('uvdist','amp')

    # Compare the (u,v) coordinates for flagged and averaged amplitudes to the original data

    eh.comp_plots.plotall_obs_compare([obs_orig,obs],'u','v',conj=True)

    # Helper function to repeat imaging with and without blurring to assure good convergence

    def converge():
        for repeat in range(3):
            imgr.init_next = imgr.out_last().blur_circ(res)
            imgr.make_image_I(show_updates=False)
            for repeat2 in range(3):
                imgr.init_next = imgr.out_last()
                imgr.make_image_I(show_updates=False)

    # Make a Gaussian prior

    gaussprior    = eh.image.make_square(obs, npix, fov).add_gauss(zbl, (prior_fwhm_maj, prior_fwhm_min, prior_fwhm_angle, prior_xshift, prior_yshift))

    # Make an image -- with visibility amplitudes

    print("Imaging...")
    imgr = eh.imager.Imager(obs, gaussprior, prior_im=gaussprior, data_term=data_term, maxit=maxit, clipfloor=clip_floor, norm_reg=True, systematic_noise=systematic_noise, reg_term = reg_term, ttype='nfft', major = prior_gauss_maj, minor = prior_gauss_min, PA = prior_gauss_pa)
    imgr.make_image_I(show_updates=True)
    converge()

    # Check the closure phase chi^2 after dropping low-snr points

    print("Closure phase chi^2 after flagging low-snr points:",obs.flag_low_snr(3).chisq(imgr.out_last(),dtype='cphase',ttype='nfft'))

    # Store this image for later reference

    im1 = imgr.out_last().copy()

    # Self calibrate to the previous model (phase-only)

    im = imgr.out_last()
    print('\nFIT_GAUSS:', im.fit_gauss(units = 'natural'), '\n')

    obs_sc = eh.self_cal.self_cal(obs, im, method='phase', ttype='nfft', processes=8)
    obs_sc = eh.self_cal.self_cal(obs_sc, im, method='phase', ttype='nfft', processes=8)

    # Make an image -- now with complex visibilities

    imgr = eh.imager.Imager(obs_sc, gaussprior, prior_im=gaussprior, data_term={'vis':20, 'cphase':100, 'logcamp':100}, maxit=maxit, clipfloor=clip_floor, norm_reg=True, systematic_noise=systematic_noise, reg_term = reg_term, ttype='nfft', major = prior_gauss_maj, minor = prior_gauss_min, PA = prior_gauss_pa)
    imgr.make_image_I(show_updates=True)
    converge()

    # Store this image for later reference

    im2 = imgr.out_last().copy()

    # Self calibrate to the previous model (amplitude and phase)

    im = imgr.out_last()
    print('\nFIT_GAUSS:', im.fit_gauss(units = 'natural'), '\n')

    obs_sc = eh.self_cal.self_cal(obs_sc, im, method='phase', ttype='nfft', processes=8)
    obs_sc = eh.self_cal.self_cal(obs_sc, im, method='phase', ttype='nfft', processes=8)

    if amp_cal_all:
        obs_sc = eh.self_cal.self_cal(obs_sc, im, method='amp', ttype='nfft', processes=8)
        obs_sc = eh.self_cal.self_cal(obs_sc, im, method='amp', ttype='nfft', processes=8)

    if amp_cal_LMT:
        obs_sc = eh.self_cal.self_cal(obs_sc, im, sites=['LM','LM'], method='amp', ttype='nfft', processes=8)
        obs_sc = eh.self_cal.self_cal(obs_sc, im, sites=['LM','LM'], method='amp', ttype='nfft', processes=8)

    obs_sc = eh.self_cal.self_cal(obs_sc, im, method='phase', ttype='nfft', processes=8)
    obs_sc = eh.self_cal.self_cal(obs_sc, im, method='phase', ttype='nfft', processes=8)

    eh.comp_plots.plotall_obs_im_compare(obs_sc,im,'uvdist','amp')
    eh.comp_plots.plotall_obs_im_compare(obs_sc.flag_low_snr(5),im,'uvdist','amp')

    # Make an image -- now with complex visibilities; common systematic noise

    imgr = eh.imager.Imager(obs_sc, gaussprior, prior_im=gaussprior, data_term=data_term_vis, maxit=maxit, clipfloor=clip_floor, norm_reg=True, systematic_noise=sys_noise, reg_term = reg_term, ttype='nfft', major = prior_gauss_maj, minor = prior_gauss_min, PA = prior_gauss_pa)
    imgr.make_image_I(show_updates=True)
    converge()

    # Store this image for later reference

    im2p = imgr.out_last().copy()

    # Self calibrate to the previous model (amplitude and phase)

    im = imgr.out_last()
    print('\nFIT_GAUSS:', im.fit_gauss(units = 'natural'), '\n')

    obs_sc = eh.self_cal.self_cal(obs_sc, im, method='phase', ttype='nfft', processes=8)
    obs_sc = eh.self_cal.self_cal(obs_sc, im, method='phase', ttype='nfft', processes=8)

    if amp_cal_all:
        obs_sc = eh.self_cal.self_cal(obs_sc, im, method='amp', ttype='nfft', processes=8)
        obs_sc = eh.self_cal.self_cal(obs_sc, im, method='amp', ttype='nfft', processes=8)

    if amp_cal_LMT:
        obs_sc = eh.self_cal.self_cal(obs_sc, im, sites=['LM','LM'], method='amp', ttype='nfft', processes=8)
        obs_sc = eh.self_cal.self_cal(obs_sc, im, sites=['LM','LM'], method='amp', ttype='nfft', processes=8)

    obs_sc = eh.self_cal.self_cal(obs_sc, im, method='phase', ttype='nfft', processes=8)
    obs_sc = eh.self_cal.self_cal(obs_sc, im, method='phase', ttype='nfft', processes=8)

    eh.comp_plots.plotall_obs_im_compare(obs_sc,im,'uvdist','amp')
    eh.comp_plots.plotall_obs_im_compare(obs_sc.flag_low_snr(5),im,'uvdist','amp')

    # Make an image -- now with complex visibilities; common systematic noise

    imgr = eh.imager.Imager(obs_sc, gaussprior, prior_im=gaussprior, data_term=data_term_vis, maxit=maxit, clipfloor=clip_floor, norm_reg=True, systematic_noise=sys_noise, reg_term = reg_term, ttype='nfft', major = prior_gauss_maj, minor = prior_gauss_min, PA = prior_gauss_pa)
    imgr.make_image_I(show_updates=True)
    converge()

    # This is the final image

    im3 = imgr.out_last().copy()
    print('\nFIT_GAUSS:', im3.fit_gauss(units = 'natural'), '\n')
    # Self-calibrate once again and compute chisq etc.

    obs_sc = eh.self_cal.self_cal(obs_sc, im3, method='phase', ttype='nfft', processes=8)
    obs_sc = eh.self_cal.self_cal(obs_sc, im3, method='both', ttype='nfft', processes=8)
    eh.comp_plots.plotall_obs_im_compare(obs_sc,im3,'uvdist','amp')
    eh.comp_plots.plotall_obs_im_compare(obs_sc,im3,'uvdist','phase')

    for dtype in ['cphase','logcamp','vis','amp']:
        print('\nFinal '+dtype+' chisq', obs_sc.chisq(im3,dtype=dtype))

    # Display and save fits data

    im3.display(export_pdf=name_savefile + '_image.pdf')
    im3.save_fits(name_savefile + '.fits')

    # Write out other stuff

    ct = eh.self_cal.self_cal(obs, imgr.out_last(), method='phase', ttype='nfft', processes=8, caltable=True)
    ct = eh.self_cal.self_cal(obs, imgr.out_last(), method='both', ttype='nfft', processes=8, caltable=True)
    ct.plot_gains(list(np.sort(list(ct.data.keys()))), yscale='log', export_pdf=name_savefile + '_gains.pdf')
    obs_sc = ct.applycal(obs)
    obs_sc.save_uvfits(name_savefile + '.uvfits')

    os.system('python ../../../imgsum.py ./' + name_savefile + '.fits ./' + name_savefile + '.uvfits  -o ./' )

    im3_flux = im3.total_flux()
    print ("im3 flux: ", im3_flux)

    mask = im3.mask(cutoff=0.3, beamparams=1.5e-10, frac=1.)
    mask.display()

    im3_masked = im3.apply_mask(mask, fill_val=0.)
    im3_masked.display()
    im3_masked_flux = im3_masked.total_flux()
    print ("im3_masked flux: ", im3_masked_flux)


def imaging_Michael_univ(obs,zbl=1.1,fov=160,npix=128,t_avg=20,flag_zbl=False,reg_term = {'simple':10, 'tv2':10}):
    fovuas=fov
    # Tunable parameters
    band = 'lo'                  # Observing band; 'lo' or 'hi'
    expt = '3598'                # HOPS Experiment code
    pol  = 'R'                   # Polarization to use; 'L' or 'R'
    ##############
    #zbl  = 1.1                   # Total compact flux density (Jy)
    flag_zbl = flag_zbl             # Option to flag zero baselines
    sys_noise = 0.05             # Systematic noise added to visibilities to account for (e.g.,) leakage
    fit_amps  = True             # Whether or not to include visibility amplitudes in the imaging
    flag_amps = False            # Whether to flag anomalous amplitudes
    snr_cut   = 0                # SNR cutoff
    #t_avg     = 20              # coherent averaging time (seconds)
    npix = npix                  # number of pixels across the reconstructed image
    fov = fovuas*eh.RADPERUAS       # field of view of the reconstructed image
    prior_fwhm = (fovuas/2)*eh.RADPERUAS # Gaussian prior size
    LZ_gauss = (fovuas/4)*eh.RADPERUAS   # Gaussian FWHM for self-calibration of the LMT-SMT baseline
    systematic_noise = {'AA':0.1, 'AP':0.1, 'AZ':0.1, 'LM':0.3, 'PV':0.2, 'SM':0.1, 'JC':0.1} # systematic noise on a priori amplitudes
    reg_term = reg_term # Image regularization parameters

    if fit_amps:
        data_term={'amp':20, 'cphase':100, 'logcamp':100}
    else:
        data_term={'cphase':100, 'logcamp':100}

    # I/O variables
    SCRIPTNAME='M87singleframe_v1-07.06.18'
    pathtodata = '../../../M87/er4v1/data/' + band + '/'
    pathtooutput = './'

    # Load the data
    #obs = eh.obsdata.load_uvfits(pathtodata + 'hops_' + expt + '_M87.' + pol + pol + '+netcal.uvfits')
    obs_orig = obs.copy()

    res = obs.res() # nominal array resolution, 1/longest baseline

    # Flag problematic data
    if flag_amps:
        print("Flagging anomalous amplitudes...")
        obs = obs.flag_anomalous('amp',max_diff_seconds=1200.0)

    # Do additional averaging
    print("Coherently averaging the data...")
    obs = obs.avg_coherent(t_avg)

    # Drop low-snr points
    if snr_cut > 0:
        print("\nFlagging low-snr points...")
        obs = obs.flag_low_snr(snr_cut)

    # Flag problematic data again
    if flag_amps:
        print("Flagging anomalous amplitudes...")
        obs = obs.flag_anomalous('amp',max_diff_seconds=1200.0)

    # Flag zero baselines 
    if flag_zbl:
        obs = obs.flag_uvdist(uv_min = 0.1e9)

    # Order stations
    obs.tarr = obs.tarr[obs.tarr['site']!='SR']
    #obs.reorder_tarr_snr()

    # Add systematic noise for leakage (reminder: this must be done *after* any averaging)
    for d in obs.data:
        d[-4] = (d[-4]**2 + np.abs(sys_noise*d[-8])**2)**0.5
        d[-3] = d[-4]
        d[-2] = d[-4]
        d[-1] = d[-4]

    # Look at all the observations
    obs.plotall('uvdist','amp')
    # Compare the (u,v) coordinates for flagged and averaged amplitudes to the original data
    eh.comp_plots.plotall_obs_compare([obs_orig,obs],'u','v',conj=True)

    # Helper function to repeat imaging with and without blurring to assure good convergence
    def converge():
        for repeat in range(5):
            imgr.init_next = imgr.out_last().blur_circ(res)
            imgr.make_image_I(show_updates=False)
            for repeat2 in range(5):
                imgr.init_next = imgr.out_last()
                imgr.make_image_I(show_updates=False)

    # Make a Gaussian prior
    gaussprior    = eh.image.make_square(obs, npix, fov).add_gauss(zbl, (prior_fwhm, prior_fwhm, 0, 0, 0))
    gausspriorLMT = eh.image.make_square(obs, npix, fov).add_gauss(zbl, (LZ_gauss, LZ_gauss, 0, 0, 0))
    #gaussprior.display()
    # Self calibrate the LMT to a Gaussian model 
    print("Self-calibrating the LMT to a Gaussian model...")
    for repeat in range(3):
        caltab = eh.self_cal.self_cal(obs.flag_uvdist(uv_max=2e9), gausspriorLMT, sites=['LM','LM'], method='vis', ttype='nfft', processes=4, caltable=True, gain_tol=1.0)
        obs = caltab.applycal(obs, interp='nearest', extrapolate=True)

    eh.comp_plots.plotall_obs_im_compare(obs, gausspriorLMT, 'uvdist', 'amp')

    # Make an image -- with visibility amplitudes
    print("Imaging...")
    imgr = eh.imager.Imager(obs, gaussprior, prior_im=gaussprior, data_term=data_term, maxit=200, clipfloor=-1., norm_reg=True, systematic_noise=systematic_noise, reg_term = reg_term, ttype='nfft')
    imgr.make_image_I(show_updates=True)
    converge()

    # Check the closure phase chi^2 after dropping low-snr points
    print("Closure phase chi^2 after flagging low-snr points:",obs.flag_low_snr(3).chisq(imgr.out_last(),dtype='cphase',ttype='nfft'))

    # Store this image for later reference
    im1 = imgr.out_last().copy()

    # Self calibrate to the previous model (phase-only)
    im = imgr.out_last()
    obs_sc = eh.self_cal.self_cal(obs, im, method='phase', ttype='nfft')
    obs_sc = eh.self_cal.self_cal(obs_sc, im, method='phase', ttype='nfft')

    # Make an image -- now with complex visibilities
    imgr = eh.imager.Imager(obs_sc, gaussprior, prior_im=gaussprior, data_term={'vis':20, 'cphase':100, 'logcamp':100}, maxit=200, clipfloor=-1., norm_reg=True, systematic_noise=systematic_noise, reg_term = reg_term, ttype='nfft')
    imgr.make_image_I(show_updates=True)
    converge()

    # Store this image for later reference
    im2 = imgr.out_last().copy()

    # Self calibrate to the previous model (amplitude and phase)
    im = imgr.out_last()
    obs_sc = eh.self_cal.self_cal(obs_sc, im, method='both', ttype='nfft')
    obs_sc = eh.self_cal.self_cal(obs_sc, im, method='both', ttype='nfft')
    eh.comp_plots.plotall_obs_im_compare(obs_sc,im,'uvdist','amp')
    eh.comp_plots.plotall_obs_im_compare(obs_sc.flag_low_snr(5),im,'uvdist','amp')

    # Make an image -- now with complex visibilities; common systematic noise
    imgr = eh.imager.Imager(obs_sc, gaussprior, prior_im=gaussprior, data_term={'vis':20, 'cphase':100, 'logcamp':100}, maxit=200, clipfloor=-1., norm_reg=True, systematic_noise=0.05, reg_term = reg_term, ttype='nfft')
    imgr.make_image_I(show_updates=True)
    converge()

    # This is the final image
    im3 = imgr.out_last().copy()

    tag = '-' + expt + '-' + pol + '-' + band

    imgr.out_last().display(export_pdf=SCRIPTNAME + tag + '.pdf')
    #imgr.out_last().save_fits(SCRIPTNAME + tag + '.fits')
    ct = eh.self_cal.self_cal(obs, imgr.out_last(), method='both', ttype='nfft', processes=0, caltable=True)
    ct.plot_gains(list(np.sort(list(ct.data.keys()))), yscale='log', export_pdf=SCRIPTNAME + tag + '_gains.pdf')
    obs_sc = ct.applycal(obs)
    #obs_sc.save_uvfits('obs_sc.uvfits')
    return obs_sc, im3
    #os.system('python3 ../../../imgsum.py ' + pathtooutput + '/' + SCRIPTNAME + tag + '.fits ' + pathtooutput + '/' + SCRIPTNAME + tag + '.uvfits  --no_ebar --o .' ) 


def image_batch_version(obs,zbl=1.,fov=160.,sys_noise=0.02,flag_zbl=False,ratio_prior=0.5,npix=128):

    ### Range of parameters to test for simple search (180 possibilities)
    ### zbl in [0.4,0.6,0.8,1.0,1.2]
    ### fovfactor in [1,2,4]
    ### reg_term weights in [1,10,100] for each regularizer used independently ('simple, 'tv', 'tv2', 'l1')
    #tstart = time.time()


    #argdict = {
    #        'l1':args.l1, 
    #        'simple': args.simple, 
    #        'tv':args.tv,
    #        'tv2':args.tv2,
    #        'fovfactor':args.fovfactor}
    #zbl       = args.zbl
    reg_term = {'simple':10, 'tv2':10}
    #reg_term  = {'simple': args.simple,
    #            'tv'    : args.tv,
    #            'tv2'   : args.tv2,
    #            'l1'    : args.l1}
    #reg_term = {}

    #fovfactor = args.fovfactor
    #outfile   = args.outfile
    # Systematic noise added to complex visibilities to account for (e.g.,) leakage
    #sys_noise = args.sys_noise


    # Additional parameters
    #obsfile   = '/Users/klbouman/Research/vlbi_imaging/data/EHT2017/eht_team_1_dropbox/Dropbox (Personal)/eht_team_1/M87/er4v2/scan_averaged/flags_subscan/hops_3601_M87.LL+netcal_lo_scan_avg.uvfits'
    #obsfile   = '/Users/klbouman/Research/vlbi_imaging/data/EHT2017/eht_team_1_dropbox/Dropbox (Personal)/eht_team_1/M87/er4v2/scan_averaged/hops_3600_M87.LL+netcal_lo_scan_avg.uvfits'
    #obsfile   = '/Users/klbouman/Research/vlbi_imaging/data/EHT2017/eht_team_1_dropbox/Dropbox (Personal)/eht_team_1/M87/er4v2/data/lo/hops_3601_M87.LL+netcal.uvfits'

    #'obs.uvfits' # Pre-processed observation  file
    ttype     = 'nfft'          # Type of Fourier transform ('direct', 'nfft', or 'fast')
    processes = 0               # Number of parallel processes for self-cal (-1 = no parallelization; 0 = max)
    LZ_gauss  = 40*eh.RADPERUAS # Gaussian FWHM for self-calibration of the LMT-SMT baseline
    #systematic_noise = {'AA':0.1,
    #                    'AP':0.1,
    #                    'AZ':0.1,
    #                    'LM':0.3,
    #                    'PV':0.2,
    #                    'SM':0.1,
    #                    'JC':0.1} # systematic noise on a priori amplitudes
    systematic_noise = {'AA':0.05,
                        'AP':0.05,
                        'AZ':0.05,
                        'LM':0.15,
                        'PV':0.1,
                        'SM':0.05,
                        'JC':0.05} # systematic noise on a priori amplitudes

    # Fixed parameters
    #print('kurwa')
    #npix       = 128          # number of pixels across the reconstructed image
    fovuas=fov
    fov        = fovuas*eh.RADPERUAS  # field of view of the reconstructed image
    prior_fwhm = fovuas*ratio_prior*eh.RADPERUAS  # Gaussian prior size
    maj_cycles = 8                          # imager loop major cycles (blurring)
    min_cycles = 3                          # imager minor cycles (no blurring)
    alpha_flux = 1e3                        # weight on the total flux
    alpha_vis = 20                          # weight on visibility amplitude / complex visibility
    alpha_cphase = 100                      # weight on closure phase chi^2 
    alpha_logcamp = 100                     # weight on log closure amplitudes chi^2
    maxit = 200                             # number of imager iterations
    stop = 1.e-10                           # convergence criterion
    transform = 'log'                        # enforce positivity ('log') or not (None)
    def converge():
        for repeat in range(maj_cycles):
            imgr.init_next = imgr.out_last().blur_circ(res)
            imgr.make_image_I(show_updates=False)

            for repeat2 in range(min_cycles):
                imgr.init_next = imgr.out_last()
                imgr.make_image_I(show_updates=False)

    ########################

    # Order stations
    obs.tarr = obs.tarr[obs.tarr['site']!='SR']
    obs.reorder_tarr_snr()

    # Flag zero baselines 
    if flag_zbl:
        obs = obs.flag_uvdist(uv_min = 0.1e9)

    #obs = obs.avg_coherent(180)

    # Add systematic noise for leakage (reminder: this must be done *after* any averaging)
    for d in obs.data:
        d[-4] = (d[-4]**2 + np.abs(sys_noise*d[-8])**2)**0.5
        d[-3] = d[-4]
        d[-2] = d[-4]
        d[-1] = d[-4]

    ########################

    # Make prior and initial image
    res = obs.res()
    gaussprior    = eh.image.make_square(obs, npix, fov).add_gauss(zbl, (prior_fwhm, prior_fwhm, 0, 0, 0))
    #gaussprior.display()
    gausspriorLMT = eh.image.make_square(obs, npix, fov).add_gauss(zbl, (LZ_gauss, LZ_gauss, 0, 0, 0))
    init = gaussprior.copy()
    np.random.seed(10) # TODO: save initial seed
    init.imvec *= (1.0 + (np.random.random_sample(len(init.imvec))-0.5)/100.0)


    # Self calibrate the LMT to a Gaussian model
    print("Self-calibrating the LMT to a Gaussian model...")
    for repeat in range(3):
        caltab = eh.self_cal.self_cal(obs.flag_uvdist(uv_max=2e9), gausspriorLMT, sites=['LM','LM'], method='vis', ttype=ttype, processes=processes, caltable=True, gain_tol=1.0)
        obs = caltab.applycal(obs, interp='nearest', extrapolate=True)

    # Make an image -- with visibility amplitudes
    print("Imaging...")
    reg_term['flux']=alpha_flux
    imgr = eh.imager.Imager(obs, init, prior_im=gaussprior, data_term={'amp':alpha_vis, 'cphase':alpha_cphase, 'logcamp':alpha_logcamp}, 
                            maxit=maxit, clipfloor=-1., norm_reg=True, systematic_noise=systematic_noise, reg_term=reg_term, ttype=ttype, transform=transform, stop=stop)
    imgr = eh.imager.Imager(obs, init, prior_im=gaussprior, data_term={'amp':alpha_vis, 'cphase':alpha_cphase, 'logcamp':alpha_logcamp}, 
                            maxit=maxit, norm_reg=True, systematic_noise=systematic_noise, reg_term=reg_term, ttype=ttype, transform=transform, stop=stop)

    imgr.make_image_I(show_updates=False)
    converge()

    # Self calibrate to the previous model (phase-only)
    im = imgr.out_last()
    #im.display()
    
    obs_sc = eh.self_cal.self_cal(obs,    im, method='phase', ttype=ttype, processes=processes)
    obs_sc = eh.self_cal.self_cal(obs_sc, im, method='phase', ttype=ttype, processes=processes)

    # Make an image -- now with complex visibilities
    reg_term['flux']=10*alpha_flux
    imgr = eh.imager.Imager(obs_sc, init, prior_im=gaussprior, data_term={'vis':alpha_vis, 'cphase':alpha_cphase, 'logcamp':alpha_logcamp}, 
                            maxit=maxit, clipfloor=-1., norm_reg=True, systematic_noise=systematic_noise, reg_term = reg_term, ttype=ttype,  transform=transform, stop=stop)
    imgr.make_image_I(show_updates=False)
    converge()

    # Self calibrate to the previous model (amplitude and phase)
    im = imgr.out_last()
    obs_sc = eh.self_cal.self_cal(obs_sc, im, method='both', ttype=ttype, processes=processes)
    obs_sc = eh.self_cal.self_cal(obs_sc, im, method='both', ttype=ttype, processes=processes)

    # Make an image -- now with complex visibilities; common systematic noise
    reg_term['flux']=10*alpha_flux
    imgr = eh.imager.Imager(obs_sc, init, prior_im=gaussprior, data_term={'vis':alpha_vis, 'cphase':alpha_cphase, 'logcamp':alpha_logcamp}, 
                            maxit=maxit, clipfloor=-1., norm_reg=True, systematic_noise=0.05, reg_term = reg_term, ttype=ttype,  transform=transform, stop=stop)
    imgr.make_image_I(show_updates=False)
    converge()

    # Final image
    im_out = imgr.out_last()
    im_out.display()

    # Save out statistics and info
    reg_dict_out = {'flux':1,'cm':1,'l1':1,'simple':1,'tv':1,'tv2':1}
    chisq_dict_out = {'vis':1, 'amp':1,'cphase':1, 'logcamp':1, 'camp':1, 'bs':1}
    imgr.dat_term_next = chisq_dict_out
    imgr.reg_term_next = reg_dict_out
    imgr._change_imgr_params = True
    imgr.init_imager_I()
    chisqdict_final = imgr.make_chisq_dict(im_out.imvec)
    regdict_final = imgr.make_reg_dict(im_out.imvec)
    objfunc_final = imgr.tmpout.fun
    success_final = imgr.tmpout.success
    msg_final = imgr.tmpout.message
    #tstop = time.time()
    return im_out


def image_simplest_0(obs,zbl=1.,fov=160.,sys_noise=0.02):

    ### Range of parameters to test for simple search (180 possibilities)
    ### zbl in [0.4,0.6,0.8,1.0,1.2]
    ### fovfactor in [1,2,4]
    ### reg_term weights in [1,10,100] for each regularizer used independently ('simple, 'tv', 'tv2', 'l1')
    #tstart = time.time()


    #argdict = {
    #        'l1':args.l1, 
    #        'simple': args.simple, 
    #        'tv':args.tv,
    #        'tv2':args.tv2,
    #        'fovfactor':args.fovfactor}
    #zbl       = args.zbl
    reg_term = {'simple':10, 'tv2':10}
    #reg_term  = {'simple': args.simple,
    #            'tv'    : args.tv,
    #            'tv2'   : args.tv2,
    #            'l1'    : args.l1}
    #reg_term = {}

    #fovfactor = args.fovfactor
    #outfile   = args.outfile
    # Systematic noise added to complex visibilities to account for (e.g.,) leakage
    #sys_noise = args.sys_noise


    # Additional parameters
    #obsfile   = '/Users/klbouman/Research/vlbi_imaging/data/EHT2017/eht_team_1_dropbox/Dropbox (Personal)/eht_team_1/M87/er4v2/scan_averaged/flags_subscan/hops_3601_M87.LL+netcal_lo_scan_avg.uvfits'
    #obsfile   = '/Users/klbouman/Research/vlbi_imaging/data/EHT2017/eht_team_1_dropbox/Dropbox (Personal)/eht_team_1/M87/er4v2/scan_averaged/hops_3600_M87.LL+netcal_lo_scan_avg.uvfits'
    #obsfile   = '/Users/klbouman/Research/vlbi_imaging/data/EHT2017/eht_team_1_dropbox/Dropbox (Personal)/eht_team_1/M87/er4v2/data/lo/hops_3601_M87.LL+netcal.uvfits'

    #'obs.uvfits' # Pre-processed observation  file
    ttype     = 'nfft'          # Type of Fourier transform ('direct', 'nfft', or 'fast')
    processes = 0               # Number of parallel processes for self-cal (-1 = no parallelization; 0 = max)
    LZ_gauss  = 40*eh.RADPERUAS # Gaussian FWHM for self-calibration of the LMT-SMT baseline
    #systematic_noise = {'AA':0.1,
    #                    'AP':0.1,
    #                    'AZ':0.1,
    #                    'LM':0.3,
    #                    'PV':0.2,
    #                    'SM':0.1,
    #                    'JC':0.1} # systematic noise on a priori amplitudes
    systematic_noise = {'AA':0.05,
                        'AP':0.05,
                        'AZ':0.05,
                        'LM':0.15,
                        'PV':0.1,
                        'SM':0.05,
                        'JC':0.05} # systematic noise on a priori amplitudes

    # Fixed parameters
    npix       = 128          # number of pixels across the reconstructed image
    fovuas=fov
    fov        = fovuas*eh.RADPERUAS  # field of view of the reconstructed image
    prior_fwhm = fovuas*0.5*eh.RADPERUAS  # Gaussian prior size
    maj_cycles = 8                          # imager loop major cycles (blurring)
    min_cycles = 3                          # imager minor cycles (no blurring)
    alpha_flux = 1e3                        # weight on the total flux
    alpha_vis = 20                          # weight on visibility amplitude / complex visibility
    alpha_cphase = 100                      # weight on closure phase chi^2 
    alpha_logcamp = 100                     # weight on log closure amplitudes chi^2
    maxit = 200                             # number of imager iterations
    stop = 1.e-10                           # convergence criterion
    transform = 'log'                        # enforce positivity ('log') or not (None)
    def converge():
        for repeat in range(maj_cycles):
            imgr.init_next = imgr.out_last().blur_circ(res)
            imgr.make_image_I(show_updates=False)

            for repeat2 in range(min_cycles):
                imgr.init_next = imgr.out_last()
                imgr.make_image_I(show_updates=False)

    ########################

    # Order stations
    obs.tarr = obs.tarr[obs.tarr['site']!='SR']
    obs.reorder_tarr_snr()

    #obs = obs.avg_coherent(180)

    # Add systematic noise for leakage (reminder: this must be done *after* any averaging)
    for d in obs.data:
        d[-4] = (d[-4]**2 + np.abs(sys_noise*d[-8])**2)**0.5
        d[-3] = d[-4]
        d[-2] = d[-4]
        d[-1] = d[-4]

    ########################

    # Make prior and initial image
    res = obs.res()
    gaussprior    = eh.image.make_square(obs, npix, fov).add_gauss(zbl, (prior_fwhm, prior_fwhm, 0, 0, 0))
    #gaussprior.display()
    gausspriorLMT = eh.image.make_square(obs, npix, fov).add_gauss(zbl, (LZ_gauss, LZ_gauss, 0, 0, 0))
    init = gaussprior.copy()
    np.random.seed(10) # TODO: save initial seed
    init.imvec *= (1.0 + (np.random.random_sample(len(init.imvec))-0.5)/100.0)


    # Self calibrate the LMT to a Gaussian model
    print("Self-calibrating the LMT to a Gaussian model...")
    for repeat in range(3):
        caltab = eh.self_cal.self_cal(obs.flag_uvdist(uv_max=2e9), gausspriorLMT, sites=['LM','LM'], method='vis', ttype=ttype, processes=processes, caltable=True, gain_tol=1.0)
        obs = caltab.applycal(obs, interp='nearest', extrapolate=True)

    # Make an image -- with visibility amplitudes
    print("Imaging...")
    reg_term['flux']=alpha_flux
    imgr = eh.imager.Imager(obs, init, prior_im=gaussprior, data_term={'amp':alpha_vis, 'cphase':alpha_cphase, 'logcamp':alpha_logcamp}, 
                            maxit=maxit, clipfloor=-1., norm_reg=True, systematic_noise=systematic_noise, reg_term=reg_term, ttype=ttype, transform=transform, stop=stop)
    imgr = eh.imager.Imager(obs, init, prior_im=gaussprior, data_term={'amp':alpha_vis, 'cphase':alpha_cphase, 'logcamp':alpha_logcamp}, 
                            maxit=maxit, norm_reg=True, systematic_noise=systematic_noise, reg_term=reg_term, ttype=ttype, transform=transform, stop=stop)

    imgr.make_image_I(show_updates=False)
    im0 = imgr.out_last()
    im0.display()
    converge()

    # Self calibrate to the previous model (phase-only)
    im = imgr.out_last()
    im.display()
    '''
    obs_sc = eh.self_cal.self_cal(obs,    im, method='phase', ttype=ttype, processes=processes)
    obs_sc = eh.self_cal.self_cal(obs_sc, im, method='phase', ttype=ttype, processes=processes)

    # Make an image -- now with complex visibilities
    reg_term['flux']=10*alpha_flux
    imgr = eh.imager.Imager(obs_sc, init, prior_im=gaussprior, data_term={'vis':alpha_vis, 'cphase':alpha_cphase, 'logcamp':alpha_logcamp}, 
                            maxit=maxit, clipfloor=-1., norm_reg=True, systematic_noise=systematic_noise, reg_term = reg_term, ttype=ttype,  transform=transform, stop=stop)
    imgr.make_image_I(show_updates=False)
    converge()

    # Self calibrate to the previous model (amplitude and phase)
    im = imgr.out_last()
    obs_sc = eh.self_cal.self_cal(obs_sc, im, method='both', ttype=ttype, processes=processes)
    obs_sc = eh.self_cal.self_cal(obs_sc, im, method='both', ttype=ttype, processes=processes)

    # Make an image -- now with complex visibilities; common systematic noise
    reg_term['flux']=10*alpha_flux
    imgr = eh.imager.Imager(obs_sc, init, prior_im=gaussprior, data_term={'vis':alpha_vis, 'cphase':alpha_cphase, 'logcamp':alpha_logcamp}, 
                            maxit=maxit, clipfloor=-1., norm_reg=True, systematic_noise=0.05, reg_term = reg_term, ttype=ttype,  transform=transform, stop=stop)
    imgr.make_image_I(show_updates=False)
    converge()

    # Final image
    im_out = imgr.out_last()
    im_out.display()

    # Save out statistics and info
    reg_dict_out = {'flux':1,'cm':1,'l1':1,'simple':1,'tv':1,'tv2':1}
    chisq_dict_out = {'vis':1, 'amp':1,'cphase':1, 'logcamp':1, 'camp':1, 'bs':1}
    imgr.dat_term_next = chisq_dict_out
    imgr.reg_term_next = reg_dict_out
    imgr._change_imgr_params = True
    imgr.init_imager_I()
    chisqdict_final = imgr.make_chisq_dict(im_out.imvec)
    regdict_final = imgr.make_reg_dict(im_out.imvec)
    objfunc_final = imgr.tmpout.fun
    success_final = imgr.tmpout.success
    msg_final = imgr.tmpout.message
    #tstop = time.time()
    '''


def image_simplest(obs,zbl=1.,fovuas=160.,npix=128.):

    # Fixed parameters
    fov        = fovuas*eh.RADPERUAS  # field of view of the reconstructed image
    alpha_vis = 10                    # weight on visibility amplitude / complex visibility
    alpha_cphase = 100                # weight on closure phase chi^2 
    alpha_logcamp = 200               # weight on log closure amplitudes chi^2
    
    gaussprior = eh.image.make_square(obs, npix, fov).add_gauss(zbl, (0.5*fov, 0.5*fov, 0, 0, 0))
    imgr = eh.imager.Imager(obs, gaussprior, prior_im=gaussprior, data_term={'amp':alpha_vis, 'cphase':alpha_cphase, 'logcamp':alpha_logcamp}, 
                            maxit=1000, norm_reg=True, reg_term={'simple':10, 'tv2':10})

    imgr.make_image_I(show_updates=False)
    im0 = imgr.out_last()
    im0.display()
    return im0
