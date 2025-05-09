#!/usr/bin/env python
# coding: utf-8

# In[]:
import os
import sys
import copy
import pickle
import numpy as np
from pathlib import Path

# major pmm imports
sys.path.append('../')
from rotation import *


## insar datasets, std scaling, original file path
DATADICT = {
            'a087' : {'std_scale':1, 'maindir':'hpc_isce_stack/mintpy_1' },
            'a014' : {'std_scale':1, 'maindir':'hpc_isce_stack/mintpy_1' },
            'a116' : {'std_scale':1, 'maindir':'hpc_isce_stack/mintpy_1' },
            'a028' : {'std_scale':1, 'maindir':'hpc_topsStack/mintpy'    },
            'a057' : {'std_scale':1, 'maindir':'hpc_topsStack/mintpy'    },
            'd021' : {'std_scale':1, 'maindir':'hpc_isce_stack/mintpy_1' },
            'd123' : {'std_scale':1, 'maindir':'hpc_isce_stack/mintpy_1' },
            'd006' : {'std_scale':1, 'maindir':'hpc_topsStack2/mintpy'   },
            'd108' : {'std_scale':1, 'maindir':'hpc_topsStack2/mintpy'   , 'ramp':'/marmot-nobak/ykliu/aqaba/topsStack/mosaic/itrf14/data/yemen/post-ionoCorrection/vel_d108_mskRamp.h5'},
            #'a014WA' : {'std_scale':1, 'maindir':'hpc_isce_stack/mintpy_1' },
            }

# paper revision process, ramp rate std [meter/yr/pixel]
rampRateErr = {'a087': (3.15318152410333e-07  , 5.096963377305148e-07),
               'a014': (2.447958495788524e-07 , 3.949107213228061e-07),
               'a116': (3.5157561528016907e-07, 5.303506598162267e-07),
               'd021': (2.960456619763153e-07 , 5.51412149021013e-07),
               'd123': (2.2305233428011914e-07, 4.153723924711564e-07),
               'd006': (1.180642928456881e-06 , 1.3414208805574935e-06),
               'd108': (9.342749468243502e-07 , 1.2271907587535428e-06),
               'a028': (1.739132432103894e-06 , 1.4972728776128665e-06),
               'a057': (7.659508728564257e-07 , 1.024977881963351e-06)}
DATADICT = {
            'a087' : {'sig_ramp':rampRateErr['a087'], 'maindir':'hpc_isce_stack/mintpy_1' },
            'a014' : {'sig_ramp':rampRateErr['a014'], 'maindir':'hpc_isce_stack/mintpy_1' },
            'a116' : {'sig_ramp':rampRateErr['a116'], 'maindir':'hpc_isce_stack/mintpy_1' },
            'a028' : {'sig_ramp':rampRateErr['a028'], 'maindir':'hpc_topsStack/mintpy'    },
            'a057' : {'sig_ramp':rampRateErr['a057'], 'maindir':'hpc_topsStack/mintpy'    },
            'd021' : {'sig_ramp':rampRateErr['d021'], 'maindir':'hpc_isce_stack/mintpy_1' },
            'd123' : {'sig_ramp':rampRateErr['d123'], 'maindir':'hpc_isce_stack/mintpy_1' },
            'd006' : {'sig_ramp':rampRateErr['d006'], 'maindir':'hpc_topsStack2/mintpy'   },
            'd108' : {'sig_ramp':rampRateErr['d108'], 'maindir':'hpc_topsStack2/mintpy'   , 'ramp':'/marmot-nobak/ykliu/aqaba/topsStack/mosaic/itrf14/data/yemen/post-ionoCorrection/vel_d108_mskRamp.h5'},
            }

FLAG = {
    # names
    'projName'  : 'arabia',            # choose ['sinai', 'arabia', ...]
    'plateName' : 'Arabia',            # plate name for reading PMM, boundary files, etc.
    'nickname'  : 'arab'  ,            # a short name
    'featName'  : ['out10'],             # output path strings

    # i/o paths
    'dtype'     : 'real'            ,  # type of dataset  'real'   : real data
                                    #                  'realMC' : real data with monte carlo sampling reference points
                                    #                  'pmmAbs' : absolute plate motion model
                                    #                  'pmm'    : referenced plate motion model
    'noise'     : False,               # add noise ['whitenoise', (mean, std)]
                                    #           ['refnoise'  , (mean, std)]
                                    #           ['ramp'      , a, b, c, ...]
                                    #           ['residual'  , 'dump_in.pkl']
                                    #           ['modeldiff' , 'dump_in.pkl']
    'data_dir'  : f"/marmot-nobak/ykliu/aqaba/topsStack/mosaic/itrf14/data",
    'roi_dir'   : f"/marmot-nobak/ykliu/aqaba/topsStack/mosaic/rotation_roi/arab/strict", # basename: {full, relax, strict}
    'res_ofile' : 'all_tracks_resid_out.npy',

    # gnss paths
    'gnssfile'  : f'/marmot-nobak/ykliu/aqaba/topsStack/invert_pole/viltres/viltres_2022_TableS1.txt',
    'polyfile'  : f'/marmot-nobak/ykliu/aqaba/topsStack/invert_pole/viltres/plot-data_arab_internal_polygon.csv',
    'gpslist1'  : f'/marmot-nobak/ykliu/aqaba/topsStack/invert_pole/viltres/insar_gps_oman_itrf2014.txt',
    'gpslist2'  : f'/marmot-nobak/ykliu/aqaba/topsStack/invert_pole/viltres/insar_gps_overlap.txt',
    'useGPS'    : False               ,   # choose from: False, True, ['TB01','AULA']
    'gpsDC'     : False               ,   # True to enable DC shifts est. directly from gps

    # insar data preparation
    'looks'      : 5                  ,   # downsample the data
    #'demean'    : False              ,   # remove {mean,median} from data (mimic applying a ref_point when PMM as input)
    #'conStd'    : False              ,   #    use {mean,median} std as a constant averaged Cd
    #'priorDC'   : False              ,   # use known DC shift from PMM, ex: False, 'all', ['your_track_string',]

    # model G
    'est_ramp'  : True               ,   # estimate an x-y plane
    'refG'      : True               ,   # reference the G matrix
    #'biases'    : False              ,   # components in the bias ['enu', 'los', False]
    #'initMdl'   : False              ,   # a-priori initial model for iterations (e.g., ITRF2014)

    'synthetic' : False             ,

    # error model
    'errname'   : 'Cx'             ,   # error model assumption 'Cdt' 'Cds' 'Cdts' 'Cx'
    'errform'   : 'full'             ,   # use of error form 'no' 'diag' 'full'
    #'diaglzCov' : False              ,   # {True,False} to diagonalize the coavarinace matrix
    #'loadLD'    : False              ,   # {True,False} to load pre-saved diagonalized coavarinace matrix (False will rerun diagonalization)
    #'saveLD'    : False              ,   # {True,False} to save & overwrite diagonalized coavarinace matrix
    'gpuno'     : 7                  ,   # gpu device number [0-7] for KAMB (valid when errform=full and diaglzCov=False)

    # Monte carlo realizations
    'MC_num'    : False               ,   # False or number of realizations
    'mcdatadir' : '/marmot-nobak/ykliu/aqaba/topsStack',

    # other (mute!)
    #'run_Cp'    : False              ,   # whether to FURTHER RUN inversion with Cp (variogram)
    #'run_IRLS'  : False              ,   # whether to FURTHER RUN IRLS with residuals, #iteration (0: equivalent to False)
}


# common inputs
projName  = FLAG['projName']
plateName = FLAG['plateName']
nickname  = FLAG['nickname']
gnssfile  = FLAG['gnssfile']
polyfile  = FLAG['polyfile']
gpslist1  = FLAG['gpslist1']
gpslist2  = FLAG['gpslist2']

## Published Euler poles
# primary plate A
poleA = EulerPole(name=plateName, itrf='2014')
poleA.get_uncertainty(src='tableStd')
poleA20 = EulerPole(name=plateName, itrf='2020')
poleA20.get_uncertainty(src='tableStd')

# nearby plate B
poleB = EulerPole(name='Nubia')
poleB.get_uncertainty(src='tableStd')
poleB20 = EulerPole(name='Nubia', itrf='2020')
poleB20.get_uncertainty(src='tableStd')

# nearby plate C from [Castro-Perdomo et al., 2022]
poleC = EulerPole(name='Sinai', pole_lat=54.7, pole_lon=-12.2, rot_rate=0.417, unit='deg/Ma')

# primary plate A from [Viltres e al., 2022]
poleA_viltres = EulerPole(name='Viltres', pole_lat=50.93, pole_lon=-6.09, rot_rate=0.524, unit='deg/Ma')
cov = ut.make_symm_mat(1.76, 1.63, 1.07, 1.59, 1.02, 0.72) * 1e-6 * 1e-12 * (np.pi/180)**2 # rad^2/yr^2
poleA_viltres.get_uncertainty(in_err={'xyz_cov': cov}, src='in_err')

# primary plate A from GSRM
_pmm = GSRM_NNR_V2_1_PMM[plateName]
poleA_GSRM = EulerPole(name='GSRM-NNR', pole_lat=_pmm.Lat, pole_lon=_pmm.Lon, rot_rate=_pmm.omega, unit='deg/Ma')
#cov = ut.make_symm_mat(3.36e+00, 4.02e+00, 1.01e+01, 4.91e+00, 3.05e+00, 1.95e+00) * 1e-20 # rad^2/yr^2
cov = ut.make_symm_mat(3.36e+00, 0, 0, 4.91e+00, 0, 1.95e+00) * 1e-20 # rad^2/yr^2
poleA_GSRM.get_uncertainty(in_err={'xyz_cov': cov}, src='in_err')

# primary plate A from MORVEL56
_pmm = NNR_MORVEL56_PMM[plateName]
poleA_MORVEL = EulerPole(name='MORVEL56-NNR', pole_lat=_pmm.Lat, pole_lon=_pmm.Lon, rot_rate=_pmm.omega, unit='deg/Ma')
cov = ut.make_symm_mat(208, 182, 112, 398, 104, 87) * 1e-10 * 1e-12 # rad^2/yr^2
poleA_MORVEL.get_uncertainty(in_err={'xyz_cov': cov}, src='in_err')


## Published GPS
columns = ['Station','Lon','Lat','Ve','Vn',f'Ve_{nickname}',f'Vn_{nickname}','Vesig','Vnsig']
GPSDF_FULL = read_GNSS_files(infile    = gnssfile,
                             columns   = columns,
                             platename = plateName,
                             source    = r'Viltres$^+$',
                             )
GPSDF_EW = read_GNSS_files(infile    = gnssfile,
                           columns   = columns,
                           platename = plateName,
                           txtfile   = gpslist1,
                           source    = r'Viltres$^+$',
                           )

GPSDF_NW = read_GNSS_files(infile    = gnssfile,
                           columns   = columns,
                           platename = plateName,
                           txtfile   = gpslist2,
                           source    = r'Viltres$^+$',
                           )

##################################################################################

def user_inputs(tracks='auto'):
    """
    Modify user inputs here
    """
    if tracks == 'auto':
        tracks = [*DATADICT]

    # input dataset(s)
    dataDict = {}
    for tk in tracks:
        dataDict[tk] = DATADICT[tk]

    # input flag object
    flag = flag2obj(FLAG, dataDict)

    return flag, dataDict


def run_main(dataDict, flag, poleDict, gpsGDf=None, ref_i=None, extname=None, color='#ffa600'):
    # file extension and inverted pole obj name
    if extname is None or extname=='':
        ext = ''
        pname = 'InSAR'
    else:
        ext = f'_{extname}'
        pname = extname

    flag     = copy.deepcopy(flag)
    dataDict = copy.deepcopy(dataDict)
    poleDict = copy.deepcopy(poleDict)

    # read insar data to dictionary
    for m in [*poleDict]:
        if m.startswith('Altamimi') or m.startswith('ITRF2014'):
            pole_ref = poleDict[m][0]

    dataDict = inputflags2read(dataDict, flag, pole_ref=pole_ref, ref_i=ref_i)
    out_dir  = Path(flag.out_dir)



    #########################################
    #       Step 0 : spatial covariance
    #########################################

    # calc the structural function first.
    covSaveFile = out_dir/f'Cchi{ext}.pkl'
    covSaveText = out_dir/f'Cchi{ext}.txt'
    overwrite   = False
    Cds_dict    = None

    if covSaveFile and covSaveFile.is_file():
        if overwrite:
            print(f'delete and re-calculate: {covSaveFile}')
            covSaveFile.unlink() # delete existing file
        else:
            # Load the list of arrays from the .pkl file
            print(f'read from existing Cds: {covSaveFile}')
            with open(covSaveFile, 'rb') as f:
                Cds_dict = pickle.load(f)

    if flag.errname in ['Cds', 'Cdts', 'Cx'] and Cds_dict is None:
        # variogram
        insarDeramp, Cds_dict = calc_insar_variogram(dataDict, out_dir=out_dir, distmax=300., rampEst=True, frac=0.1)

        # plot
        plot_all_variograms(dataDict, insarDeramp, out_dir, savefig=out_dir/f'semivariograms.pdf')
        plot_all_variograms_image(dataDict, insarDeramp, out_dir, savefig=out_dir/f'deramp.png')

        # save Cd, Cp into file
        report_avg_covariance(dataDict, Cds_dict, 20, covSaveText, covSaveFile)


    # plot input data
    plot_datain(dataDict, out_dir/f'datain{ext}.pdf')

    # set gps DC as my prior DC
    if flag.gpsDC:
        gpsLOS, gpsDC = plot_gps_insar_offset(gpsGDf, dataDict, radius=5, outdir=flag.out_dir)
        flag.priorDC = gpsDC


    #########################################
    #       Step 1 : least-squares
    #########################################

    blk = run_build_block(dataDict, flag, poleA, gpsGDf, Cds_dict=Cds_dict, extname=ext)

    # save G
    with open(out_dir/f'G{ext}.pkl', 'wb') as f:
        pickle.dump(blk.G_all, f, protocol=pickle.HIGHEST_PROTOCOL)

    # add Cp
    if flag.errname == 'Cx':
        Cp_file = out_dir/f'Cp.pkl'

        if Cp_file.exists():
            # Load the list of arrays to a .pkl file
            print(f'read from existing Cp: {Cp_file}')
            with open(Cp_file, 'rb') as f:
                blk.Cp_set = pickle.load(f)

        else:
            # m0 = np.array([[ 5.74061937e-09],[-6.77681770e-12],[ 7.58764100e-09]])  # 7 tracks result
            # m0 = np.array([[ 6.00759390e-09],[-1.16945673e-10],[ 7.66076665e-09]])  # 9 track result

            #blk.insert_Cps(mc_dir='out7_realMC_strict_refG_Cdts_full', m0=None, subset=[0,500], savefile=Cp_file)
            blk.insert_Cps(mc_dir='out8_realMC_strict_refG_Cdts_full', m0=None, subset=[0,1000], savefile=Cp_file)
            #blk.compute_Cp(mc_dir='out9_realMC_strict_refG_Cdts_full', m0=None, subset=[0,200], savefile=Cp_file)

    blk = run_inversion(blk, flag, extname=ext, plotCov=out_dir/f'Cov.pdf')



    #########################################
    #       Step 2 : plots
    #########################################
    plot_datapdf(blk, dataDict, out_dir/f'dataHist.pdf')


    pole = postfit_outputs(blk, dataDict, flag, pole_ref=poleA, extname=ext, dump=False)

    poleDict[pname] = [pole, color]

    wrt = ['ITRF2014'] + [k.lstrip('*') for k in poleDict.keys() if k.startswith('*')]

    plot_quadriptych(poleDict,
                     frames    = wrt,
                     gpsGDf    = gpsGDf,
                     compare   = [pname, r'Altamimi$^+$'],
                     savefig   = out_dir/f'quadriptych{ext}.pdf',
                     rate_lim  = [.45,.61],
                     pmap_ext  = [-19.5,2,39.5,62],
                     plateName = flag.plateName,
                     cov_n_std = [2,1],
                     qunit     = 10,
                     qscale    = 0.12,
                     )


    plot_geodesic_profile(pole, poleA, blk, dataDict, gpsGDf, GPSDF_FULL,
                          extent = [15,30,35,60],
                          fname  = Path(flag.out_dir)/f'geodesic_profile{ext}.pdf',
                          has_GPS = flag.useGPS,
                          ad_nominal_pred=True)

    #######################
    # clear mem
    #######################
    del flag, dataDict, blk #, Cds_dict

    return


##################################################################################

if __name__ == "__main__":


    # pre-defined poles
    poleDict = {
         #'InSAR'         : [pole_centroid, '#ffa600'],
        r'Altamimi$^+$'  : [poleA        , '#7a0067'],
        r'Viltres$^+$'   : [poleA_viltres, '#1f6f6f'],
         'GSRM'          : [poleA_GSRM   , '#f45b3a'],
         'MORVEL56'      : [poleA_MORVEL , '#0a105c'],
         '*Nubia'        : [poleB        ,    'gray'],
    }

    # InSAR
    if True:
        #############################
        # no gps
        flag, dataDict = user_inputs([*DATADICT])
        run_main(dataDict, flag, poleDict, extname='9sar')  # pure-insar, out10, with ramp_rate_err and MC1000
        #run_main(dataDict, flag, poleDict, GPSDF_EW, extname='9sar')

        # #############################
        # # no gps, NW insar
        # tracks=['a087','a014','a116','d021','d123']
        # flag, dataDict = user_inputs(tracks)
        # run_main(dataDict, flag, poleDict, GPSDF_NW, extname='5sar')

        # #############################
        # # no gps, NW+Oman insar
        # tracks=['a087','a014','a116','a028','a057','d021','d123']
        # flag, dataDict = user_inputs(tracks)
        # run_main(dataDict, flag, poleDict, GPSDF_NW, extname='7sar')

    # InSAR + gps
    if False:
        # #############################
        # # NW-itrf-Oman gps, all insar
        # flag, dataDict = user_inputs([*DATADICT])
        # flag.useGPS = True
        # run_main(dataDict, flag, poleDict, GPSDF_EW, extname='9sar20gps')

        # #############################
        # # NW gps, NW insar
        # tracks=['a087','a014','a116','d021','d123']
        # flag, dataDict = user_inputs(tracks)
        # flag.useGPS = True
        # run_main(dataDict, flag, poleDict, GPSDF_NW, extname='5sar15gps')

        # #############################
        # # NW gps, NW+Yemen insar
        # tracks=['a087','a014','a116','d021','d123','d006','d108']
        # flag, dataDict = user_inputs(tracks)
        # flag.useGPS = True
        # run_main(dataDict, flag, poleDict, GPSDF_NW, extname='7sar15gps')

        #############################
        # NW gps, all insar
        flag, dataDict = user_inputs([*DATADICT])
        flag.useGPS = True
        run_main(dataDict, flag, poleDict, GPSDF_NW, extname='9sar15gps')

        # #############################
        # # NW gps, NW+Oman insar
        # tracks=['a087','a014','a116','a028','a057','d021','d123']
        # flag, dataDict = user_inputs([*DATADICT])
        # flag.useGPS = True
        # run_main(dataDict, flag, poleDict, GPSDF_NW, extname='7sar15gps')

        # #############################
        # # NW-itrf-Oman gps, NW insar
        # tracks=['a087','a014','a116','a028','a057','d021','d123']
        # flag, dataDict = user_inputs([*DATADICT])
        # flag.useGPS = True
        # run_main(dataDict, flag, poleDict, GPSDF_EW, extname='7sar20gps')
