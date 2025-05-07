#!/usr/bin/env python
# coding: utf-8

# In[]:
import sys
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# major pmm imports
sys.path.append('../')
from rotation import *
from pypmm.models import EARTH_RADIUS_A,EARTH_ECCENT
from pypmm.block import DC_from_block, DC_from_pole


def get_user_inputs():
    ######################################
    #          ( MODIFY BELOW )

    np.random.seed(40)

    ## asc/dsc datasets and std scaling
    dataDict = {
                'a087' : {'std_scale' : 1},
                'a014' : {'std_scale' : 1},
                'a116' : {'std_scale' : 1},
                'a028' : {'std_scale' : 1},
                'a057' : {'std_scale' : 1},
                'd021' : {'std_scale' : 1},
                'd123' : {'std_scale' : 1},
                }

    ## input flags
    flag = {
        # names
        'projName'  : 'arabia',            # choose ['sinai', 'arabia', ...]
        'plateName' : 'Arabia',            # plate name for reading PMM, boundary files, etc.
        'nickname'  : 'arab'  ,            # a short name
        'featName'  : [],                  # output path strings
        # i/o paths
        'dtype'     : 'real'            ,    # type of dataset  'real'   : real data
                                            #                  'realMC' : real data with monte carlo sampling reference points
                                            #                  'pmmAbs' : absolute plate motion model
                                            #                  'pmm'    : referenced plate motion model
        'noise'     : False,               # add noise ['whitenoise', (mean, std)]
                                        #           ['refnoise'  , (mean, std)]
                                        #           ['ramp'      , a, b, c, ...]
                                        #           ['residual'  , 'dump_in.pkl']
                                        #           ['modeldiff' , 'dump_in.pkl']
        'data_dir'  : f"/marmot-nobak/ykliu/aqaba/topsStack/mosaic/itrf14/data",
        'roi_dir'   : f"/marmot-nobak/ykliu/aqaba/topsStack/mosaic/rotation_roi/strict", # basename: {full, relax, strict}
        'res_ofile' : 'all_tracks_resid_out.npy',
        # gnss paths
        'gnssfile'  : f'/marmot-nobak/ykliu/aqaba/topsStack/invert_pole/viltres/viltres_2022_TableS1.txt',
        'polyfile'  : f'/marmot-nobak/ykliu/aqaba/topsStack/invert_pole/viltres/plot-data_arab_internal_polygon.csv',
        'txtfile'   : f'/marmot-nobak/ykliu/aqaba/topsStack/invert_pole/viltres/insar_gps_overlap.txt',
        # specs
        'looks'     : 5                  ,   # downsample the data
        'demean'    : False              ,   # remove {mean,median} from data (mimic applying a ref_point when PMM as input)
        'conStd'    : False              ,   #    use {mean,median} std as a constant averaged Cd
        'priorDC'   : False              ,   # use known DC shift from PMM, ex: False, 'all', ['your_track_string',]
        'biases'    : 'los'              ,   # components in the bias ['enu', 'los', False]
        'initMdl'   : False              ,   # a-priori initial model for iterations (e.g., ITRF2014)
        'lsop'      : 1                  ,   # first inversion option: 0=OLS, 1=Cd, 2=Cp, 3=Cd+Cp
        'diagCp'    : True               ,   # True=use only diagonals; False=use full Cp
        'aug_GPS'   : False              ,   # choose from: False, ['*TB01','AULA']
        'run_Cp'    : False              ,   # whether to FURTHER RUN inversion with Cp (variogram)
        'run_IRLS'  : 10                 ,   # whether to FURTHER RUN IRLS with residuals, #iteration (0: equivalent to False)
    }

    plateName = flag['plateName']
    nickname  = flag['nickname']
    gnssfile  = flag['gnssfile']
    polyfile  = flag['polyfile']
    txtfile   = flag['txtfile']

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
    cov = ut.make_symm_mat(3.36e+00, 4.02e+00, 1.01e+01, 4.91e+00, 3.05e+00, 1.95e+00) * 1e-20 # rad^2/yr^2
    poleA_GSRM.get_uncertainty(in_err={'xyz_cov': cov}, src='in_err')

    # primary plate A from MORVEL56
    _pmm = NNR_MORVEL56_PMM[plateName]
    poleA_MORVEL = EulerPole(name='MORVEL56-NNR', pole_lat=_pmm.Lat, pole_lon=_pmm.Lon, rot_rate=_pmm.omega, unit='deg/Ma')
    cov = ut.make_symm_mat(208, 182, 112, 398, 104, 87) * 1e-10 * 1e-12 # rad^2/yr^2
    poleA_MORVEL.get_uncertainty(in_err={'xyz_cov': cov}, src='in_err')


    ## Published GPS
    columns = ['Station','Lon','Lat','Ve','Vn',f'Ve_{nickname}',f'Vn_{nickname}','Vesig','Vnsig']
    gpsGDf_full = read_GNSS_files(infile    = gnssfile,
                                  columns   = columns,
                                  platename = plateName,
                                  )
    gpsGDf = read_GNSS_files(   infile    = gnssfile,
                                columns   = columns,
                                platename = plateName,
                                polyfile  = polyfile,  # select only ones within insar datasets (NW arabia)
                                )
    gpsGDf = read_GNSS_files(   infile    = gnssfile,
                                columns   = columns,
                                platename = plateName,
                                txtfile   = txtfile,  # select those sites in the txtfile
                                )


    #          ( MODIFY ABOVE )
    ######################################

    # pre-defined poles
    poleDict = {
        'ITRF2014'  : [poleA        , 'r'     ] ,
        'ITRF2020'  : [poleA20      , 'orange'] ,
        'GSRM_v2.1' : [poleA_GSRM   , 'g'     ] ,
        'MORVEL56'  : [poleA_MORVEL , 'purple'] ,
        'Viltres+'  : [poleA_viltres, 'pink'  ] ,
        '*Nubia'    : [poleB        , 'grey'  ] ,
        '*Nubia20'  : [poleB20      , 'grey'  ] ,
    }

    globals().update(locals())
    return



#%%

if True:
    get_user_inputs()

    #infile = '/home/ykliu/marmot-nobak/aqaba/topsStack/invert_pole/arabia/out_seven_vanilla_real_losBias_diagCp_strict/dump.pkl'
    #infile = '/home/ykliu/marmot-nobak/aqaba/topsStack/invert_pole/arabia/out_seven_vanilla_real_losBias_diagCp_strict_gps/dump.pkl'
    #infile = '/home/ykliu/marmot-nobak/aqaba/topsStack/invert_pole/arabia/out_seven_vanilla_real_modeldiff_losBias_diagCp_strict/dump.pkl'
    #infile = '/home/ykliu/marmot-nobak/aqaba/topsStack/invert_pole/arabia/out_seven_vanilla_GPSDC_real_losBias_diagCp_strict/dump.pkl'
    infile = '/home/ykliu/marmot-nobak/aqaba/topsStack/invert_pole/arabia/out_seven_vanilla_GPSDC_real_losBias_diagCp_strict_gps/dump.pkl'
    with open(infile, 'rb') as f:  # from dump.pkl file
        flag, blk, dataDict = pickle.load(f)
        pole = blk.create_pole()

# geodesic velocity profile
plot_geodesic_profile(pole, poleA, blk, dataDict,
                      gpsGDf,
                      fname  = Path(flag.outdir)/f'geodesic_profile{ext}.pdf',
                      has_GPS = flag.useGPS,
                    )


# %%
