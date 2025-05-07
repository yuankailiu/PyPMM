#!/usr/bin/env python
# coding: utf-8

# In[]:
import sys
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# pyproj geodesy
from pyproj import Geod
geod = Geod(ellps="WGS84")

# major pmm imports
sys.path.append('../')
from rotation import *
from pypmm.models import EARTH_RADIUS_A,EARTH_ECCENT
from pypmm.block import DC_from_block, DC_from_pole


def run_ls_pole(dataDict, flag):
    dataDict = dataDict.copy()
    flag = flag.copy()

    # flags to obj & mkdir
    flag    = flag2obj(flag, dataDict)
    out_dir = Path(flag.out_dir)

    # read insar data
    dataDict = inputflags2read(dataDict, flag)

    # plot input data
    plot_datain(dataDict, out_dir/f'datain.pdf')

    # set gps DC as my prior DC
    gpsGDf_los, gpsDC = plot_gps_insar_offset(gpsGDf, dataDict, radius=5, outdir=flag.out_dir)
    if 'GPSDC' in flag.featName:
        flag.priorDC = gpsDC

    # build G
    blk = run_build_block(dataDict, flag, poleA, gpsGDf)

    # inversion
    blk = run_inversion_Cd(blk, flag, inv_opt=flag.lsop)

    # data and resid pdf
    plot_datapdf(blk, dataDict, out_dir/f'dataHist.pdf')

    # update the block inverted pole
    pole = postfit_outputs(blk, dataDict, flag, pole_ref=poleA, extname=None)
    poleDict['InSAR'] = [pole, 'b']

    # plots
    if str(flag.dtype).startswith('real'):
        plot_pole_motion(poleDict, flag, frames=['ITRF2014','Nubia'], savefig=out_dir/f'poleVector.pdf')
    elif str(flag.dtype).startswith('pmm'):
        plot_pole_motion(poleDict, flag, frames=['ITRF2014','Nubia'], savefig=out_dir/f'poleVector.pdf', qunit=0.5, qscale=8) # shorter arrows

    plot_globes(poleDict, flag, savefigs = [
                                out_dir/f'pmm_est.pdf',
                                out_dir/f'pmms0.pdf'  ,
                                out_dir/f'pmms.pdf'   ,
                                ])

    plot_geodesic_profile(pole, poleA, blk, dataDict, gpsGDf, gpsGDf_full,
                          fname  = Path(flag.outdir)/f'geodesic_profile{ext}.pdf',
                          has_GPS = flag.useGPS)

    return blk, pole


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
        'roi_dir'   : f"/marmot-nobak/ykliu/aqaba/topsStack/mosaic/rotation_roi/arab/strict", # basename: {full, relax, strict}
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
get_user_inputs()

# real data basic
flag['featName'] = ['out','seven','vanilla']
flag['dtype']    = 'real'
flag['noise']    = False
blk, pole = run_ls_pole(dataDict, flag)

# PMM pure (perfect!)
flag['featName'] = ['out','seven','vanilla']
flag['dtype']    = 'pmm'
flag['noise']    = False
blk, pole = run_ls_pole(dataDict, flag)


# real data + 16 GPS jointly, std=1
get_user_inputs()
dataDict = {
            'a087' : {'std_scale' : 1},
            'a014' : {'std_scale' : 1},
            'a116' : {'std_scale' : 1},
            'a028' : {'std_scale' : 1},
            'a057' : {'std_scale' : 1},
            'd021' : {'std_scale' : 1},
            'd123' : {'std_scale' : 1},
            }
flag['featName'] = ['out','seven','vanilla']
flag['dtype']    = 'real'
flag['aug_GPS']  = True
blk, pole = run_ls_pole(dataDict, flag)


# real data + 16 GPS jointly, std=10
get_user_inputs()
dataDict = {
            'a087' : {'std_scale' : 10},
            'a014' : {'std_scale' : 10},
            'a116' : {'std_scale' : 10},
            'a028' : {'std_scale' : 10},
            'a057' : {'std_scale' : 10},
            'd021' : {'std_scale' : 10},
            'd123' : {'std_scale' : 10},
            }
flag['featName'] = ['out','seven','vanilla','std10']
flag['dtype']    = 'real'
flag['aug_GPS']  = True
blk, pole = run_ls_pole(dataDict, flag)


# real data + GPS_DC + 16 GPS jointly, std=1
get_user_inputs()
dataDict = {
            'a087' : {'std_scale' : 1},
            'a014' : {'std_scale' : 1},
            'a116' : {'std_scale' : 1},
            'a028' : {'std_scale' : 1},
            'a057' : {'std_scale' : 1},
            'd021' : {'std_scale' : 1},
            'd123' : {'std_scale' : 1},
            }
flag['featName'] = ['out','seven','vanilla','GPSDC']
flag['dtype']    = 'real'
flag['aug_GPS']  = True
blk, pole = run_ls_pole(dataDict, flag)


# real data + GPS_DC + 16 GPS jointly, std=10
get_user_inputs()
dataDict = {
            'a087' : {'std_scale' : 10},
            'a014' : {'std_scale' : 10},
            'a116' : {'std_scale' : 10},
            'a028' : {'std_scale' : 10},
            'a057' : {'std_scale' : 10},
            'd021' : {'std_scale' : 10},
            'd123' : {'std_scale' : 10},
            }
flag['featName'] = ['out','seven','vanilla','GPSDC','std10']
flag['dtype']    = 'real'
flag['aug_GPS']  = True
blk, pole = run_ls_pole(dataDict, flag)


#%%
if False:
    # ----------- some artifitial noise tests ---------------
    # PMM + ref noise (fine too)
    flag['featName'] = ['noise','seven']
    flag['dtype']    = 'pmm'
    flag['noise']    = ['refnoise', (0, 0.01)]
    blk, pole = run_ls_pole(dataDict, flag)

    # PMM + white noise (fine too)
    flag['featName'] = ['noise','seven']
    flag['dtype']    = 'pmm'
    flag['noise']    = ['whitenoise', (0, 0.005)]
    blk, pole = run_ls_pole(dataDict, flag)

    # PMM + high-freq residual noise (fine too)
    flag['featName'] = ['noise','seven']
    flag['dtype']    = 'pmm'
    flag['noise']    = ['residual', 'dump_in.pkl']
    blk, pole = run_ls_pole(dataDict, flag)

    # PMM + long-wavelength noise (to bias the pole, and it is biased)
    flag['featName'] = ['noise','seven']
    flag['dtype']    = 'pmm'
    flag['noise']    = ['modeldiff', 'dump_in.pkl']
    blk, pole = run_ls_pole(dataDict, flag)

    # real data + subtract out the long-wavelength noise (now the pole agrees with itrf14)
    flag['featName'] = ['noise','seven']
    flag['dtype']    = 'real'
    flag['noise']    = ['modeldiff', ['dump_in.pkl', -1]]
    blk, pole = run_ls_pole(dataDict, flag)
    # ----------- some artifitial noise tests ---------------
