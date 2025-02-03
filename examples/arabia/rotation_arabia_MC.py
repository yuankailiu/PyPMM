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
            'a087' : {'std_scale':1, 'maindir':'hpc_isce_stack/mintpy_1' , 'ramp':'/marmot-nobak/ykliu/aqaba/topsStack/mosaic/itrf14/data/new_diff/vel_a087_msk_diff_velocity.h5'},
            'a014' : {'std_scale':1, 'maindir':'hpc_isce_stack/mintpy_1' , 'ramp':'/marmot-nobak/ykliu/aqaba/topsStack/mosaic/itrf14/data/new_diff/vel_a014_msk_diff_velocity.h5'},
            'a116' : {'std_scale':1, 'maindir':'hpc_isce_stack/mintpy_1' , 'ramp':'/marmot-nobak/ykliu/aqaba/topsStack/mosaic/itrf14/data/new_diff/vel_a116_msk_diff_velocity.h5'},
            'a028' : {'std_scale':1, 'maindir':'hpc_topsStack/mintpy'    },
            'a057' : {'std_scale':1, 'maindir':'hpc_topsStack/mintpy'    },
            'd021' : {'std_scale':1, 'maindir':'hpc_isce_stack/mintpy_1' },
            'd123' : {'std_scale':1, 'maindir':'hpc_isce_stack/mintpy_1' },
            'd006' : {'std_scale':1, 'maindir':'hpc_topsStack2/mintpy'   },
            'd108' : {'std_scale':1, 'maindir':'hpc_topsStack2/mintpy'   , 'ramp':'/marmot-nobak/ykliu/aqaba/topsStack/mosaic/itrf14/data/yemen/post-ionoCorrection/vel_d108_mskRamp.h5'},
            }

FLAG = {
    # names
    'projName'  : 'arabia',            # choose ['sinai', 'arabia', ...]
    'plateName' : 'Arabia',            # plate name for reading PMM, boundary files, etc.
    'nickname'  : 'arab'  ,            # a short name
    'featName'  : ['out9'],            # output path strings

    # i/o paths
    'dtype'     : 'realMC'            ,  # type of dataset  'real'   : real data
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
    'refG'      : True               ,   # reference the G matrix
    #'biases'    : False              ,   # components in the bias ['enu', 'los', False]
    #'initMdl'   : False              ,   # a-priori initial model for iterations (e.g., ITRF2014)

    'synthetic' : False             ,
    #'synthetic' : { 'name'    : ['a014','d021','a057'] ,    # create synthetic data
    #                'comp'    : 'enu2los',
    #                'error'   : 0.001    ,
    #                'replace' : True    },

    # error model
    'errname'   : 'Cdts'             ,   # error model assumption 'Cdt' 'Cds' 'Cdts'
    'errform'   : 'full'             ,   # use of error form 'no' 'diag' 'full'
    #'diaglzCov' : False              ,   # {True,False} to diagonalize the coavarinace matrix
    #'loadLD'    : False              ,   # {True,False} to load pre-saved diagonalized coavarinace matrix (False will rerun diagonalization)
    #'saveLD'    : False              ,   # {True,False} to save & overwrite diagonalized coavarinace matrix
    'gpuno'     : 4                  ,   # gpu device number [0-7] for KAMB (valid when errform=full and diaglzCov=False)

    # Monte carlo realizations
    'MC_num'    : 1000               ,   # False or number of realizations
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


def run_main(dataDict, flag, poleDict, gpsGDf=None, ref_i=None, Cds_dict=None, extname=None, color='#ffa600'):
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



    if Cds_dict is None:
        #########################################
        #       Step 0 : spatial covariance
        #########################################
        # calc the structural function first.
        overwrite   = False
        covSaveFile = out_dir/f'Cchi{ext}.pkl'  # compute for each realization
        covSaveText = out_dir/f'Cchi{ext}.txt'

        # read or delete existing file
        if covSaveFile and covSaveFile.is_file():
            if overwrite:
                print(f'delete and re-calculate: {covSaveFile}')
                covSaveFile.unlink() # delete existing file
            else:
                # Load the list of arrays from the .pkl file
                print(f'read from existing: {covSaveFile}')
                with open(covSaveFile, 'rb') as f:
                    Cds_dict = pickle.load(f)

        # calculate Cds if needed
        if (Cds_dict is None) and (flag.errname in ['Cds', 'Cdts', 'Cx']):
            # variogram
            insarDeramp, Cds_dict = calc_insar_variogram(dataDict, out_dir=out_dir, distmax=300., rampEst=True, frac=1.0)

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

    # save G (compute theory error Cp, later use)
    with open(out_dir/'G.pkl', 'wb') as f:
        print(f"*** saving operator G, dim: \t{blk.G_all.shape}")
        pickle.dump(blk.G_all, f, protocol=pickle.HIGHEST_PROTOCOL)

    # with open(out_dir/'G.pkl', 'rb') as f:
    #     matrix = pickle.load(f)
    #     # Print the dimensions of the matrix
    #     if hasattr(matrix, 'shape'):
    #         print(f"******** {ext} G dim: \t{matrix.shape}")

    if True:
        blk = run_inversion(blk, flag, extname=ext, plotCov=out_dir/f'Cov.pdf')


        #########################################
        #       Step 2 : plots
        #########################################
        plot_datapdf(blk, dataDict, out_dir/f'dataHist.pdf')


        pole = postfit_outputs(blk, dataDict, flag, pole_ref=poleA, extname=ext, dump='vpred')

        poleDict[pname] = [pole, color]

        wrt = [k.lstrip('*') for k in poleDict.keys() if k.startswith('*')]

        if False:
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

    #covSaveFile = '/marmot-nobak/ykliu/aqaba/topsStack/invert_pole/arabia/out6_real_strict_refG_Cdts_full/Cchi_9sar_new2_median.pkl'
    #covSaveFile = '/marmot-nobak/ykliu/aqaba/topsStack/invert_pole/arabia/out8_realMC_strict_refG_Cdts_full/ref_0000/Cchi.pkl'
    #covSaveFile = '/marmot-nobak/ykliu/aqaba/topsStack/invert_pole/arabia/out9_realMC_strict_refG_Cdts_full/ref_0000/Cchi.pkl'
    covSaveFile = '/marmot-nobak/ykliu/aqaba/topsStack/invert_pole/arabia/out8_real_strict_refG_Cdts_full/Cchi_9sar.pkl'
    #covSaveFile = False

    if covSaveFile:
        # load pre-calculated spatial covariance from a prior realization
        # since we deramp, and assume stationarity and isotropic
        # so this is good. Also it uses all pixels to compute variogram
        print(f'Read from existing variogram: {covSaveFile}')
        with open(covSaveFile, 'rb') as f:
            Cds_dict = pickle.load(f)
    else:
        Cds_dict = None


    flag, dataDict = user_inputs()


    # Monte Carlo of different ref_points
    if flag.dtype=='realMC' and flag.MC_num:
        ref_arr = np.arange(int(flag.MC_num)+1)
    else:
        ref_arr = np.arange(1)

    # only test run these
    ref_arr = np.arange(795, 1001)


    # Run each reference point
    print(f'run realizations of {ref_arr}')
    for ref_i in ref_arr:
        run_main(dataDict, flag, poleDict, ref_i=ref_i, Cds_dict=Cds_dict)

# %%
