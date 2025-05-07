#!/usr/bin/env python
# coding: utf-8

#%%
import os
import sys
import copy
import pickle
import itertools
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from cartopy import crs as ccrs, feature as cfeature

# major pmm imports
sys.path.append('../')
from rotation import *

# csi for calc semivariogram
#import csi.insar as insar
#import csi.imagecovariance as imagecovariance

plt.rcParams.update({'font.size'  : 12,
                     'font.family':'Helvetica',
                     'figure.dpi' : 100,
                    })


def plot_pole_motion(pole, out_dir=False, show=False):
    """plot pole location, covariance, and predicted motions
    Input:
        pole
    global inputs:
        other poles for comparison...
    """
    pole_globe = False
    poles  = [pole, poleA, poleA20, poleA_GSRM, poleA_MORVEL, poleA_viltres]

    vector_frame = ['NNR','Nubia']

    for ax4_fr in vector_frame:
        # make two plots with different vectors in axis 4

        #------ allocate fig, axis -------
        # make figure
        fig      = plt.figure(figsize=(8,8))
        gspec    = fig.add_gridspec(2,2, wspace=0.02, hspace=0.02)
        if pole_globe:
            _lat0, _lon0 = find_extent(poles)[:2]
            ax1 = fig.add_subplot(gspec[0,0], projection=ccrs.NearsidePerspective(_lat0, _lon0, 5e6), aspect='auto')
            _pmap_ext, _sharex, _sharey = None, None, None
        else:
            ax1 = fig.add_subplot(gspec[0,0], projection=ccrs.PlateCarree(), aspect='auto')
            _pmap_ext, _sharex, _sharey = 'auto', ax1, ax1

        ax2 = fig.add_subplot(gspec[0,1], sharey=_sharey)
        ax3 = fig.add_subplot(gspec[1,0], sharex=_sharex)
        ax4 = fig.add_subplot(gspec[1,1], projection=ccrs.PlateCarree(), aspect='auto')

        #------ call the plot -------
        # axis 1 to 3: pole covariace
        names  = ['InSAR', 'ITRF2014', 'ITRF2020', 'GSRM_v2.1', 'MORVEL56', 'Viltres+']
        colors = ['b','r','orange','g','purple','pink']
        plot_pole_covariance(poles, names=names, colors=colors, n_std=2, extent=_pmap_ext, radius=None, lw=2, alpha=0.2, axes=[ax1,ax2,ax3], grids_on=False)
        ax1.legend().remove()
        ax2.patch.set_alpha(0.5)
        ax3.patch.set_alpha(0.5)
        ax2.set_facecolor('gainsboro')
        ax3.set_facecolor('gainsboro')
        ax3.legend(loc='upper left', fontsize=10)

        # axis 4: predicted plate motion
        if ax4_fr == 'NNR':
            epole_obj = [poleA, pole, pole-poleA]
        else:
            epole_obj = [poleA-poleB, pole-poleB, (pole-poleB)-(poleA-poleB)]

        qcolor    = ['r', 'b', 'k']
        qalpha    = [0.2, 0.2, 1.0]
        qname     = ['Altamimi+ (ITRF14)', 'InSAR (ITRF14)', 'Model discrep.']

        bndPoly = read_plate_outline('MORVEL', flag.plateName)
        plot_plate_motion(  plate_boundary   = bndPoly,
                            epole_obj        = epole_obj,
                            satellite_height = 5*1e5,
                            map_style        = 'platecarree',
                            helmert          = None,
                            orb              = False,
                            extent           = [33,43,25,34],
                            qcolor           = qcolor,
                            qalpha           = qalpha,
                            qname            = qname,
                            qunit=5, qscale=80, qwidth=.005,
                            pts_ms=5, qnum=10, unit='mm', font_size=8,
                            c_ocean='#D4F1F4', ax=ax4, grid_lc='none'
                            )
        ax4.set_title(f'({ax4_fr} fixed)', x=1, y=0.9, loc='right', fontsize=12)
        [ll.set_linewidth(1.2) for ll in ax1.spines.values()]
        [ll.set_linewidth(1.2) for ll in ax2.spines.values()]
        [ll.set_linewidth(1.2) for ll in ax3.spines.values()]
        [ll.set_linewidth(1.2) for ll in ax4.spines.values()]
        ax4.set_rasterization_zorder(0)
        if out_dir:
            fig.savefig(out_dir/f'{flag.projName}_poles_ax4in{ax4_fr}.pdf', dpi=200, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
    return


######################################
#          ( MODIFY BELOW )

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
    'featName'  : ['out','fixDC_test'],  # output path strings
    # i/o paths
    'data_dir'  : f"/marmot-nobak/ykliu/aqaba/topsStack/mosaic/itrf14/data",
    'roi_dir'   : f"/marmot-nobak/ykliu/aqaba/topsStack/mosaic/rotation_roi/strict", # basename: {full, relax, strict}
    'resfile'   : f'all_tracks_resid_out.npy',
    # gnss paths
    'gnssfile'  : f'/marmot-nobak/ykliu/aqaba/topsStack/invert_pole/viltres/viltres_2022_TableS1.txt',
    'polyfile'  : f'/marmot-nobak/ykliu/aqaba/topsStack/invert_pole/viltres/plot-data_arab_internal_polygon.csv',
    # specs
    'pmmData'   : (False,False,False),   # use PMM as data input / add previous post-fit residual as artificial noise / use absolute PMM (no ref_point)
    'demean'    : False              ,   # remove {mean,median} from data (mimic applying a ref_point when PMM as input)
    'avgStd'    : False              ,   #    use realizations ref_points avg std as Cd
    'conStd'    : False              ,   #    use {mean,median} std as a constant averaged Cd
    'priorDC'   : 'all'              ,   # the known DC shift from PMM, ex: False, 'all', [yourtracks]
    'biases'    : 'los'              ,   # components in the bias ['enu', 'los', False]
    'initMdl'   : False              ,   # a-priori initial model for iterations (e.g., ITRF2014)
    'lsop'      : 1                  ,   # first inversion option: 0=OLS, 1=Cd, 2=Cp, 3=Cd+Cp
    'diagCp'    : True               ,   # True=use only diagonals; False=use full Cp
    'aug_GPS'   : False              ,   # choose from: False, ['*TB01','AULA'], 'insar_area', 'plate_area'
    'run_Cp'    : False              ,   # whether to FURTHER RUN inversion with Cp (variogram)
    'run_IRLS'  : 20                 ,   # whether to FURTHER RUN IRLS with residuals, #iteration (0: equivalent to False)
}
# convert the messy flag dictionary to an object
flag = flag2obj(flag, dataDict)


## Published Euler poles
# primary plate A
poleA = EulerPole(name=flag.plateName, itrf='2014')
poleA.get_uncertainty(src='tableStd')
poleA20 = EulerPole(name=flag.plateName, itrf='2020')
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
_pmm = GSRM_NNR_V2_1_PMM[flag.plateName]
poleA_GSRM = EulerPole(name='GSRM-NNR', pole_lat=_pmm.Lat, pole_lon=_pmm.Lon, rot_rate=_pmm.omega, unit='deg/Ma')
cov = ut.make_symm_mat(3.36e+00, 4.02e+00, 1.01e+01, 4.91e+00, 3.05e+00, 1.95e+00) * 1e-20 # rad^2/yr^2
poleA_GSRM.get_uncertainty(in_err={'xyz_cov': cov}, src='in_err')

# primary plate A from MORVEL56
_pmm = NNR_MORVEL56_PMM[flag.plateName]
poleA_MORVEL = EulerPole(name='MORVEL56-NNR', pole_lat=_pmm.Lat, pole_lon=_pmm.Lon, rot_rate=_pmm.omega, unit='deg/Ma')
cov = ut.make_symm_mat(208, 182, 112, 398, 104, 87) * 1e-10 * 1e-12 # rad^2/yr^2
poleA_MORVEL.get_uncertainty(in_err={'xyz_cov': cov}, src='in_err')


## Published GPS
gpsGDf = read_GNSS_files(infile=flag.gnssfile, platename=flag.plateName, polyfile=flag.polyfile)

#          ( MODIFY ABOVE )
######################################


# read
for k in [*dataDict]:
    downscale = 5

    # --- some different cases ----
    if flag.pmmData[0] and flag.pmmData[2]:
        velfile = Path(flag.data_dir) / f'pmm_arab_abs/pmm_{k}.h5'  # data_in: ITRF pmm pred insar velo (absolute)
    elif flag.pmmData[0] and not flag.pmmData[2]:
        velfile = Path(flag.data_dir) / f'pmm_arab/pmm_{k}.h5'      # data_in: ITRF pmm pred insar velo (referenced)
    else:
        velfile = Path(flag.data_dir) / f'vel_{k}_msk.h5' # data_in: real data
    if flag.avgStd:
        stdfile = Path(flag.data_dir) / f'avgVel_{k}.h5'  # std_in : averaged out the ref_point
    else:
        stdfile = Path(flag.data_dir) / f'vel_{k}_msk.h5' # std_in : actual insar vstd (a ref_point)

    # ------ general readings ------
    geofile = Path(flag.data_dir) / f'geo_{k}.h5'
    roifile = Path(flag.roi_dir)  / f'roi_{flag.projName}_interior_{k}.h5'
    inps = read_data_geom(velfile, stdfile, geofile, roifile=roifile, downscale=downscale, flag_demean=flag.demean, flag_conStd=flag.conStd)
    vlos, std, inc_angle, azi_angle, lats, lons, roi, ref_los_vec, refyx, reflalo = inps

    # ------------ add residual as fake noise ----------------
    if flag.pmmData[0] and flag.pmmData[1]:
        # load residuals from file
        print('!! Add residual noise file(s) as noise contribution !!')
        in_resid = np.load(resfile, allow_pickle=True)
        vlos += in_resid[()][k]

    # ---------- get the std input scaling factor ------------
    std_scl = dataDict[k]['std_scale']

    # ------- build the dataDict: a tuple of data input ------
    dataDict[k] = (vlos, std, inc_angle, azi_angle, lats, lons, roi, ref_los_vec, refyx, reflalo, std_scl)



#%%
permutations = list(itertools.product([False,True], repeat=len(dataDict)))

if False:
    # simulate on/off of DC for each track
    for itr in range(len(permutations)):
        # add subfolder
        out_dir = Path(flag.out_dir) / f'permu_{itr}'
        out_dir.mkdir(parents=True, exist_ok=True)

        # tracks which removes DC
        flag.priorDC = list(np.array([*dataDict])[np.array(permutations[itr])])
        print(flag.priorDC)

        #########################################
        #       Step 1 : least-squares
        #########################################

        # Linear least-sqaures for an Euler pole
        blk = None

        # initialize
        blk = blockModel(name=flag.projName, print_msg=True)

        ## 1 - ADD DATA
        for j, (name, inps) in enumerate(dataDict.items()):
            vlos, vstd, los_inc_angle, los_azi_angle, lats, lons, roi, ref_los_vec, refyx, reflalo, std_scl  = inps
            vlos = np.array(vlos)

            # CHEATING here: remove the known DC shift from Altamimi PMM of poleA
            dc_pmm = -ref_los_vec @ poleA.get_velocity_enu(reflalo[0], reflalo[1])
            if name in flag.priorDC:
                print(f' *** remove DC {dc_pmm} from track {name} *** ')
                vlos -= dc_pmm

            # which dataset will need DC bias term
            blk.bias_dset = np.where([k not in flag.priorDC for k in [*dataDict]])[0]

            # add dataset to block obj
            blk.add_insar(vlos, lats, lons, vstd*std_scl, los_inc_angle, los_azi_angle, ref_los_vec, roi, name=name)


        ## 2 - BUILD G MATRICES
        # form the Gc part of G matrix: pole cross product with location = v_xyz
        blk.build_Gc()

        # form the T part of G matrix: transform v_xyz to v_enu
        blk.build_T()

        # form the L part of G matrix: project v_enu to InSAR LOS
        blk.build_L()

        # combine above parts to a whole G
        blk.build_G()

        # add a constant offset to account for reference point
        if flag.biases:
            blk.build_bias(fac=1e6, comp=flag.biases)

        blk.stack_data_operators()

        ## 3 - Augment with GPS stations?
        if flag.aug_GPS:
            if isinstance(flag.aug_GPS, (list, tuple)): # specific stations in a list
                blk = plugin_GNSS(blk, gpsGDf, flag.aug_GPS, uniErr=1e-4)
            else:
                blk = plugin_GNSS(blk, gpsGDf,  'all')  # all stations within the geopandas

        ## 4 - INVERT
        blk.invert(option=flag.lsop)

        if flag.initMdl:
            # use ITRF2014 Eurasia as my initial model, overwrites the inverted model
            # (See if Makran deviates away from Eura pole)
            print('*** a-priori initial model overwrites the current model params ***')
            poleA = EulerPole(name=flag.plateName, itrf='2014')
            poleA.get_uncertainty(src='tableStd')
            blk = blk.copy_from_pole(poleA)

        blk.get_model_pred(model_out=out_dir/'block_out.txt')
        blk.get_residRMS()
        print(f'#> 0 iteration with Cd only; condition number = {blk.cond}')

        # plot G
        ax = blk.plot_G(quantile=95)
        plt.savefig(out_dir/'G.png', dpi=200, transparent=True, bbox_inches='tight')

        # create a InSAR_based pole object
        pole = blk.create_pole(name=flag.plateName+'_InSAR')
        pole.get_uncertainty(block=blk, src='block')
        pole.print_info(outfile=out_dir/f'{flag.projName}_insar.ep')

        # plot the location
        #plot_pole_motion(pole, out_dir=out_dir, show=False)

        # Show the DC shifts estimate
        dc_diff, deg_p, dist_p = compare_DC_Shift(blk, dataDict, poleA, savefig=flag.projName, out_dir=out_dir)

        # Saving the objects:
        with open(out_dir/'dump.pkl', 'wb') as f:
            pickle.dump([blk, dataDict], f)


#%% Load the output pickles
if True:
    results = []
    for itr in range(len(permutations)):
        # get subfolder
        out_dir = Path(flag.out_dir) / f'permu_{itr}'

        # Getting back the objects:
        with open(out_dir/'dump.pkl', 'rb') as f:
            blk, dataDict = pickle.load(f)

        # make pole
        pole = blk.create_pole(name=flag.plateName+'_InSAR')
        pole.get_uncertainty(block=blk, src='block')

        # plot the location
        plot_pole_motion(pole, out_dir=out_dir, show=False)

        # Show the DC shifts estimate
        dc_diff, deg_p, dist_p = compare_DC_Shift(blk, dataDict, poleA)

        # append result
        results.append([flag.priorDC, blk.bias_dset, pole, dc_diff, deg_p, dist_p])


# %%
# Visualize the overall results

bias_dset = []
nbiasdset = []
dc_diffs = np.full((len(dataDict), len(results)), np.nan)
deg_ps   = np.full(len(results), np.nan)
dist_ps  = np.full(len(results), np.nan)
for i, res in enumerate(results):
    bias_dset.append(res[1])
    nbiasdset.append(len(res[1]))
    dc_diffs[:,i] = res[3]
    deg_ps[i]     = res[4]
    dist_ps[i]    = res[5]

x, y = np.meshgrid(np.arange(len(results)), np.arange(len(dataDict)))
x = x.flatten()
y = y.flatten()
c = dc_diffs.flatten()

gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [30, 1]}
fig, axs = plt.subplots(figsize=[8,4], nrows=2, ncols=2, gridspec_kw=gridspec_kw, sharex='col')
ax1 = axs[0,0]
ax2 = axs[1,0]
axc = axs[0,1]
axs[1,1].axis('off')
im = ax1.imshow(dc_diffs, cmap='coolwarm', aspect='auto', vmin=-2, vmax=2)
#im = ax1.scatter(x, y, c=c, s=20, cmap='seismic', vmin=-2, vmax=2)
plt.colorbar(im, cax=axc, label='DC deviation [mm/yr]')
ax1.set_xlim(np.min(x), np.max(x))
ax1.set_yticks(np.arange(len(dataDict)))
ax1.set_yticklabels([*dataDict])
ax1.set_ylabel('# Track')

sigma = np.sqrt(1.0835**2 + 0.5534**2) # 1_sigma of ITRF14 error

ax2.plot(np.arange(len(results)), deg_ps, c='k')
ax2.axhline(y=2*sigma, ls='--', c='grey', label=r'2$\sigma$ of ITRF PMM')
ax2.scatter(np.arange(len(results))[np.array(nbiasdset)==1], deg_ps[np.array(nbiasdset)==1], marker='o', ec='b', fc='none', s=40, lw=2, label='est. 1 track')
ax2.scatter(np.arange(len(results))[np.array(nbiasdset)==6], deg_ps[np.array(nbiasdset)==6], marker='o', ec='r', fc='none', s=40, lw=2, label='est. N-1 track')
ax2.set_ylabel('Pole deviation\nfrom ITRF14 [deg]')
ax2.set_ylim(0,9)
ax2.set_xticks([])
ax2.legend(fontsize=8)
ax2.set_xlabel('Permutations of fixing DC')
plt.savefig(Path(flag.out_dir) / f'DC_metrics.pdf', dpi=200, transparent=True, bbox_inches='tight')
plt.close()


plt.figure(figsize=[5,4])
plt.scatter(nbiasdset, deg_ps, s=40, fc='lightgrey', ec='k')
plt.xlabel('Estimate DC on how many #tracks')
plt.ylabel('Pole deviation\nfrom ITRF14 [deg]')
plt.axhline(y=2*sigma, ls='--', c='grey', label=r'2$\sigma$ of ITRF PMM')
plt.legend(fontsize=10)
plt.savefig(Path(flag.out_dir) / f'DC_trend.pdf', dpi=200, transparent=True, bbox_inches='tight')
plt.close()

