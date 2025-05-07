import os
import numpy as np
from pathlib import Path
import pickle
import copy

import pandas as pd
import geopandas as gpd
from shapely import Polygon, Point
from skimage.transform import downscale_local_mean

import matplotlib
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt

from cartopy import crs as ccrs, feature as cfeature

# pyproj geodesy
from pyproj import Geod

# still depends on MintPy
from mintpy.utils import readfile, utils as mut
from mintpy.stdproc.multilook import multilook_data_kai as multilook_data

# csi for calc semivariogram
import csi.insar as insar
import csi.imagecovariance as imagecovariance

# shift to pypmm now
from pypmm.block import blockModel, DC_from_block, DC_from_pole
from pypmm import utils as ut
from pypmm.models import (HELMERT_ITRF20_ITRF14,
                          GSRM_NNR_V2_1_PMM,
                          NNR_MORVEL56_PMM,
                          MAS2RAD, MASY2DMY,
                          EARTH_RADIUS_A, EARTH_ECCENT,
                          )

from pypmm.euler_pole import EulerPole
from pypmm.plot_utils import (read_plate_outline,
                              plot_plate_motion,
                              plot_imshow,
                              plot_pole_covariance,
                              draw_confidence_ellipse,
                              find_extent,
                              tweak_color,
                              text_accomodating_xylim,
                              update_handles,
                              tablelegend,
                              )

plt.rcParams.update({'font.size'  : 12,
                     'font.family':'Helvetica',
                     'figure.dpi' : 100,
                    })

########################

def read_faults_gmt(txtfile):
    """
    read points in lon lat format (GMT)
    """
    faults = []
    with open(txtfile) as f:
        lines = f.readlines()
        points = []
        for line in lines:
            if line.startswith('#'):
                continue
            elif line[0].isdigit():
                points.append(np.array(line.split(',')).astype(float))
            else:
                faults.append(np.array(points))
                points = []

    if len(faults)==0:
        faults = np.array(points)

    return faults


def gps_ref_pole(gpsDf, pole, sign='+',
                 columns=['Lat','Lon','Ve','Vn'],
                 unit='mm'):
    """
    output column name will be 'Ve', 'Vn'
    and will overwrite the existing column if names are the same
    """
    la, lo, ee, nn = columns

    v_enu = np.array(pole.get_velocity_enu(gpsDf[la], gpsDf[lo]))

    if unit == 'mm':
        uu = 1e3
    elif unit == 'cm':
        uu = 1e2
    elif unit == 'm':
        uu = 1.

    if sign == '+':
        c = 1
    elif sign == '-':
        c = -1

    gpsDf['Ve'] = gpsDf[ee] + c * uu * v_enu[0]
    gpsDf['Vn'] = gpsDf[nn] + c * uu * v_enu[1]

    return gpsDf

########################

def extract_numeric(string):
    return ''.join([char for char in string if char.isdigit()])


def read_pole_files(files, names=None, error='none'):
    """Read multiploe pole files

    Report the centroid pole by assuming error being
        'none'      :   just take the mean, convariance from the spread of the params
        'propagate' :   use formal error propagation with covariance matrices --> posterior
        'uniform'   :   assume cov matrix is I matrix
    """

    Nfiles = len(files)
    if names is None: names = [None] * len(files)

    # Initialize arrays and lists
    lats    = np.zeros(Nfiles)
    lons    = np.zeros(Nfiles)
    rates   = np.zeros(Nfiles)
    e2s     = np.zeros(Nfiles)
    n_samps = np.zeros(Nfiles)
    conds   = np.zeros(Nfiles)
    ms      = []
    Cms     = []

    # read data post-fit quality
    for i, blk_file in enumerate(files):
        with open(blk_file, 'r') as f:
            lines = f.readlines()
            for li, line in enumerate(lines):
                line = line.splitlines()[0].strip()
                if line.startswith('m    ='):
                    l0 = int(li + 1)
                if line.startswith('e2'):
                    l1 = int(li)
                    e2s[i] = float(line.split('=')[-1].strip().strip('[]'))
                if line.startswith('n_samps'):
                    n_samps[i] = float(line.split('=')[-1].strip())
                if line.startswith('cond'):
                    conds[i] = float(line.split('=')[-1].strip())
                if line.startswith('Cm    ='):
                    l2 = int(li + 1)
                if line.startswith('m_std ='):
                    l3 = int(li)

            # get model params
            _m = []
            for line in lines[l0:l1]:
                _m.append(float(line.strip().strip('[[]]').strip()))
            ms.append(_m)

            # get Cm
            _Cm = []
            for line in lines[l2:l3]:
                _Cm.append(np.array(line.strip().strip('[[]]').strip().split()).astype(float))
            Cms.append(_Cm)

    ms  = np.array(ms)
    Cms = np.array(Cms)


    # realizations of poles
    poles = []
    for i, (m, cov, name) in enumerate(zip(ms, Cms, names)):
        if name is None:
            name = f'real_{i}'
        pole = EulerPole(name=name, wx=m[0], wy=m[1],  wz=m[2], unit='rad') # rad/yr
        pole.get_uncertainty(in_err={'xyz_cov': cov}, src='in_err')   # rad^2/yr^2
        poles.append(pole)
        # store pole sph locations
        lats[i]  = pole.poleLat             # deg
        lons[i]  = pole.poleLon             # deg
        rates[i] = pole.rotRate * MASY2DMY  # deg/Ma


    # population centroid & covariance
    if error == 'none':
        centroid_lo   = np.mean(lons)
        centroid_la   = np.mean(lats)
        centroid_rate = np.mean(rates)
        pole_centroid = EulerPole(name='centroid', pole_lat=centroid_la, pole_lon=centroid_lo, rot_rate=centroid_rate, unit='deg/Ma')

        # use euler parameters (cartesian in rad/yr), compute covariance of the model spread
        post_Cm = np.cov(ms.T)  # rad^2/yr^2
        pole_centroid.get_uncertainty(in_err={'xyz_cov': post_Cm}, src='in_err')   # rad^2/yr^2

    elif error in ['propagate', 'uniform']:

        if error == 'uniform':
            for i in range(len(Cms)):
                if i % 10 == 0: print(Cms[i])
                Cms[i] = np.eye(len(Cms[i]))
                if i % 10 == 0: print(Cms[i])

        post_m, post_Cm, G = ut.multivariate_normal_centroid(ms, Cms)

        # these poles are all correlated (data are draw from the same dataset, only shifted)
        if error == 'propagate':
            print(f'Assume independent Linear system, (G.T G)^-1 = \n', np.linalg.pinv(G.T@G))
            print(f'Rescale the variance by #{len(ms)} samples, times {len(ms)}')
            post_Cm *= len(ms)

        pole_centroid = EulerPole(name='centroid', wx=post_m[0], wy=post_m[1], wz=post_m[2], unit='rad/yr')
        pole_centroid.get_uncertainty(in_err={'xyz_cov': post_Cm}, src='in_err')   # rad^2/yr^2


    return poles, lats, lons, rates, n_samps, e2s, ms, Cms, conds, pole_centroid



def plot_deviations(ms, Cms, conds, names, rms=None, ref_pole=None, outpic=None, figsize=[6,3.4], ylog=True):
    wx = ref_pole.wx * MAS2RAD  # rad/yr
    wy = ref_pole.wy * MAS2RAD  # rad/yr
    wz = ref_pole.wz * MAS2RAD  # rad/yr

    # 2-sigma of Altamimi in rad/yr
    thresh_2sigma = 2*np.linalg.norm(np.sqrt(np.diag(ref_pole.xyz_cov)))
    ys      = []    # value: rad/yr
    c_trace = []    # rad/yr **2
    c_det   = []    # rad/yr **3

    for i, (name, m, Cm) in enumerate(zip(names, ms, Cms)):
        ys.append(np.linalg.norm([m[0]-wx, m[1]-wy, m[2]-wz]))
        c_trace.append(np.trace(Cm))
        c_det.append(np.linalg.det(Cm))

    RADY2DMY = (180./np.pi)*1e6
    ys      = np.array(ys)
    c_trace = np.array(c_trace)
    c_det   = np.array(c_det)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize,
                            gridspec_kw={'width_ratios' : [figsize[0]*5,1],
                                         'height_ratios': [0.75, 1],
                                         'hspace'       : 0.07,
                                         'wspace'       : 0.03},
                            sharex='col')
    ax1, ax2, axc1, axc2 = axs[0,0], axs[1,0], axs[0,1], axs[1,1]

    # metric coloring ax1
    if c_det is not None:
        clog   = np.log10(c_det*(RADY2DMY**3))
        norm   = plt.Normalize(clog.min(), clog.max())
        cmap   = matplotlib.cm.get_cmap('YlGnBu')
        bar_cs = cmap(norm(clog))
    else:
        clog   = None
        bar_cs = 'lightgrey'
        axc1.axis('off')

    # metric coloring ax2
    if rms is not None:
        dot_cs = np.log10(rms)  # rms assumed unit == mm/yr
    else:
        dot_cs = 'lightgrey'
        axc2.axis('off')

    # upper panel
    bar = ax1.bar(names, conds, color=bar_cs, ec='k')
    ax1.set_ylabel('Condition\nnumber')
    if clog is not None:
        bsc  = ax1.scatter(names, conds, s=0, c=clog, cmap='YlGnBu', vmin=clog.min(), vmax=clog.max())
        cbar = plt.colorbar(bsc, cax=axc1, label=r'$log_{10}(det(C_{m}))$')
        cbar.ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, integer=True))

    # lower panel
    cmap = 'OrRd'
    # cut the numerics lower than that to for plotting purpose
    low_lim = 2e-4  # deg/Ma
    top_lim = 6     # deg/Ma
    ys[ys*RADY2DMY<low_lim] = low_lim/RADY2DMY
    if isinstance(dot_cs, str):
        sc = ax2.scatter(names, ys*RADY2DMY, s=80, c=dot_cs, ec='k', lw=1.5, zorder=10, clip_on=False)
    else:
        sc = ax2.scatter(names, ys*RADY2DMY, s=80, c=dot_cs, ec='k', cmap=cmap, lw=1.5, zorder=10, clip_on=False)
    ax2.grid(axis='y', color='gainsboro', linestyle='-', linewidth=1)
    ax2.axhline(y=thresh_2sigma*RADY2DMY, c='k', ls='--')
    ax2.set_xticks(ticks=names, labels=names, rotation=60.)
    ax2.set_xlim([-0.6, len(names)-0.5])
    ax2.set_ylim([low_lim, top_lim])
    ax2.set_ylabel('Euler pole deviation\n[deg/Myr]')
    cbar = plt.colorbar(sc, cax=axc2, label=r'$log_{10}(rms)$')
    cbar.ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, integer=True))

    # Draw alternating background colors
    for i, name in enumerate(names):
        color = 'none' if i % 2 == 0 else 'gainsboro'
        ax2.axvspan(xmin=i-0.5, xmax=i+0.5, color=color, zorder=0)
    ax2.text(i, thresh_2sigma*RADY2DMY, fr'Altamimi PMM $2\sigma$', va='bottom', ha='right')

    # final edits
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.4)
        ax.spines['bottom'].set_linewidth(1.4)
        ax.spines['bottom'].set_zorder(9)
        ax.spines['left'].set_zorder(9)
        if ylog:
            ax.set_yscale('log')
            # set y ticks
            y_major = matplotlib.ticker.LogLocator(base=10.0, numticks=4)
            ax.yaxis.set_major_locator(y_major)
            y_minor = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(1,10)*0.1, numticks=10)
            ax.yaxis.set_minor_locator(y_minor)
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


    # return
    if outpic is not None:
        plt.savefig(outpic, dpi=600, transparent=True, bbox_inches='tight')
        plt.show()
    else:
        return fig, axs


########################

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def gps_project_los(gpsDF, geom_file, velo_file=None, geom_name=None, columns=['Lat','Lon','Ve','Vn'], radius=2):
    atr       = readfile.read_attribute(geom_file)
    inc_angle = readfile.read(geom_file, datasetName='incidenceAngle')[0]
    azi_angle = readfile.read(geom_file, datasetName='azimuthAngle')[0]
    lats      = readfile.read(geom_file, datasetName='latitude')[0]
    lons      = readfile.read(geom_file, datasetName='longitude')[0]

    if velo_file is not None:
        vlos = readfile.read(velo_file, datasetName='velocity')[0]
    else:
        vlos = np.full(inc_angle.shape, np.nan)

    if geom_name is None:
        geom_name = 'track'+atr['trackNumber']

    width, length = int(atr['WIDTH']), int(atr['LENGTH'])
    radius = int(radius)
    coord  = mut.coordinate(atr)

    # columns in dataframe
    lat = gpsDF[columns[0]]
    lon = gpsDF[columns[1]]
    ve  = gpsDF[columns[2]]
    vn  = gpsDF[columns[3]]

    # get vlos data at these locations
    col_vlos = np.full(len(gpsDF), np.nan)
    col_glos = np.full(len(gpsDF), np.nan)

    gps_out = gpsDF.copy()

    for i, (la, lo, e, n) in enumerate(zip(lat, lon, ve, vn)):
        y, x = coord.lalo2yx(la, lo)
        if (y<0) or (y>length) or (x<0) or (x>width):
            continue
        y0 = np.clip(y-radius, a_min=0, a_max=length)
        y1 = np.clip(y+radius, a_min=0, a_max=length)
        x0 = np.clip(x-radius, a_min=0, a_max=width)
        x1 = np.clip(x+radius, a_min=0, a_max=width)

        vlos_chip = np.nanmedian(vlos[y0:y1, x0:x1])
        inc_chip  = inc_angle[y,x]
        azi_chip  = azi_angle[y,x]
        los_vec   = np.array(ut.get_unit_vector4component_of_interest(los_inc_angle=inc_chip, los_az_angle=azi_chip))
        glos_chip = np.dot(los_vec, np.array([e, n, 0]))

        col_glos[i] = glos_chip
        col_vlos[i] = vlos_chip

    gps_out[geom_name+'_gps']= col_glos
    gps_out[geom_name+'_sar']= col_vlos
    return gps_out


def plot_gps_insar_offset(gpsGDf, dataDict, radius=5, outdir=None):
    """GNSS vs InSAR to get the DC shifts
    """
    # copy current gps DF
    gpsOut = gpsGDf.copy()
    for name in [*dataDict]:
        path = dataDict[name][12]   # insar files paths
        # in gpsDF, add colums for: LOS info and collocated insar data
        gpsOut = gps_project_los(gpsOut, path['geofile'], path['velfile'], geom_name=name, radius=radius)

    sarlist = list(gpsOut.filter(regex='_sar').columns)
    gpslist = list(gpsOut.filter(regex='_gps').columns)
    gpsDC = {}

    # compute median diff and plot
    for name, _ss, _gg in zip([*dataDict], sarlist, gpslist):
        gpsOut[_ss] *= 1e3   # convert sar to mm/yr
        glos = np.array(gpsOut[_gg])
        vlos = np.array(gpsOut[_ss])
        x0, y0 = np.nanmedian(vlos), np.nanmedian(glos)
        xmin, xmax = 1.2*np.nanmin(vlos), 1.2*np.nanmax(vlos)
        diff = y0-x0
        plt.figure(figsize=[2.5,2.5])
        plt.scatter(vlos, glos, s=20, fc='lightgrey', ec='k')
        plt.plot([xmin,xmax],[y0-(x0-xmin),y0+(xmax-x0)], ls='--', c='k')
        plt.title(fr'{name} $GPS-SAR={diff:.3f}$', fontsize=10)
        if outdir is not None:
            savefig = Path(outdir) / f'gpssar_{name}.png'
            plt.savefig(savefig, dpi=200, transparent=True, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        if ~np.isnan(diff):
            gpsDC[name] = diff * 1e-3

    return gpsOut, gpsDC


def unpack_inps(inps):
    if isinstance(inps, str):
        parts = [inps, 1.] # default value for sign
    elif isinstance(inps, (list, tuple, np.ndarray)):
        parts = inps
    else:
        raise ValueError("Invalid input format")
    if len(parts) == 2:
        return parts[0], float(parts[1])
    else:
        raise ValueError("Invalid input format")


def flag2obj(flag, dataDict=None):
    """ convert input flag dict to an obj
    """
    flag = copy.deepcopy(flag)

    # *********   flags to strings   *********
    if  'dtype'   in flag and flag['dtype']   :  flag['featName'] += [flag['dtype']]
    if  'roi_dir' in flag and flag['roi_dir'] :  flag['featName'] += [os.path.basename(flag['roi_dir'])]

    if  'noise'   in flag and flag['noise']   :  flag['featName'] += [flag['noise'][0]]
    else                                      :  flag['noise'] = False

    if  'demean'  in flag and flag['demean']  :  flag['featName'] += ['demean']
    else                                      :  flag['demean'] = False

    if  'conStd'  in flag and flag['conStd']  :  flag['featName'] += ['conStd']
    else                                      :  flag['conStd'] = False

    if  'refG'    in flag and flag['refG']    :  flag['featName'] += ['refG']
    else                                      :  flag['refG'] = False

    if  'est_ramp'in flag and flag['est_ramp']:  flag['featName'] += ['estRamp']
    else                                      :  flag['est_ramp'] = False

    if  'biases'  in flag and flag['biases']  :  flag['featName'] += [flag['biases']+'Bias']
    else                                      :  flag['biases'] = False

    if  'initMdl' in flag and flag['initMdl'] :  flag['featName'] += ['initMdl'+flag['initMdl']]
    else                                      :  flag['initMdl'] = False

    if  'useGPS'  in flag and flag['useGPS']  :  flag['featName'] += ['gps']
    else                                      :  flag['useGPS'] = False

    if  'gpsDC'   in flag and flag['gpsDC']   :  flag['featName'] += ['gpsDC']
    else                                      :  flag['gpsDC'] = False

    if  'priorDC' in flag:
              if not flag['priorDC']          :  flag['priorDC'] = []
              elif   flag['priorDC'] == 'all' :  flag['featName'] += ['priorDC'] ; flag['priorDC'] = [*dataDict]
              else                            :  flag['featName'] += ['priorDC']
    else                                      :  flag['priorDC'] = []

    if  'errname' in flag and flag['errname'] :  flag['featName'] += [flag['errname']]
    if  'errform' in flag and flag['errform'] :  flag['featName'] += [flag['errform']]

    if 'diaglzCov' not in flag                :  flag['diaglzCov'] = False
    if 'saveLD'    not in flag                :  flag['saveLD']    = False
    if 'loadLD'    not in flag                :  flag['loadLD']    = False
    if 'gpuno'     not in flag                :  flag['gpuno']     = None

    # *****************************************
    flag = Struct(**flag)
    return flag


def inputflags2read(dataDict, flag, **kwargs):
    """Fill the input dataDict by reading datasets from files
    Inputs:
        dataDict  - input empty data dict
        flag      - user input flags
        kwargs    - 'ref_i' for MC realizations;

    Outputs:
        dataDict  - filled dataDict

    Examples:
        # normal read
        dataDict = inputflags2read(dataDict, flag)

        # reading structured paths with iterations
        for ref_i in range(ref_realizations):
            dataDict = inputflags2read(dataDict, flag, pole_ref=poleA, ref_i=ref_i)

    """
    # name and make the folder
    flag.out_dir = '_'.join(flag.featName)

    # subfolders?
    if flag.dtype == 'realMC':
        sub_folder = 'ref_' + str(kwargs['ref_i']).zfill(4)
        print(sub_folder)
        flag.out_dir = os.path.join(flag.out_dir, sub_folder)

    os.makedirs(flag.out_dir, exist_ok=True)


    print(f'\ngetting flags to read {len(dataDict)} datasets...')

    for ki, (name, item) in enumerate(dataDict.items()):
        print(f'\nprepare dataset {ki+1}/{len(dataDict)}, {name}')

        # geom
        fbase   = 'geo_{}.h5'
        vals    = [name]
        geofile = Path(flag.data_dir) / fbase.format(*vals)

        # roi mask
        fbase   = 'roi_{}_interior_{}.h5'
        vals    = [flag.projName, name]
        roifile = Path(flag.roi_dir)  / fbase.format(*vals)

        # velocity
        if flag.dtype == 'real':
            # normal referenced velo
            fbase = 'vel_{}_msk.h5'
            vals  = [name]
            velfile = Path(flag.data_dir) / fbase.format(*vals)
        elif flag.dtype == 'realMC':
            # MC referenced velo
            fbase = '{}/{}/MCref_1000/{}/velocity.h5'
            vals  = [name, item['maindir'], sub_folder]
            velfile = Path(flag.mcdatadir) / fbase.format(*vals)
        elif flag.dtype == 'pmmAbs':
            # pmm abs
            fbase = 'pmm_{}_abs/pmm_{}.h5'
            vals  = [flag.nickname, name]
            velfile = Path(flag.data_dir) / fbase.format(*vals)
        elif flag.dtype == 'pmm':
            # pmm
            fbase = 'pmm_{}/pmm_{}.h5'
            vals  = [flag.nickname, name]
            velfile = Path(flag.data_dir) / fbase.format(*vals)

        # velocity std
        if flag.dtype == 'realMC':
            # MC referenced veloStd
            fbase = '{}/{}/MCref_1000/{}/velocity.h5'
            vals  = [name, item['maindir'], sub_folder]
            stdfile = Path(flag.mcdatadir) / fbase.format(*vals)
        else:
            # normal referenced veloStd
            fbase = 'vel_{}_msk.h5'
            vals  = [name]
            stdfile = Path(flag.data_dir) / fbase.format(*vals)

        # accounting external ramp?
        if 'ramp' in item:
            rampfile = item['ramp']
        else:
            rampfile = None

        # read from files
        input_data = read_data_geom( velfile                      ,
                                     geofile                      ,
                                     stdfile                      ,
                                     roifile     = roifile        ,
                                     rampfile    = rampfile       ,
                                     looks       = flag.looks     ,
                                     flag_demean = flag.demean    ,
                                     flag_conStd = flag.conStd    ,
                                     )

        (vlos, vstd, inc_angle, azi_angle, lats, lons, roi, ref_los_vec, refyx, reflalo, bbox) = input_data
        refy, refx = refyx[0], refyx[1]

        # ------------ add artificial noise ----------------
        if flag.noise:
            key, param = flag.noise
            print(f'Add artificial noise to velocity: {key,param}')

            # white noise
            if key == 'whitenoise':
                mean, sigma = param
                in_noise = np.random.normal(mean, sigma, vlos.shape)
                #in_noise[refy, refx] = float(mean)  # ref point always be at zero noise

            # noise at reference point
            elif key == 'refnoise':
                mean, sigma = param
                in_noise = np.zeros_like(vlos)
                in_noise[refy, refx] = np.random.normal(mean, sigma)

            # add some prior residual as noise
            elif key == 'residual':
                #in_noise = np.load(param, allow_pickle=True)[()][name] # from my old .npy file
                infile, sign = unpack_inps(param)
                with open(infile, 'rb') as f:  # from dump.pkl file
                    block = pickle.load(f)[1]
                    in_noise = sign * np.array(block.res_set[ki])

            # add prior model diff (a long-wavelength screen)
            elif key == 'modeldiff':
                infile, sign = unpack_inps(param)
                with open(infile, 'rb') as f:  # from dump.pkl file
                    block = pickle.load(f)[1]
                    in_noise = sign * np.array(block.model_diff[ki])

            vlos += in_noise
            print(f'after noise adding ({in_noise[refy, refx]:.6f}), ref_val = {vlos[refy, refx]:.6f}')


        # ---------- get the std input scaling factor ------------
        std_scl = item.get('std_scale', 1.0)

        # ---------- paper revision process: get ramp rate error ------------
        ramp_rate_err = item.get('sig_ramp', None)

        # ------- build the dataDict: a tuple of data input ------
        paths = {}
        paths['velfile'] = velfile
        paths['geofile'] = geofile
        paths['stdfile'] = stdfile
        paths['roifile'] = roifile
        comp = 'los'
        dataDict[name] = (vlos, vstd, inc_angle, azi_angle, lats, lons, roi, ref_los_vec, refyx, reflalo, bbox, std_scl, paths, comp, ramp_rate_err)


    # ------ add synthetic data from a existing geometry ------
    if flag.synthetic:
        names = flag.synthetic['name']
        comp  = flag.synthetic['comp']
        errs  = flag.synthetic['error']
        pole  = kwargs['pole_ref']
        print(f'Add {comp} synthetic from existing geometry {names}, set error {errs}')
        print(f'Synthetic pole: {pole}')

        if isinstance(names, str):
            names = [names]
            errs  = [float(errs)]
        elif isinstance(names, (list, tuple)):
            if isinstance(errs, float):
                errs = [float(errs)] * len(names)

        for (name, err) in zip(names, errs):
            # read a copy from existing dataset
            vlos, vstd, inc_angle, azi_angle, lats, lons, roi, ref_los_vec, refyx, reflalo, bbox, std_scl, paths, _ = dataDict[name]

            # get 2d array of azimuthal velocity
            v_proj = ut.project_synthetic_motion(pole, lats, lons, inc_angle, azi_angle, roi, comp)

            # component and reference info
            if comp in ['azi','azimuth','en2az']:
                in_comp = 'azimuth'
                ref_los_vec = None
                refyx       = None
                reflalo     = None
                paths       = None
                v_proj_std  = roi * err  # azimuth data uncorrelated error
                v_proj_std[~roi] = np.nan
            elif comp in ['los','enu2los']:
                in_comp = 'los'
                v_proj -= v_proj[refyx[0], refyx[1]]
                v_proj_std = np.array(vstd)
            else:
                sys.exit(f'cannot recognize compoenent: {comp}')

            # store into dataDict
            std_scl = 1.
            dataDict[name+comp] = (v_proj, v_proj_std, inc_angle, azi_angle, lats, lons, roi, ref_los_vec, refyx, reflalo, bbox, std_scl, paths, in_comp)

            # whether to replace the orig dataset
            if flag.synthetic['replace']:
                dataDict.pop(name)

    return dataDict


def show_mask(mask, refy, refx, width, length, win=None, cmap='viridis', vmin=None, vmax=None):
    if win is not None:
        y0 = np.clip(refy-win, a_min=0, a_max=length)
        y1 = np.clip(refy+win, a_min=0, a_max=length)
        x0 = np.clip(refx-win, a_min=0, a_max=width)
        x1 = np.clip(refx+win, a_min=0, a_max=width)
        roi = np.array(mask[y0:y1,x0:x1])
        extent=[x0,x1,y1,y0]
    else:
        roi = np.array(mask)
        extent=[0,width,length,0]
    plt.figure()
    plt.imshow(roi, interpolation='none', origin='upper', extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.scatter(refx, refy, c='k')
    plt.title(f'{(refy, refx)} : {mask[refy,refx]}')
    plt.show()


def mask_roi_and_data(roi, data, std, inc, azi, lat, lon):
    nan_bool = np.isnan(data) + np.isnan(std)
    for geo in [inc, azi, lat, lon]:
        nan_bool += np.isnan(geo)
    roi[nan_bool] = False
    data[~roi]    = np.nan
    std[~roi]     = np.nan
    inc[~roi]     = np.nan
    azi[~roi]     = np.nan
    print(f'  ::: valid data dimension: {np.sum(~np.isnan(data))}')
    return


def read_data_geom( velfile     : str,
                    geofile     : str,
                    stdfile     : str   | None = None,
                    roifile     : str   | None = None,
                    rampfile    : str   | None = None,
                    looks       : int   | None = 1,
                    refyx       : tuple | None = None,
                    flag_demean : bool  | None = False,
                    flag_conStd : bool  | None = False,
                    ) -> tuple:
    """Read and downsample the input data
    + based on MintPy read.py module
    + skimage downscale_local_mean()
    """

    atr       = readfile.read(velfile)[1]
    vlos      = readfile.read(velfile, datasetName='velocity')[0]
    inc_angle = readfile.read(geofile, datasetName='incidenceAngle')[0]
    azi_angle = readfile.read(geofile, datasetName='azimuthAngle')[0]
    lats      = readfile.read(geofile, datasetName='latitude')[0]
    lons      = readfile.read(geofile, datasetName='longitude')[0]

    if roifile is not None: roi  = readfile.read(roifile)[0]
    else:                   roi  = np.full(vlos.shape, 1).astype(bool)

    if stdfile is not None: vstd  = readfile.read(stdfile, datasetName='velocityStd')[0]
    else:                   vstd  = np.full(vlos.shape, 1.0)

    if rampfile is not None: ramp  = readfile.read(rampfile, datasetName='velocity')[0]
    else:                    ramp  = None


    width, length = int(atr['WIDTH']), int(atr['LENGTH'])
    if 'REF_Y' in atr.keys():
        ref_y   = int(atr['REF_Y'])
        ref_x   = int(atr['REF_X'])
        ref_la  = float(atr['REF_LAT'])
        ref_lo  = float(atr['REF_LON'])
        ref_inc = inc_angle[ref_y, ref_x]
        ref_azi = azi_angle[ref_y, ref_x]

        if ramp is not None:
            ramp -= ramp[ref_y, ref_x]

    else:
        ref_los_vec    = None
        ref_y,  ref_x  = None, None
        ref_la, ref_lo = None, None


    if ramp is not None:
        print(f' remove external ramp from file: {rampfile}')
        print(f' vlos median = {np.nanmedian(vlos)}')
        print(f' ramp median = {np.nanmedian(ramp)}')
        vlos -= ramp

    # los unit vector is positive, pointing from ground to satellite; up motion proj to positive los motion
    ref_los_vec = np.array(ut.get_unit_vector4component_of_interest(los_inc_angle=ref_inc, los_az_angle=ref_azi))
    print(f'use reference point at Lat/Lon : {ref_la:.2f}/{ref_lo:.2f} ({ref_y}/{ref_x})')
    print(f'      incidence/azimuth angles : {ref_inc:.2f}/{ref_azi:.2f}')
    print(f'               LOS unit vector : {ref_los_vec}')
    roi_ref  = bool(roi[ref_y, ref_x])
    vlos_ref = float(vlos[ref_y, ref_x])
    vstd_ref = float(vstd[ref_y, ref_x])


    # clean weird edges and update roi and data to mask union of nans
    vstd[vstd==0.0] = np.nan
    vlos[vlos==0.0] = np.nan
    vstd[ref_y,ref_x] = 0.0
    vlos[ref_y,ref_x] = 0.0
    mask_roi_and_data(roi, vlos, vstd, inc_angle, azi_angle, lats, lons)


    if 'LAT_REF1' in atr.keys():
        lat1, lon1 = float(atr['LAT_REF1']), float(atr['LON_REF1'])
        lat2, lon2 = float(atr['LAT_REF2']), float(atr['LON_REF2'])
        lat3, lon3 = float(atr['LAT_REF3']), float(atr['LON_REF3'])
        lat4, lon4 = float(atr['LAT_REF4']), float(atr['LON_REF4'])
        latc = [lat1, lat2, lat4, lat3]
        lonc = [lon1, lon2, lon4, lon3]


    if int(looks) > 1:
        lk_opt = 'mintpy'
        print(f'  ::: downsample data with {looks}, {lk_opt}')
        ref_y     = int(ref_y / looks)
        ref_x     = int(ref_x / looks)

        if lk_opt == 'mintpy': # mintpy multilook
            roi       = multilook_data(roi      , looks, looks, 'mean', no_data_val=np.nan) == 1
            vlos      = multilook_data(vlos     , looks, looks, 'median', no_data_val=np.nan)
            vstd      = multilook_data(vstd     , looks, looks, 'median', no_data_val=np.nan)
            inc_angle = multilook_data(inc_angle, looks, looks, 'median', no_data_val=np.nan)
            azi_angle = multilook_data(azi_angle, looks, looks, 'median', no_data_val=np.nan)
            lats      = multilook_data(lats     , looks, looks, 'median', no_data_val=np.nan)
            lons      = multilook_data(lons     , looks, looks, 'median', no_data_val=np.nan)

        if lk_opt == 'scipy': # scipy can screw up your nans, end up more holes
            roi       = downscale_local_mean(roi.astype(float), (looks, looks), cval=np.nan) == 1
            vlos      = downscale_local_mean(vlos     , (looks, looks), cval=np.nan)
            vstd      = downscale_local_mean(vstd     , (looks, looks), cval=np.nan)
            inc_angle = downscale_local_mean(inc_angle, (looks, looks), cval=np.nan)
            azi_angle = downscale_local_mean(azi_angle, (looks, looks), cval=np.nan)
            lats      = downscale_local_mean(lats     , (looks, looks), cval=np.nan)
            lons      = downscale_local_mean(lons     , (looks, looks), cval=np.nan)

        print(f'  ::: mininum abs velo/vstd = {np.nanmin(np.abs(vlos))}/{np.nanmin(vstd)}')

    # set ref point and lowest uncertainty
    vstd_low            = 5e-5    # do not set to zero or too small (m/year)
    roi[ref_y, ref_x]   = True
    vlos[ref_y, ref_x]  = 0.0
    vstd[ref_y, ref_x]  = vstd_low
    vstd[vstd<vstd_low] = vstd_low


    # update roi and data to mask union of nans
    mask_roi_and_data(roi, vlos, vstd, inc_angle, azi_angle, lats, lons)

    if True:
        print('-'*60)
        print('  ::: roi at refyx B/A sampling: ', roi_ref, roi[ref_y, ref_x])
        print(f'  ::: -> all valids {np.sum(roi)}')
        print(f'  ::: --> roi  {np.sum(~np.isnan(roi))}')
        print(f'  ::: --> vlos {np.sum(~np.isnan(vlos))}')
        print(f'  ::: --> vstd {np.sum(~np.isnan(vstd))}')
        print(f'  ::: --> inc  {np.sum(~np.isnan(inc_angle))}')
        print(f'  ::: --> azi  {np.sum(~np.isnan(azi_angle))}')
        print(f'  ::: --> lats {np.sum(~np.isnan(lats))}')
        print(f'  ::: --> lons {np.sum(~np.isnan(lons))}')
        print(f'  ::: subsampled data dimension: {np.sum(~np.isnan(vlos))}')
        #show_mask(1e3*vstd, ref_y, ref_x, width//looks, length//looks, cmap='magma_r', vmin=0.2, vmax=0.8)


    if int(looks) > 1:
        print(f'prior downsamp, ref_val = {vlos_ref}, ref_vstd = {vstd_ref}')
        print(f'after downsamp, ref_val = {vlos[ref_y, ref_x]}, ref_vstd = {vstd[ref_y, ref_x]}')

    # --------------- forcing option ---------- (not recommended)
    # demean or use median of the data Std
    if flag_conStd:
        vstd = np.nanmedian(vstd) * np.ones_like(vstd)
        print(f'use constant field veloStd: {np.nanmedian(vstd)} -> a constant diag of Cd')

    # remove the median from data (like applying centroid of ref points)
    if flag_demean:
        mean = np.nanmedian(vlos)
        print(f'subtract data mean: {mean}')
        vlos -= mean
    # --------------- forcing option ---------- (not recommended)

    return vlos, vstd, inc_angle, azi_angle, lats, lons, roi, ref_los_vec, (ref_y,ref_x), (ref_la,ref_lo), (latc,lonc)


def read_lalo_angles(dataDict, lat, lon, eps=1e-2):
    LOS = {}
    for k in [*dataDict]:
        incs = dataDict[k][2]
        azis = dataDict[k][3]
        lats = dataDict[k][4]
        lons = dataDict[k][5]
        y = np.nanargmin(np.abs(lats-lat)) // lats.shape[1]
        x = np.nanargmin(np.abs(lons-lon)) %  lons.shape[1]

        if np.nanmean(np.abs(lats[y,:]-lat)) < eps:
            if np.nanmean(np.abs(lons[:,x]-lon)) < eps:
                print(f'{k} x,y={x},{y} | lat={lats[y,x]} lon={lons[y,x]} inc={incs[y,x]} azi={azis[y,x]}')
                inc = incs[y,x]
                azi = azis[y,x]
                los_vec = ut.get_unit_vector4component_of_interest(los_inc_angle=inc, los_az_angle=azi)
                LOS[k] = {}
                LOS[k]['unit_vector']    = np.array(los_vec)
                LOS[k]['incidenceAngle'] = float(inc)
                LOS[k]['azimuthAngle']   = float(azi)
    return LOS


def read_GNSS_files(infile,
                    columns   = ['Station','Lon','Lat','Ve','Vn','Ve_arab','Vn_arab','Vesig','Vnsig'],
                    bbox      = None,
                    platename = None,
                    polyfile  = None,
                    txtfile   = None,
                    source    = 'mydata',
                    ):
    # pandas loaded table
    #gpsDf  = pd.read_table(infile, skiprows=7, skipfooter=1, names=columns, sep=r'\s+', engine='python')
    gpsDf  = pd.read_table(infile, names=columns, sep=r'\s+', comment='#', engine='python')

    # to geopandas
    gpsGDf = gpd.GeoDataFrame(gpsDf, geometry=[Point(xy) for xy in zip(gpsDf.Lon, gpsDf.Lat)])
    gpsGDf.Station = gpsGDf.Station.str.split('*').str[-1]


    # 1. apply bounding box filter if provided
    if bbox:
        lat0, lat1, lon0, lon1 = bbox
        bbox_polygon = Polygon([(lon0,lat0), (lon1,lat0), (lon1,lat1), (lon0,lat1), (lon0,lat0)])
        bbox_gdf     = gpd.GeoDataFrame(index=[0], geometry=[bbox_polygon], crs=gpsGDf.crs)
        gpsGDf       = gpd.clip(gpsGDf, bbox_gdf)

    # 2. overlay plate geopandas if provided
    if platename:
        bndGDf = gpd.GeoDataFrame(index=[0], geometry=[read_plate_outline('MORVEL', platename, order='lola')])
        gpsGDf = gpd.overlay(gpsGDf, bndGDf)

    # 3. overlay polygon file geopandas if provided
    if polyfile:
        x = pd.read_csv(polyfile, comment='#', names=['lon','lat'])['lon'].tolist()
        y = pd.read_csv(polyfile, comment='#', names=['lon','lat'])['lat'].tolist()
        polyGDf = gpd.GeoDataFrame(index=[0], geometry=[Polygon(list(zip(x, y)))])
        gpsGDf  = gpd.overlay(gpsGDf, polyGDf)

    # 4. filter by station names in a txt file
    if txtfile:
        sites = []
        with open(txtfile) as f:
            for line in f.readlines():
                ele = line.splitlines()[0]
                if not ele.startswith('#') and len(ele)>1:
                    sites.append(ele.split()[0])
        gpsGDf = gpsGDf[gpsGDf.Station.isin(sites)]

    # add a dataset name
    gpsGDf.Source = source

    return gpsGDf


def plugin_GNSS(block, gpsDF, site, table_unit='mm', uniErr=False):

    if site == 'all':
        site = list(gpsDF['Station'])

    print(f'plug in GNSS: {site}')

    if isinstance(site, str):
        df = gpsDF[gpsDF['Station']==site]
    elif isinstance(site, (list, tuple, np.ndarray)):
        df = gpsDF[gpsDF['Station'].isin(site)]

    lon    = df.Lon.to_numpy()
    lat    = df.Lat.to_numpy()
    ve, vn = df.Ve.to_numpy()   , df.Vn.to_numpy()
    se, sn = df.Vesig.to_numpy(), df.Vnsig.to_numpy()

    if table_unit == 'mm':
        ve *= 1e-3
        vn *= 1e-3
        se *= 1e-3
        sn *= 1e-3
    if table_unit == 'cm':
        ve *= 1e-2
        vn *= 1e-2
        se *= 1e-2
        sn *= 1e-2
    if uniErr:
        se = sn = float(uniErr)

    block.add_gps2insar(lat=lat, lon=lon, ve=ve, vn=vn, se=se, sn=sn)

    return block


def compare_block_pole(block, pmm_pole, names=None, savefig=None):
    bComp = int(block.bias_comp)

    # pmm-based DC shifts
    los_vecs = np.array( block.ref_los_vec_set )
    lats     = np.array( block.ref_lalo_set )[:,0]
    lons     = np.array( block.ref_lalo_set )[:,1]
    dc_pmm, dc_pmm_std = DC_from_pole(pmm_pole, lats, lons, los_vecs, bComp=bComp)

    # insar-based LOS DC shifts
    if block.bias:
        dc_est, dc_est_std = DC_from_block(block, bComp=bComp, priorDC=True)
    else:
        dc_est, dc_est_std = np.full_like(dc_pmm, np.nan), np.full_like(dc_pmm, np.nan)

    dc_est     = dc_est     * 1e3
    dc_est_std = dc_est_std * 1e3
    dc_pmm     = dc_pmm     * 1e3
    dc_pmm_std = dc_pmm_std * 1e3

    # difference
    dc_diff = dc_est - dc_pmm

    # pole and DC info
    pole = block.create_pole(name='input')
    pole.get_uncertainty(block=block, src='block')
    dist_dc       = np.linalg.norm(dc_diff[~np.isnan(dc_diff)])    # mm/yr
    deg_p, dist_p = ut.haversine_distance([pole.poleLat, pole.poleLon], [pmm_pole.poleLat, pmm_pole.poleLon]) # deg, meters
    print(f'DC shift deviation norm : {dist_dc:10.2f} [mm/yr]')
    print(f'Pole deviation          : {deg_p:10.2f} [deg]')
    print(f'Pole deviation          : {dist_p*1e-3:10.2f} [km]')


    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(figsize=[9,4], ncols=2, layout='tight')
    args = { 'Asc'    : [axs[0],    'crimson'],
             'Dsc'    : [axs[1], 'dodgerblue'],
             'prior'  : 'lightgrey',
             'marker' : 'o',
             'mew'    : 2,
             'ms'     : 12,
             'mec'    : 'k',
             'elw'    : 2,
             'cap_sz' : 6,
             'cap_th' : 2,
             }

    if names is None: names = [''] * len(dc_est)
    for i, (x, y, xerr, yerr, name) in enumerate(zip(dc_pmm, dc_est, dc_pmm_std, dc_est_std, names)):
        tk   = ut.get_track_name(name, style='3')[:3]
        delta= y-x
        if not np.isnan(yerr):
            text = fr'  {name} $^{{\Delta={delta:.2f}}}_{{±{yerr:.2f}}}$'
        else:
            text = fr'  {name} $^{{\Delta={delta:.2f}}}$'
        ax   = args[tk][0]
        mfc  = args[tk][1]

        # if you didn't estimate this bias term, keep it grey
        if np.isnan(yerr):
            mfc = args['prior']

        errc = tweak_color(mfc, 0.8)

        ax.plot([x,x], [x,y], ls='--', c='k', lw=1.5)

        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=args['marker'], ms=args['ms'], mew=args['mew'],
                    mec=args['mec'], mfc=mfc, c=errc, elinewidth=args['elw'],
                    capsize=args['cap_sz'], capthick=args['cap_th'])

        if isinstance(x, (list, tuple, np.ndarray)):
            for _x, _y in zip(x,y):
                ax.text(_x, _y, text)
        else:
            ax.text(x, y, text)

    for i, (ax, tk) in enumerate(zip(axs, ['Asc','Dsc'])):
        # 1:1 line
        ymin, ymax = ax.get_ylim()
        ymax = ymax + (ymax - ymin)*0.2
        ax.plot([ymin,ymax], [ymin,ymax], ls='-', c='lightgrey', lw=2.5)
        ax.text((ymin+ymax)/2, (ymin+ymax)/2, '1:1', va='bottom', rotation=45)

        # title
        name = ut.get_track_name(tk, 'full')
        ax.text(0.02, 0.98, name, transform=ax.transAxes, fontsize=14, va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        # legend
        p0 = Line2D([0],[0], label=f'GPS prior', marker=args['marker'],
                    ms=args['ms'], mfc=args['prior'], mec='k', mew=args['mew'],linestyle='')
        p1 = Line2D([0],[0], label=f'InSAR est.',  marker=args['marker'],
                    ms=args['ms'],   mfc=args[tk][1], mec='k', mew=args['mew'],linestyle='')
        handles = [p0, p1]
        ax.legend(handles=handles, loc='lower right', fontsize=10, markerscale=0.95, bbox_to_anchor=(1,0))

        # axis setting
        ax.set_xlabel('Plate model DCs [mm/yr]')
        text_accomodating_xylim(fig, ax)
    axs[0].set_ylabel('Estimated DCs [mm/yr]')

    if savefig:
        plt.savefig(savefig, dpi=200, transparent=True, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return dc_diff, deg_p, dist_p



def plot_DCs_old(dataDict, savefig=None):

    # split asc/dsc datasets
    iasc = [k.lower().startswith('a') for k in [*dataDict]]
    idsc = [k.lower().startswith('d') for k in [*dataDict]]

    # scatter size
    ss = 7

    fig, axs = plt.subplots(figsize=[5,5], nrows=2, sharex=True, layout='tight')
    ax1 = axs[0]
    acolor = 'crimson'
    ax1.errorbar(np.arange(0,sum(iasc)), dc_pmm[iasc], yerr=2*dc_pmm_std[iasc], capsize=ss, marker='s', markersize=ss, ls='none', c='k', label='PMM')
    ax1.errorbar(np.arange(0,sum(iasc)), dc_est[iasc], yerr=2*dc_est_std[iasc], capsize=ss, marker='o', markersize=ss, ls='none', c=acolor)
    ax1.set_ylabel('Ascending DC shift [mm/yr]', color=acolor)
    ax1.tick_params(axis='y', labelcolor=acolor)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    dcolor = 'dodgerblue'
    ax2.errorbar(np.arange(sum(iasc), len(iasc)), dc_pmm[idsc], yerr=2*dc_pmm_std[idsc], capsize=ss, marker='s', markersize=ss, ls='none', c='k', label='PMM')
    ax2.errorbar(np.arange(sum(iasc), len(iasc)), dc_est[idsc], yerr=2*dc_est_std[idsc], capsize=ss, marker='o', markersize=ss, ls='none', c=dcolor)
    ax2.set_ylabel('Descending DC shift [mm/yr]', color=dcolor)
    ax2.tick_params(axis='y', labelcolor=dcolor)

    ax3 = axs[1]
    ax3.axhline(y=0, ls='--', c='k', alpha=0.7)
    ax3.errorbar(np.arange(0,sum(iasc)),          dc_est[iasc]-dc_pmm[iasc], yerr=2*dc_est_std[iasc], capsize=ss, marker='o', markersize=ss, ls='none', c=acolor, label='Asc')
    ax3.errorbar(np.arange(sum(iasc), len(iasc)), dc_est[idsc]-dc_pmm[idsc], yerr=2*dc_est_std[idsc], capsize=ss, marker='o', markersize=ss, ls='none', c=dcolor, label='Dsc')
    ax3.set_xlabel('# InSAR track')
    ax3.set_ylabel('Deviation [mm/yr]')
    ax3.set_xticks(np.arange(len(iasc)))
    ax3.set_xticklabels([*dataDict])
    ylim = 1.1 * np.nanmax(np.abs(dc_diff))
    try:
        ax3.set_ylim(-ylim, ylim)
    except:
        pass

    ax1.legend(loc='lower left', frameon=True, markerscale=0.95)
    ax3.legend(loc='lower left', frameon=True, markerscale=0.95)
    if savefig:
        plt.savefig(savefig, dpi=200, transparent=True, bbox_inches='tight')

    plt.close()




# ******** newly added, better make them more general for different datasets *********

def plot_datain(dataDict, savefig=None):
    # plot input data
    ncols   = len(dataDict)
    figsize = [ncols*0.9,12]
    fontsize = 10
    fig, axs = plt.subplots(ncols=ncols, nrows=4, figsize=figsize)
    axs = axs.reshape(-1,ncols)

    for i, k in enumerate([*dataDict]):
        v     = dataDict[k][0] * 1e3   # in mm/yr
        std   = dataDict[k][1] * 1e3   # in mm/yr
        inc   = dataDict[k][2]
        azi   = dataDict[k][3]
        refyx = dataDict[k][8]

        # data is referenced?
        if refyx is not None and refyx!=(None,None):
            vlim  = [-4, 4]
            vslim = [0.2, 0.8]
        else:
            vlim  = [None,None]
            vslim = [None,None]

        # plot
        plot_imshow(axs[0,i],  v,  vlim=vlim  , shrink=0.7, title=k,        label='mm/yr', aspect='auto', fontsize=fontsize)
        plot_imshow(axs[1,i], std, vlim=vslim , shrink=0.7, cmap='magma_r', label='mm/yr', aspect='auto', fontsize=fontsize)
        plot_imshow(axs[2,i], inc, label='degree', aspect='auto', fontsize=fontsize, verbose=True)
        plot_imshow(axs[3,i], azi, label='degree', aspect='auto', fontsize=fontsize, verbose=True)

        # add reference point symbol?
        if refyx is not None and refyx!=(None,None):
            for ax in axs[:,i]:
                ax.scatter(refyx[1], refyx[0], s=10, c='k', marker='s')

    axs[0,0].text(-.12, .5, 'velocity',    rotation=90, va='center', fontsize=fontsize, transform=axs[0,0].transAxes)
    axs[1,0].text(-.12, .5, 'velocityStd', rotation=90, va='center', fontsize=fontsize, transform=axs[1,0].transAxes)
    axs[2,0].text(-.12, .5, 'incidence',   rotation=90, va='center', fontsize=fontsize, transform=axs[2,0].transAxes)
    axs[3,0].text(-.12, .5, 'azimuth',     rotation=90, va='center', fontsize=fontsize, transform=axs[3,0].transAxes)
    if savefig:
        plt.savefig(savefig, dpi=200, transparent=True, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return


def plot_datapdf(block, dataDict, savefig=None):
    bins     = 32
    fontsize = 14
    histtype = 'stepfilled'
    density  = True
    fig, ax = plt.subplots(figsize=[10,3], ncols=3, sharey=False, gridspec_kw={'wspace':0.1})
    res_all = []
    for k in dataDict.keys():
        ki  = block.names.index(k)
        obs = block.Obs_set[ki].flatten() * 1e3
        std = block.std_set[ki].flatten() * 1e3
        res = block.res_set[ki].flatten() * 1e3
        ax[0].hist(obs, bins=bins, ec='#4889DA', color='#7381F4', histtype=histtype, density=density, lw=2, alpha=0.4)
        ax[1].hist(std, bins=bins, ec='#E8563F', color='#FC8370', histtype=histtype, density=density, lw=2, alpha=0.4)
        ax[2].hist(res, bins=bins, ec='#35B898', color='#35B898', histtype=histtype, density=density, lw=2, alpha=0.4)
        ax[2].axvline(x=np.nanmean(res), c='grey', alpha=0.4, zorder=-1)
        ax[0].set_ylabel('Pixel population density')
        ax[0].set_xlabel('velocity [mm/yr]'                        , fontsize=fontsize)
        ax[1].set_xlabel(r'velocity $\sigma$ ($C_d^{0.5}$) [mm/yr]', fontsize=fontsize)
        ax[2].set_xlabel('Residual velocity [mm/yr]'               , fontsize=fontsize)
        ax[0].set_xlim(-5.1,5.1); ax[1].set_xlim(-0.02,1.02); ax[2].set_xlim(-3.1,3.1)
        ax[0].set_yticks([]); ax[1].set_yticks([]); ax[2].set_yticks([])
        res_all += [*res]
    res_all = np.array(res_all)
    ax[2].text(0.05, 0.8, f'mean: {np.nanmean(res_all):.3f}\nsigma: {np.nanstd(res_all):.3f}', transform=ax[2].transAxes, fontsize=10)
    if savefig:
        plt.savefig(savefig, dpi=200, transparent=True, bbox_inches='tight')
    plt.close()
    return


def plot_globes(poleDict, flag, savefigs=None):
    """plot pole location, and predicted motions on a globe
    Input:
        poleDict

    example:
        poleDict = {
            'InSAR'     : [pole         , 'b'     ] ,
            'Altamimi'  : [poleA        , 'r'     ] ,
            '*Nubia'    : [poleB        , 'grey'  ] , # starts with * means the alternative ref plate
        }
    }

    """
    plateName = flag.plateName

    for k in poleDict.keys():
        if k.startswith('*'):
            if not k[-1].isdigit():
                refp = k.split('*')[-1]

    for m in [*poleDict]:
        if m.startswith('Altamimi') or m.startswith('ITRF2014'):
            poleA   = poleDict[m][0]

    pole    = poleDict['InSAR'][0]
    poleB   = poleDict['*'+refp][0]

    # plot plate motion & pole
    bndPoly = read_plate_outline('MORVEL', plateName)
    ax = plot_plate_motion( plate_boundary=bndPoly,
                            epole_obj=pole,
                            satellite_height=6*1e6,
                            center_lalo='mid',
                            qunit=5, qscale=1, qwidth=.004,
                            pts_ms=5, qnum=10, unit='cm',
                            lw_coast=0,
                            font_size=10,
                            )[0]
    ax.set_title(f'{plateName} rel. ITRF2014')
    cov   = np.flip(pole.sph_cov_deg[:2,:2])  # {lat,lon} flip to {lon,lat}
    draw_confidence_ellipse(ax, x=pole.poleLon, y=pole.poleLat, cov=cov, n_std=2, color='b', elp_lw=2, elp_alpha=0.3)
    if savefigs:
        plt.savefig(savefigs[0], dpi=200, transparent=True, bbox_inches='tight')
    plt.close()


    # plot plate motion in NNR
    bndPoly = read_plate_outline('MORVEL', plateName)
    ax = plot_plate_motion( plate_boundary=bndPoly,
                            epole_obj=[poleA, pole],
                            satellite_height=9*1e5,
                            helmert=[None, None], orb=True,
                            qunit=5, qscale=1, qwidth=.004,
                            qcolor=['r','b'],
                            qname=[f'{plateName} ITRF14','InSAR (ITRF14)'],
                            pts_ms=5, qnum=10, unit='cm',
                            lw_coast=0,
                            font_size=10,
                            )[0]
    ax.set_title(f'{plateName} rel. ITRF2014')
    if savefigs:
        plt.savefig(savefigs[1], dpi=200, transparent=True, bbox_inches='tight')
    plt.close()


    # plot plate motion rel. to a nearby plate
    bndPoly = read_plate_outline('MORVEL', plateName)
    ax = plot_plate_motion( plate_boundary=bndPoly,
                            epole_obj=[poleA-poleB, pole-poleB],
                            satellite_height=9*1e5,
                            helmert=[None, None], orb=True,
                            qunit=5, qscale=10, qwidth=.004,
                            qcolor=['r','b'],
                            qname=[f'{plateName} ITRF14','InSAR (ITRF14)'],
                            pts_ms=5, qnum=10, unit='mm',
                            lw_coast=0,
                            font_size=10,
                            )[0]
    ax.set_title(f'{plateName} rel. {refp}')
    if savefigs:
        plt.savefig(savefigs[2], dpi=200, transparent=True, bbox_inches='tight')
    plt.close()
    return


def plot_quadriptych(poleDict, frames=['ITRF2014'], compare=None,
                     gpsGDf=None, savefig=None, show=False, bboxs=None,
                     style=1, gpsGDf_full=None, prof_extent=[15,30,35,60], figsize=None, leg_ncol=3,
                     **kwargs):
    """plot pole location, covariance, and predicted motions
    Input:
        poleDict

    example:
        poleDict = {
            'InSAR'     : [pole         , 'b'     ] ,
            'Altamimi'  : [poleA        , 'r'     ] ,
            'GSRM_v2.1' : [poleA_GSRM   , 'g'     ] ,
            'MORVEL56'  : [poleA_MORVEL , 'purple'] ,
            'Viltres+'  : [poleA_viltres, 'pink'  ] ,
            '*Nubia'    : [poleB        , 'grey'  ] , # starts with * means the alternative ref plate
                   }
    }
    """
    pole_globe  = False
    cov_n_std   = kwargs.get('cov_n_std'  , 2            )  # covariance ellipse, n-sigma
    sat_height  = kwargs.get('sat_height' , 4e5          )  # globe plot, satellite height
    center_lalo = kwargs.get('center_lalo', (26, 43)    )  # globe plot, satellite height
    font_size   = kwargs.get('font_size'  , 12           )  # figure size
    map_style   = kwargs.get('map_style'  , 'globe'      )  # map projection
    grid_lc     = kwargs.get('grid_lc'    , 'none'       )  # grid linecolor
    helmert     = kwargs.get('helmert'    , None         )  # helmert transforming the vectors
    orb         = kwargs.get('orb'        , False        )  # Altamimi's ORB
    extent      = kwargs.get('extent'     , [33,43,25,34])  # map extent
    rate_lim    = kwargs.get('rate_lim'   , [None,None]  )  # angular velocity min/max
    qscale      = kwargs.get('qscale'     , 1            )  # quiver scale
    qwidth      = kwargs.get('qwidth'     , .0075        )  # quiver width
    pts_ms      = kwargs.get('pts_ms'     , 5            )  # points size
    qunit       = kwargs.get('qunit'      , 10           )  # quiver physical length in terms of quiver unit
    qnum        = kwargs.get('qnum'       , 10           )  # quiver number every n grid
    unit        = kwargs.get('unit'       , 'mm'         )  # quiver unit
    plateName   = kwargs.get('plateName'  , 'Arabia'     )  # polygon file
    qkX         = kwargs.get('qkX'        , None         )  # quiver key location params
    qkY         = kwargs.get('qkY'        , None         )  # quiver key location params
    qkdX        = kwargs.get('qkdX'       , None         )  # quiver key location params
    qkdY        = kwargs.get('qkdY'       , None         )  # quiver key location params


    # show all poles & covariance
    names = [p for p in poleDict.keys() if not (p.startswith('*') or p.startswith('+'))]

    # Find the index of 'InSAR' or 'GPS' first
    if 'InSAR' in names:
        idx_0 = names.index('InSAR')
        names = [names[idx_0]] + names[:idx_0] + names[idx_0+1:]
    elif 'GNSS' in names:
        idx_0 = names.index('GNSS')
        names = [names[idx_0]] + names[:idx_0] + names[idx_0+1:]


    poles  = [poleDict[key][0] for key in names]
    colors = [poleDict[key][1] for key in names]

    # plot vector with diff ref plate
    arts = []
    for ax4_frame in frames:
        # make two plots with different vectors in axis 4

        #------ allocate fig, axis -------
        if style==0:
            if figsize is None:
                figsize=(8,4*1.15)
            fig      = plt.figure(figsize=figsize)
            gspec    = fig.add_gridspec(2, 2, wspace=0.03, hspace=0.03, height_ratios=[1, 0.3])
            if leg_ncol is None:
                leg_ncol = 3
            qkX, qkY, qkdX, qkdY = 0.1, -0.03, 0.18, -0.06

        elif style==1:
            if figsize is None:
                figsize=(8,8*1.15)
            fig      = plt.figure(figsize=figsize)
            gspec    = fig.add_gridspec(3, 2, wspace=0.03, hspace=0.03, height_ratios=[1, 1, 0.3])
            if leg_ncol is None:
                leg_ncol = 3

        elif style==2:
            if figsize is None:
                figsize=(8,8*1.3)
            fig      = plt.figure(figsize=figsize)
            gspec    = fig.add_gridspec(3, 2, wspace=0.03, hspace=0.03, height_ratios=[1, 1, 0.6])
            if leg_ncol is None:
                leg_ncol = 2
            qkX, qkY, qkdX, qkdY = 0.5, 0.2, 0.18, -0.06
            axprof   = fig.add_subplot(gspec[2,1])


        # ax1
        if pole_globe:
            _lat0, _lon0 = find_extent(poles)[:2]
            ax1 = fig.add_subplot(gspec[0,0], projection=ccrs.NearsidePerspective(_lat0, _lon0, 5e6), aspect='auto')
            _pmap_ext, _sharex, _sharey = None, None, None
        else:
            ax1 = fig.add_subplot(gspec[0,0], projection=ccrs.PlateCarree(), aspect='auto')
            _pmap_ext, _sharex, _sharey = 'auto', ax1, ax1

        pmap_ext = kwargs.get('pmap_ext', _pmap_ext)

        # ax2 to ax4, and axleg, axprof
        ax2 = fig.add_subplot(gspec[0,1], sharey=_sharey)
        ax3 = fig.add_subplot(gspec[1,0], sharex=_sharex)
        ax4 = fig.add_subplot(gspec[1,1])
        axleg = fig.add_subplot(gspec[2,0])
        axleg.axis('off')

        # rotation rate shown limit
        ax2.set_xlim(rate_lim[0], rate_lim[1])
        ax3.set_ylim(rate_lim[0], rate_lim[1])

        #---------------------------------
        # axis 1 to 3: pole realizations / markers
        supp_keys = [key for key in poleDict if key.startswith('+')]
        for key in supp_keys:
            pole_supps = poleDict[key][0]

            if poleDict[key][1] is not None:
                pc, ps, pm, pa = poleDict[key][1][:4]
            else:
                pc, ps, pm, pa = 'k', 4, '.', 1.0

            # `pole_supps` is a list of objects with `poleLat`, `poleLon`, and `rotRate` attributes
            lats = np.array([pr.poleLat            for pr in pole_supps])
            lons = np.array([pr.poleLon            for pr in pole_supps])
            rots = np.array([pr.rotRate * MASY2DMY for pr in pole_supps])

            # Create scatter plots
            sc = ax1.scatter(lons, lats, c=pc, s=ps, marker=pm, alpha=pa, rasterized=True)
            sc = ax2.scatter(rots, lats, c=pc, s=ps, marker=pm, alpha=pa, rasterized=True)
            sc = ax3.scatter(lons, rots, c=pc, s=ps, marker=pm, alpha=pa, rasterized=True)

            #-------- REALIZATIONS HISTOGRAM ---------
            if key == '+realizations':
                if poleDict[key][1] is not None:
                    hfc, hec, ha, bins = poleDict[key][1][4:]
                else:
                    hfc, hec, ha, bins = 'gainsboro', 'k', 0.9, 48
                thick = 0.16
                ax_histla = ax2.inset_axes([1-thick, 0, thick, 1   ], sharey=ax2)
                ax_histlo = ax3.inset_axes([0      , 0, 1    ,thick], sharex=ax3)
                ax_histro = ax2.inset_axes([0      , 0, 1    ,thick], sharex=ax2)
                hist_kwargs = dict(bins=bins, histtype='stepfilled', alpha=ha, fc=hfc, ec=hec, lw=1.5, density=True)
                ax_histla.hist(lats, **hist_kwargs, orientation='horizontal')
                ax_histla.set_xlim(ax_histla.get_xlim()[::-1]) # Reverse the x-axis limits
                ax_histlo.hist(lons, **hist_kwargs)
                ax_histro.hist(rots, **hist_kwargs)
                [ax.set_zorder(1) or ax.set_axis_off() for ax in [ax_histla, ax_histlo, ax_histro]]
            #-----------------------------------------


        #---------------------------------
        # axis 1 to 3: pole covariace
        legend = plot_pole_covariance( poles,
                                       names    = names,
                                       colors   = colors,
                                       n_std    = cov_n_std,
                                       extent   = pmap_ext,
                                       radius   = None,
                                       axes     = [ax1,ax2,ax3,axleg],
                                       grids_on = False,
                                       elp_lw    = 2,
                                       elp_alpha = 0.2,
                                       elp_lglw  = 1.6,
                                       axLabels  = [None,None,r'Angular velocity [$^{\circ}$/Ma]'],
                                       leg_ncol  = leg_ncol,
                                       **kwargs,
                                       )[-1]
        if compare:
            key1, key2 = compare[:2]
            geod84 = Geod(ellps="WGS84")
            dist = geod84.inv(poleDict[key1][0].poleLon, poleDict[key1][0].poleLat,
                              poleDict[key2][0].poleLon, poleDict[key2][0].poleLat)[2]
            show_str = fr'|$\Delta D$|$=${dist*1e-3:.1f} km'
            ax1.annotate(show_str, xy=(0.14, 0.90), xycoords='axes fraction', fontsize=14, annotation_clip=False)

            drate = poleDict[key1][0].rotRate - poleDict[key2][0].rotRate
            show_str = fr'$\Delta \omega=${ut.as_sci_fmt(drate*MASY2DMY,1)}' + r' $^{\circ}$/Ma'
            ax2.annotate(show_str, xy=(0.14, 0.90), xycoords='axes fraction', fontsize=14, annotation_clip=False)
            ax3.annotate(show_str, xy=(0.14, 0.90), xycoords='axes fraction', fontsize=14, annotation_clip=False)
        #---------------------------------

        # prepare for plate motion plot
        if compare:
            key1, key2 = compare[:2]
        else:
            key1 = names[0]  # usually the InSAR pole
            key2 = names[1]  # usually Altamimi PMM
            for m in [*poleDict]:
                if 'insar' in m.lower():
                    key1 = m
                elif m.startswith('Altamimi') or m.startswith('ITRF2014'):
                    if key1 != m: key2 = m

        if ax4_frame == 'ITRF2014':
            epole_obj = [   poleDict[key1][0],
                            poleDict[key2][0],
                            ]
        else:
            fr = '*'+ax4_frame
            epole_obj = [   poleDict[key1][0]-poleDict[fr][0],
                            poleDict[key2][0]-poleDict[fr][0],
                            ]

        qcolor    = [poleDict[key1][1], poleDict[key2][1]]
        qalpha    = [1, 1]
        qname     = [key1, key2]

        if plateName is not None:
            bndPoly   = read_plate_outline('MORVEL', plateName)
        else:
            bndPoly   = None

        if gpsGDf is None:
            Lats, Lons = None, None
            Ve  , Vn   = None, None
        else:
            _gps = gpsGDf.copy()
            if ax4_frame != 'ITRF2014':
                _gps = gps_ref_pole(_gps, poleDict[fr][0], sign='-', columns=['Lat','Lon','Ve','Vn'], unit='mm')
            Lats, Lons = _gps.Lat, _gps.Lon
            Ve  , Vn   = _gps.Ve*1e-3 , _gps.Vn*1e-3    # m/yr


        #---------------------------------
        # axis 4: predicted plate motion
        if bndPoly is not None:
            ax4 = plot_plate_motion(plate_boundary   = bndPoly    ,
                                    epole_obj        = epole_obj  ,
                                    compare_duel     = compare    ,
                                    compare_stat     = 'diff'     ,
                                    satellite_height = sat_height ,
                                    center_lalo      = center_lalo,
                                    map_style        = map_style  ,
                                    helmert          = helmert    ,
                                    orb              = orb        ,
                                    extent           = extent     ,
                                    Lats             = Lats   ,
                                    Lons             = Lons   ,
                                    Ve               = Ve     ,
                                    Vn               = Vn     ,
                                    qcolor  = qcolor , qalpha    = qalpha , qname  = qname ,
                                    qunit   = qunit  , qscale    = qscale , qwidth = qwidth,
                                    qnum    = qnum   , unit      = unit   ,
                                    ax      = ax4    , font_size = font_size,
                                    pts_ms  = pts_ms ,
                                    grid_lc = grid_lc,
                                    qkX     = qkX,
                                    qkY     = qkY,
                                    qkdX    = qkdX,
                                    qkdY    = qkdY,
                                    )[0]

            # InSAR data bbox
            if bboxs is not None:
                for bbox in bboxs:
                    ax4.fill(bbox[1], bbox[0], ec='gray',
                            fc=tweak_color('grey', luminos=0.4, alpha=0.5),
                            lw=0.8, ls='-', zorder=2.8,
                            transform=ccrs.PlateCarree())

            ax4.set_title(f'{ax4_frame} fixed', x=0.05, y=0.05, loc='left',
                            fontsize=12, bbox=dict(facecolor='ivory', edgecolor='k',
                            linestyle='--', linewidth=1.4, boxstyle='round,pad=0.3'))

            ax4.set_rasterization_zorder(0)

        [spine.set_linewidth(1.4) for ax in [ax1, ax2, ax3, ax4] for spine in ax.spines.values()]
        [ax.set_facecolor('none') for ax in [ax1, ax2, ax3, ax4]] # Makes inset background transparent
        [ax.set_zorder(5)         for ax in [ax1, ax2, ax3, ax4]]
        #---------------------------------


        # annotate panels
        ax1.annotate('a',xy=(0.02, 0.94), xycoords='axes fraction', fontsize=14, weight='bold', annotation_clip=False)
        ax2.annotate('b',xy=(0.02, 0.94), xycoords='axes fraction', fontsize=14, weight='bold', annotation_clip=False)
        ax3.annotate('c',xy=(0.02, 0.94), xycoords='axes fraction', fontsize=14, weight='bold', annotation_clip=False)
        ax4.annotate('d',xy=(0.02, 0.94), xycoords='axes fraction', fontsize=14, weight='bold', annotation_clip=False)

        if style==2:
            axprof = plot_geodesic_profile(poleDict[key2][0], poleDict[key1][0],
                                           gpsGDf          = gpsGDf,
                                           gpsGDf_full     = gpsGDf_full,
                                           extent          = prof_extent,
                                           fname           = False,
                                           colorDict       = None,
                                           ad_nominal_pred = True,
                                           has_GPS         = True,
                                           ax1             = axprof,   # input an ax1 to plot
                                           ax1_label       = 'e',      # input an ax1 to plot
                                           ax2_label       = '',       # input an ax1 to plot
                                           plot_fig2       = False,    # input an ax1 to plot
                                           ylabel          = 'ITRF velocity [mm/yr]',
                                           )
            [spine.set_linewidth(1.4) for ax in [ax1, ax2, ax3, ax4, axprof] for spine in ax.spines.values()]

        if plateName is None:
            ax4.axis('off')

        if savefig:
            parent  = Path(savefig).parent
            stem    = Path(savefig).stem
            suff    = Path(savefig).suffix
            if suff == '': suff = '.pdf'
            fname = parent / (stem+'_'+ax4_frame+suff)
            print(f'save to : {fname}')
            fig.savefig(fname, dpi=400, bbox_inches='tight')

        if show:
            plt.show()
            plt.close(fig)

        arts.append([fig, [ax1, ax2, ax3, ax4]])

    return arts


def calc_insar_variogram(dataDict, out_dir=None, **kwargs):
    """calculate the structural function co/variogram from datasets
    * take input dataset to compute the variogram
    """
    if out_dir is None: out_dir = Path('./')

    # csi imagecovariance args
    verbose     = kwargs.get('verbose'    , True   )
    function    = kwargs.get('function'   , 'exp'  )  # variogram function
    frac        = kwargs.get('frac'       , 0.1    )  # fitting some fraction of pixels
    every       = kwargs.get('every'      , 1.     )  # fitting every n pixels
    distmax     = kwargs.get('distmax'    , 50.    )  # maximum distance
    rampEst     = kwargs.get('rampEst'    , True   )  # quadratic ramp estimation?
    tol         = kwargs.get('tol'        , 1e-10  )  # tolerance
    plotData    = kwargs.get('plotData'   , False  )  # plot input data dots in each variogram?
    savefig     = kwargs.get('savefig'    , False  )  # save it?
    show        = kwargs.get('show'       , False  )  # show it?
    diagonalVar = kwargs.get('diagonalVar', True   )  # substitute the diagonal by the standard deviation of the measurement squared

    # compute variagram from the difference of two plate motion models
    PMM1       = kwargs.get('PMM1'       , None)  # (wx,wy,wz) in mas/yr
    PMM2       = kwargs.get('PMM2'       , None)  # (wx,wy,wz) in mas/yr


    insarData = []
    Cds_dict  = {}

    for i, name in enumerate([*dataDict]):
        vlos, lat, lon, los = ut.get_array4csi(dataDict, name)

        # PMM1 - PMM2
        if PMM1 is not None and PMM2 is not None:
            pole1 = EulerPole(name='1', wx=PMM1[0], wy=PMM1[1], wz=PMM1[2], unit='mas/yr')
            pole2 = EulerPole(name='2', wx=PMM2[0], wy=PMM2[1], wz=PMM2[2], unit='mas/yr')
            pole_diff = pole1 - pole2
            ve, vn, vu = pole_diff.get_velocity_enu(lat, lon, alt=0.0)
            print(' ### use PMM1 - PMM2 discrepency to compute variogram')
            print(los.shape)
            print(ve.shape)
            vlos = (  ve * los[:,0]
                    + vn * los[:,1]
                    + vu * los[:,2])

        # csi insar obj
        insar_data = insar(name=name, lon0=np.mean(lon), lat0=np.mean(lat))

        insar_data.read_from_binary(data=vlos, lon=lon, lat=lat, los=los)
        #insar_data.plot(norm=[-4,4], cmap='RdYlBu_r', figsize=[7,7], markersize=50, plotType='scatter', alpha=1., cbaxis=[0.3,0.0,0.4,0.02])

        # imgcov obj
        data_cov = imagecovariance(name, insar_data, verbose=verbose)
        data_cov.selectedZones = []

        # maskout where there is a lot of coherent signal
        # data_cov.maskOut([[35., 37.5, 35., 39.], [37.5, 39., 37.5, 39.]])
        # can use gauss or exp function, try both and chose the best
        print(f'compute covariance: func {function}; frac {frac}; every {every}; distmax {distmax}; rampEst {rampEst}; tol {tol}')
        data_cov.computeCovariance(function=function, frac=frac, every=every, distmax=distmax, rampEst=rampEst, tol=tol)

        # write to a file
        data_cov.write2file(savedir=out_dir)

        # plot
        data_cov.plot(data='all', plotData=plotData, savefig=savefig, show=show, savedir=out_dir)
        plt.close()

        # Structural parameters
        sill  = data_cov.datasets[name]['Sill']
        sigma = data_cov.datasets[name]['Sigma']
        lam   = data_cov.datasets[name]['Lambda']

        # make Cd
        insar_data.err = np.full(len(insar_data.vel), np.sqrt(sill))
        insar_data.buildCd(sigma=sigma, lam=lam, function=function, diagonalVar=diagonalVar)

        # store it
        Cds_dict[name] = insar_data.Cd.astype(np.float64)
        insarData.append(insar_data)

    return insarData, Cds_dict


def plot_all_variograms(dataDict, insarData=None, out_dir=None, savefig=None):
    """plot a figure with all semi-variograms
    """
    if out_dir is None: out_dir = Path('./')

    ncols = 2
    nrows = len(dataDict)//ncols + len(dataDict) % ncols
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=[10,12], sharex=True, sharey=False,
                            gridspec_kw={'wspace':0.15,'hspace':0.2})
    axs = axs.flatten()
    for i, name in enumerate([*dataDict]):
        if insarData is None:
            vlos, lat, lon, los = ut.get_array4csi(dataDict, name)
            insar_data = insar(name=name, lon0=np.mean(lon), lat0=np.mean(lat))
            insar_data.read_from_binary(data=vlos, lon=lon, lat=lat, los=los)
        else:
            insar_data = insarData[i]

        data_cov = imagecovariance(name, insar_data, verbose=False)
        data_cov.selectedZones = []

        # read from file
        data_cov.read_from_covfile(name, str(out_dir/f'{name}.cov'))
        data_cov.plot(data='all', plotData=False, savefig=False, show=False, savedir=out_dir,
                        markersize=4, axin=axs[i], colors=['lightcoral','orangered','dodgerblue','steelblue'])

        # Structural parameters
        func  = data_cov.datasets[name]['function']
        sill  = data_cov.datasets[name]['Sill']
        sigma = data_cov.datasets[name]['Sigma']
        lam   = data_cov.datasets[name]['Lambda']
        print('structural func:', func, sill, lam)

        # annotate some info
        axs[i].axhspan(0, sill, fc='lightgrey', alpha=0.7, label=r'Sill ($\sigma$)')
        axs[i].axvline(x=3*lam, lw=2, c='k',    ls='--',  label=r'$3\lambda$')
        axs[i].annotate(name, xy=(0.04,0.85), xycoords='axes fraction', fontsize=12,
                        bbox=dict(boxstyle='round', fc='w', ec='k'))
        #axs[i].set_ylim(-1e-7, 7e-7)

    # hide useless axis
    for i in np.arange(len(dataDict), nrows*ncols):
        axs[i].axis('off')

    plt.text(0.06, 0.420, r'Covariance | variance [$m^2$/$yr^2$]', rotation=90, transform=fig.transFigure)
    plt.text(0.45, 0.065, r'Distance [$km$]', transform=fig.transFigure)
    fig.legend(*axs[0].get_legend_handles_labels(), bbox_to_anchor=(0.5, 0.06), loc='upper center', ncol=2, markerscale=3)
    if savefig:
        plt.savefig(savefig, dpi=200, transparent=True, bbox_inches='tight')
    plt.close()

    return


def plot_all_variograms_image(dataDict, insarData=None, out_dir=None, savefig=None):
    """For each datasetset , plot the insar input data for the variogram
    * this is just for checking the variogram inputs are reasonable
    * If you did deramp, then the image plotted here is deramped as well
    """
    if out_dir is None: out_dir = Path('./')

    fig, Axs = plt.subplots(nrows=len(dataDict), ncols=3, figsize=[8,3*len(dataDict)], sharey=True, layout='tight')

    if len(dataDict) == 1:
        Axs = [Axs]

    for i, (axs, name) in enumerate(zip(Axs, [*dataDict])):

        if insarData is None:
            vlos, lat, lon, los = ut.get_array4csi(dataDict, name)
            insar_data = insar(name=name, lon0=np.mean(lon), lat0=np.mean(lat))
            insar_data.read_from_binary(data=vlos, lon=lon, lat=lat, los=los)
        else:
            insar_data = insarData[i]

        data_cov = imagecovariance(name, insar_data, verbose=False)
        data_cov.selectedZones = []


        # read from file
        data_cov.read_from_covfile(name, str(out_dir/f'{name}.cov'))
        if 'Ramp' in data_cov.datasets[name].keys():
            print(data_cov.datasets[name]['Ramp'])
            pars  = np.array(data_cov.datasets[name]['Ramp'], dtype=float)
            a, b, c, u, v, w = pars
            print(a, b, c, u, v, w)
        else:
            a, b, c, u, v, w = 0., 0., 0., 0., 0., 0.

        # 2d arrays
        roi  = dataDict[name][6]
        vlos = ut.get_image_from_arr(insar_data.vel, roi)
        y    = ut.get_image_from_arr(insar_data.y  , roi)
        x    = ut.get_image_from_arr(insar_data.x  , roi)

        # Estimated Orbital Plane: ux2 + vy2 + wxy + ax + by + c
        ramp = u*x*x + v*y*y + w*x*y + a*x + b*y + c
        vmin, vmax = -np.nanmax(np.abs(ramp)), np.nanmax(np.abs(ramp))
        vlos_deramp = vlos - ramp

        # show
        dsets = [vlos, ramp, vlos_deramp]
        titles = ['data','quadratic ramp','deramped']
        for j, (ax, val, title) in enumerate(zip(axs, dsets, titles)):
            ax.set_title(title)
            im = ax.imshow(val*1e3, cmap='RdBu_r', vmin=1e3*vmin, vmax=1e3*vmax)
            plt.colorbar(im, ax=ax)

    if savefig:
        plt.savefig(savefig, dpi=200, transparent=True, bbox_inches='tight')
    plt.close()

    return


def full_Autocorrelation(data, verbose=False):
    '''
    Computes the full autocorrelation of an image.

    Kwargs:
        * data  : shape (length, width), N=length*width

    Returns:
        * None  : shape (N, N)
    '''


    # print
    if verbose:
        print('Computing full autocorrelation for dataset')

    # Flatten the input data
    d = data.flatten()
    Nsamp = len(d)

    # Create the lower triangle indices (excluding diagonal)
    ii, jj = np.tril_indices(Nsamp, -1)

    # Initialize the full symmetric matrix
    C = np.zeros((Nsamp, Nsamp), dtype=float)

    # Set the lower triangle values
    C[ii, jj] = np.abs(d[ii] * d[jj])

    # Set the upper triangle values
    C += C.T

    # Set the diagonal values
    np.fill_diagonal(C, np.abs(d * d))

    return C



def report_avg_covariance(dataDict, Cds_dict, sk=20, txtfile='Cov.txt', savefile=None):
    """
    Input:
        sk -  skip samples to be faster (used in plotting & INVERSION!!)
    """
    header  = '# Sigmas from Cd avg diagonals and Cds diagonal (sill)\n'
    header += '#  Name  Cdt_sig Cds_sig  factor  total_sig\n'
    header += '#'+'-'*50 +'\n'
    header += f'#\t [mm/yr]  [mm/yr] [-]  [mm/yr]'
    print(header)
    with open(txtfile, 'w') as file:
        file.write(header+'\n')
        for i, (name, input_data) in enumerate(dataDict.items()):
            roi = input_data[6]
            std = input_data[1][roi]
            fac = input_data[11]  # un-used here
            cd_sig = 1e3 * np.nanmean(std)  # use the un-scaled std (the original version)
            cp_sig = 1e3 * np.nanmean(np.diag(Cds_dict[name][::sk,::sk])**0.5)
            print_str = f'{name:>8s} {cd_sig:>7.4f} {cp_sig:>7.4f} {fac:>4} {cd_sig+cp_sig:>7.4f}'
            print(print_str)
            file.write(print_str+'\n')

    if savefile:
        # Save the list of arrays to a .pkl file
        with open(savefile, 'wb') as f:
            pickle.dump(Cds_dict, f)

    return



def postfit_outputs(block, dataDict, flag, pole_ref=None, extname=None, dump=True, **kwargs):
    vlim1    = kwargs.get('vlim1'    , (-5,5)  )  # for obs & prediction
    vlim2    = kwargs.get('vlim2'    , (-5,5)  )  # for residual & block1-block2
    demean   = kwargs.get('demean'   , False   )  # for block1-block2
    figsize  = kwargs.get('figsize'  , None    )  # for figure, None for auto
    fontsize = kwargs.get('fontsize' , 10      )  # for figure
    wspace   = kwargs.get('wspace'   , 0.2     )  # for figure
    hspace   = kwargs.get('hspace'   , 0.2     )  # for figure
    aspect   = kwargs.get('aspect'   , 'auto'  )  # for figure

    # common flags
    out_dir   = Path(flag.out_dir)
    projName  = flag.projName
    plateName = flag.plateName

    # create a InSAR_based pole object
    pole = block.create_pole(name=plateName+'_InSAR')
    pole.get_uncertainty(block=block, src='block')
    pole.print_info(outfile=out_dir/f'block{extname}.out')

    # Show the DC shifts estimate
    if block.bias:
        dc_diff, deg_p, dist_p = compare_block_pole(block, pole_ref, [*dataDict], out_dir/f'dcShifts{extname}.pdf')

    # Prediction from ITRF14 pole
    block_ref = block.copy_from_pole(pole_ref)

    # plot dataset fitting
    fig, axs = block.plot_post_fit([*dataDict], block2=block_ref, vlim1=vlim1, vlim2=vlim2, demean=demean,
                                    figsize=figsize, fontsize=fontsize, wspace=wspace, hspace=hspace, aspect=aspect)
    plt.savefig(out_dir/f'dataFit{extname}.pdf', dpi=200, transparent=True, bbox_inches='tight')
    plt.close()

    # dump results to pickle
    if dump == 'whole':
        with open(out_dir/f'dump{extname}.pkl', 'wb') as f:
            pickle.dump([flag, block, dataDict], f)

    elif dump == 'vpred':
        with open(out_dir/f'dump{extname}.pkl', 'wb') as f:
            pickle.dump([flag, block.v_pred_set], f)

    return pole



def run_build_block(dataDict, flag, pole_ref=None, gpsGDf=None, Cds_dict=None, extname=None):
    """build least-squares Euler block model
    Inputs
        dataDict    insar datasets from multi-tracks
        flag        user input options
        gpsGDf      gps geopandas frame
    Outputs
        block       a block model object
    """
    # common flags
    out_dir   = Path(flag.out_dir)
    projName  = flag.projName
    plateName = flag.plateName

    # init a model
    block = None
    block = blockModel(name=projName, print_msg=True)

    ## 1. feed data and geometry, one track at a time
    for ki, (name, input_data) in enumerate(dataDict.items()):
        (vlos, vstd, los_inc_angle, los_azi_angle, lats, lons, roi, ref_los_vec, refyx, reflalo, bbox, std_scl, paths, comp, ramp_rate_err) = input_data

        # **********
        # account for known DC shift(s), put vlos to ITRF frame
        # Convention: vlos + DC = vlos_ITRF  (DC = ITRF plate motion at ref point)
        if name in [*flag.priorDC]:
            # from independent prior like GPS
            if isinstance(flag.priorDC, dict):
                dcin = float(flag.priorDC[name])
                print(f' *** track {name}: add prior input DC {dcin} ***')
            # from PMM like Altamimi pole
            elif pole_ref:
                dcin = ref_los_vec @ pole_ref.get_velocity_enu(reflalo[0], reflalo[1])
                print(f' *** track {name}: add prior pole PMM DC {dcin} ***')

            vlos += dcin
            block.DCs.append(dcin)

        else:
            block.DCs.append(np.nan)
        # **********

        # add dataset to block obj
        block.add_insar(vlos, lats, lons, vstd*std_scl, los_inc_angle, los_azi_angle, roi,
                        name        = name,
                        comp        = comp,
                        ref_los_vec = ref_los_vec,
                        ref_lalo    = reflalo,
                        ref_yx      = refyx,
                        )

    ## 2 - build G
    block.build_G()

    ## 3. account for referenced motion
    if flag.biases:
        # which dataset needs DC bias term
        block.bias_dset = np.where([k not in [*flag.priorDC] for k in [*dataDict]])[0]

        # add DC constant to G matrix
        block.build_bias(fac=1e6, comp=flag.biases)

    if flag.est_ramp:
        # collect all ramp stds
        ramp_stds = np.array([err for input_data in dataDict.values() for err in input_data[14]])
        block.build_ramp(form='xy', ramp_std=ramp_stds)

    block.joint_dG(flag.refG)


    ## 4. augment with GPS stations?
    if (flag.useGPS) and (gpsGDf is not None):
        # joint invert with GPS
        if isinstance(flag.useGPS, (list, tuple)): # specific stations in a list
            block = plugin_GNSS(block, gpsGDf, flag.useGPS, uniErr=1e-4)
        else:
            block = plugin_GNSS(block, gpsGDf,  'all')  # all stations within the geopandas

    # plot G
    fig, ax = block.plot_G(quantile=95)
    fig.savefig(out_dir/f'G{extname}.png', dpi=200, transparent=True, bbox_inches='tight')
    plt.close(fig)

    ## 5. Add Cds from semi-variogram?
    if Cds_dict is not None:
        # to be consistent? scale cov as you scale std_set (optional)
        facs = np.array([dataDict[name][11] for name in [*dataDict]])**2

        # get Cds list
        Cds_set = []
        for k in dataDict.keys():
            if 'en' in k: k = k.split('en')[0]
            Cds_set.append(Cds_dict[k])

        # insert Cds
        block.insert_Cds(Cds_set, scaling_facs=facs)


    return block


def run_inversion(block, flag, extname=None, plotCov=False):
    """
    inversion with multi options
    flag    - user inputs
    sk      - skip samples when plotting
    fig     - figname for saving Cds plot
    """
    # common flags
    out_dir   = Path(flag.out_dir)
    plateName = flag.plateName

    # out fig names
    CovName     = str(plotCov.with_suffix('')) + extname + plotCov.suffix
    CovDiagName = str(plotCov.with_suffix('')) + 'Diag' + extname + plotCov.suffix

    ## covariance matrix
    block.Covariance(errname=flag.errname, plotcov=CovName, plotdiag=CovDiagName)


    ## 1. inversion
    block.invert(errform=flag.errform, diagonalize=flag.diaglzCov, gpu_device=flag.gpuno, save=flag.saveLD, load=flag.loadLD)


    ## 2. replace with a initial model?
    if flag.initMdl:
        # use ITRF2014 Eurasia as my initial model, overwrites the inverted model
        # (See if Makran deviates away from Eura pole)
        print('*** a-priori initial model overwrites the current model params ***')
        pole_init = EulerPole(name=plateName, itrf='2014')
        pole_init.get_uncertainty(src='tableStd')
        block = block.copy_from_pole(pole_init)


    ## 3. prediction & residuals
    block.get_model_pred(model_out=out_dir/f'block{extname}.out')
    block.get_residRMS()


    print('###')
    print(f'#> inversion method: {flag.errname} {flag.errform}; condition number = {block.cond}')
    print('###')

    return block


def run_inversion_Cds(dataDict, block, flag, Cds_set, plot=False):
    ## Insert Cds in the block object
    facs = np.array([dataDict[name][11] for name in [*dataDict]])**2
    block.insert_Cds(Cds_set, scaling_facs=facs) # (remember to scale cov as you scale std_set)

    ## covariance matrix
    ax = block.Covariance(errname=flag.errname, plot=plot)
    if plot:
        plt.savefig(plot, dpi=200, transparent=True, bbox_inches='tight')
        plt.close()

    ## inversion
    # block.joint_dG(flag.refG)
    block.invert(errform=flag.errform, diagonalize=flag.diaglzCov, gpu_device=flag.gpuno, save=True, load=True)
    block.get_model_pred()
    block.get_residRMS()

    print('###')
    print(f'#> inversion method: {flag.errname} {flag.errform}; condition number = {block.cond}')
    print('###')

    return block



##########################################
# high-level display/run functions

def plot_geodesic_profile(pole, poleA,
                          blk             = None,
                          dataDict        = None,
                          gpsGDf          = None,
                          gpsGDf_full     = None,
                          extent          = [15,30,35,60],
                          fname           = 'geodesic_profile.pdf',
                          colorDict       = None,
                          ad_nominal_pred = True,
                          has_GPS         = 'auto',
                          ax1             = None,  # input an ax1 to plot
                          ax1_label       = 'a',   # input an ax1 to plot
                          ax2_label       = 'b',   # input an ax1 to plot
                          plot_fig2       = True,  # input an ax1 to plot
                          ylabel          = 'Horizontal velocity [mm/yr]',
                          ):

    if ax1 is not None: plot_fig2 = False

    if colorDict is None:
        colorDict = {   'a'        :    'crimson',  # ascending  LOS cloud color
                        'd'        : 'dodgerblue',  # descending LOS cloud color
                        'InSAR'    :    '#ffa600',  # insar predicted tengential velocity
                        'Altamimi' :    '#7a0067',  # PMM predicted tengential velocity
                    }

    geod84 = Geod(ellps="WGS84")
    if gpsGDf is not None:
        az12,az21,dist = geod84.inv(gpsGDf.Lon, gpsGDf.Lat, poleA.poleLon*np.ones(len(gpsGDf)),poleA.poleLat*np.ones(len(gpsGDf)))
        proj_gps = ut.project_vector_2d(np.array([gpsGDf.Ve, gpsGDf.Vn]), az12+90.)
        tang_gps = np.linalg.norm(proj_gps, axis=0)
    if gpsGDf_full is not None:
        az12,az21,dist = geod84.inv(gpsGDf_full.Lon, gpsGDf_full.Lat, poleA.poleLon*np.ones(len(gpsGDf_full)),poleA.poleLat*np.ones(len(gpsGDf_full)))
        proj_gps = ut.project_vector_2d(np.array([gpsGDf_full.Ve, gpsGDf_full.Vn]), az12+90.)
        tang_gps_full = np.linalg.norm(proj_gps, axis=0)


    # init the figure
    if ax1 is None and plot_fig2:
        fig, axs = plt.subplots(figsize=[5,7], nrows=3, layout='tight', gridspec_kw={'height_ratios': [1,1,.6]})
        ax1, ax2, ax3 = axs

    ax1.tick_params(axis="both", direction="in", width=1.2)

    # grid across the plate
    Npts = 50
    if extent is None:
        print('no profile extent given, read from block model inputs')
        lat_min = np.min(np.concatenate(blk.lats_set))
        lat_max = np.max(np.concatenate(blk.lats_set))
        lon_min = np.min(np.concatenate(blk.lons_set))
        lon_max = np.max(np.concatenate(blk.lons_set))
        extent  = [lat_min, lat_max, lon_min, lon_max]
    else:
        lat_min, lat_max, lon_min, lon_max = extent # 15,30,35,60 Arabia
    lons    = np.linspace(lon_min, lon_max, Npts)
    lats    = np.linspace(lat_max, lat_min, Npts)

    # distance between grid & pole
    distA_deg, distA_km = ut.haversine_distance((lats,lons), (poleA.poleLat, poleA.poleLon))
    dist_deg , dist_km  = ut.haversine_distance((lats,lons), ( pole.poleLat,  pole.poleLon))

    # model uncertainty (rad/yr)
    m_std  = np.array([ pole.wx_sig,  pole.wy_sig,  pole.wz_sig]) * MAS2RAD
    mA_std = np.array([poleA.wx_sig, poleA.wy_sig, poleA.wz_sig]) * MAS2RAD

    # tangential velocity = omega (rad/yr) * distance_from_pole_axis (m)
    #                     = omega (rad/yr) * R (m) * sin(theta)
    v_tangA     = poleA.rotRate * MAS2RAD * EARTH_RADIUS_A * np.sin(np.deg2rad(distA_deg)) * 1e3
    v_tang      =  pole.rotRate * MAS2RAD * EARTH_RADIUS_A * np.sin(np.deg2rad( dist_deg)) * 1e3
    v_tangA_std = np.linalg.norm(mA_std)  * EARTH_RADIUS_A * np.sin(np.deg2rad(distA_deg)) * 1e3
    v_tang_std  =  np.linalg.norm(m_std)  * EARTH_RADIUS_A * np.sin(np.deg2rad( dist_deg)) * 1e3


    ### Figure 1 ###
    gpsSource  = gpsGDf_full.Source

    gps_sz = 14
    er1 = ax1.fill_between(lons, v_tangA-2*v_tangA_std, v_tangA+2*v_tangA_std, color=colorDict['Altamimi'], alpha=0.1, lw=0)
    er2 = ax1.fill_between(lons,   v_tang-2*v_tang_std,   v_tang+2*v_tang_std, color=colorDict[   'InSAR'], alpha=0.2, lw=0)
    ln1 = ax1.plot(lons, v_tangA, color=colorDict['Altamimi'], ls='--')
    ln2 = ax1.plot(lons,  v_tang, color=colorDict[   'InSAR'], ls='-' )
    sc1 = ax1.scatter(gpsGDf_full.Lon, tang_gps_full, ec='k', fc='w', lw=0.7, marker='s', s=gps_sz, zorder=3)
    if has_GPS:
        print('Show GPS info')
        sc2 = ax1.scatter(gpsGDf.Lon, tang_gps, ec='k', fc='gold', marker='s', s=gps_sz+2, zorder=3)
        if plot_fig2: # show full legend
            handles = [(er1,ln1[0]), (er2,ln2[0]), sc1, sc2]
            labels  = [r'Altamimi$^+$ pred.', 'InSAR pred.', f'GNSS in {gpsSource}', 'GNSS used']
        else:  # show brief legend
            handles = [(er1,ln1[0]), (er2,ln2[0])]
            labels  = [r'Altamimi$^+$ pred.', 'InSAR pred.']
    else:
        if plot_fig2: # show full legend
            handles = [(er1,ln1[0]), (er2,ln2[0]), sc1]
            labels  = [r'Altamimi$^+$ pred.', 'InSAR pred.', f'GNSS in {gpsSource}']
        else:  # show brief legend
            handles = [(er1,ln1[0]), (er2,ln2[0])]
            labels  = [r'Altamimi$^+$ pred.', 'InSAR pred.']
    ax1.legend(handles, labels, fontsize=10, loc='lower right', frameon=False)
    ax1.set_xlim(extent[2], extent[3])
    ax1.set_xlabel(r'Longitude [$^\circ$]')
    ax1.set_ylabel(ylabel)

    rms_pmm = ut.calc_wrms(v_tang - v_tangA)
    show_str = fr'RMS$=${rms_pmm:.2f} mm/yr'
    ax1.annotate(show_str, xy=(0.14, 0.88), xycoords='axes fraction', fontsize=12, annotation_clip=False)
    ax1.annotate(ax1_label,xy=(0.02, 0.90), xycoords='axes fraction', fontsize=14, weight='bold', annotation_clip=False)

    # mute top/right axis bounds for simplicity
    if not plot_fig2:
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)


    if plot_fig2 and blk is not None:
        ### Figure 2 ###
        # LOS DCs for tracks
        # (get the inverted column, set nan bias to 0, since those rows had data shifted already)
        if blk.bias:
            m_bias = np.array(blk.m_bias[:,0])
            m_bias[np.isnan(m_bias)] = 0.
        else:
            m_bias = np.zeros(len(dataDict))

        blkA = blk.copy_from_pole(poleA)
        if blkA.bias:
            pmm_bias = np.array(blkA.m_bias[:,0])
            pmm_bias[np.isnan(pmm_bias)] = 0.
        else:
            pmm_bias = np.zeros(len(dataDict))


        # get LOS clouds
        p_all, v_all = [], []
        for ki, k in enumerate([*dataDict]):
            # insar result
            tmp = blk.Obs_set[ki]
            length, width = tmp.shape

            _lats = blk.lats_set[ki]
            _lons = blk.lons_set[ki]

            plos   = blk.v_pred_set[ki].flatten()
            vlos   = blk.d_set[ki].flatten()
            pmmlos = blkA.v_pred_set[ki].flatten()

            pabs   = plos   +   m_bias[ki]   # insar pole pred
            vabs   = vlos   +   m_bias[ki]   # data
            pmmabs = pmmlos + pmm_bias[ki]   # itrf pmm pred

            refy, refx = blk.ref_yx_set[ki]


            # ======= check name =======
            track_dir = k[0]
            # ==========================

            sc1 = ax2.scatter(_lons,   vabs*1e3, s=5, rasterized=True, color=tweak_color(colorDict[track_dir], 0.3), alpha=0.5)
            sc2 = ax2.scatter(_lons,   pabs*1e3, s=5, rasterized=True, color=tweak_color(colorDict[track_dir], 1  ), alpha=1.0)
            sc3 = ax2.scatter(_lons, pmmabs*1e3, s=1, rasterized=True, color='lightgrey', alpha=0.6)

            if all(x not in (None,np.nan) for x in [refx,refy]):
                ref_idx = refy * width + refx
                valids  = list(np.where(blk.roi_set[ki].flatten())[0])
                ref_loc = valids.index(ref_idx)
                sc4 = ax2.scatter(_lons[ref_loc], pabs[ref_loc]*1e3, s=6, c='k', marker='s')
                print('ref velo & min velo', 1e3*vlos[ref_loc], np.min(np.abs(vlos[ref_loc] * 1e3)))

            p_all.append(pabs)
            v_all.append(vabs)

        p_all = np.array([item for sublist in p_all for item in sublist])
        v_all = np.array([item for sublist in v_all for item in sublist])

        rms_pixel = ut.calc_wrms(p_all-v_all) * 1e3
        show_str = fr'Pixel RMS$=${rms_pixel:.2f} mm/yr'
        ax2.annotate(show_str, xy=(0.14, 0.88), xycoords='axes fraction', fontsize=12, annotation_clip=False)

        # Create custom handles for the first legend
        sc1 = Line2D([0], [0], marker='o', color=colorDict['a'], mec='none', ms=6,   lw=0, alpha=0.4)
        sc2 = Line2D([0], [0], marker='o', color=colorDict['a'], mec='none', ms=6,   lw=0, mew=0)
        sc3 = Line2D([0], [0], marker='o', color='lightgrey'   , mec='grey', ms=5.6, lw=0, mew=0.5)
        sc4 = Line2D([0], [0], marker='o', color=colorDict['d'], mec='none', ms=6,   lw=0, alpha=0.4)
        sc5 = Line2D([0], [0], marker='o', color=colorDict['d'], mec='none', ms=6,   lw=0, mew=0)
        sc6 = Line2D([0], [0], marker='o', color='lightgrey'   , mec='grey', ms=5.6, lw=0, mew=0.5)

        handles = []
        row_labels = []
        if any(key.startswith('a') for key in dataDict):
            handles.append([sc1,sc2,sc3])
            row_labels.append('Asc')
        if any(key.startswith('d') for key in dataDict):
            handles.append([sc4,sc5,sc6])
            row_labels.append('Dsc')

        if any(isinstance(elem, list) for elem in handles):
            handles = [item for sublist in zip(*handles) for item in sublist]

        lg1 = tablelegend(
                        ax             = ax3,
                        handles        = handles,
                        ncol           = 3,
                        col_labels     = ['Data','InSAR pred.',r'Altamimi$^+$ pred.'],
                        row_labels     = row_labels,
                        loc            = 'upper left',
                        bbox_to_anchor = (0,1.05),
                        fontsize       = 10,
                        title_fontsize = 11,
                        #title          = 'Pixel-wise fit',
                        alignment      = 'left',
                        facecolor      = 'whitesmoke',
                        edgecolor      = 'none',
                        )

        ax2.set_xlabel(r'Longitude [$^\circ$]')
        ax2.set_ylabel('SAR velocity [mm/yr]')
        ax2.tick_params(axis="both", direction="in", width=1.2)
        ax3.axis('off')
        ax3.add_artist(lg1)

        # Asc/Dsc nominal model prediction and errors
        if ad_nominal_pred:
            # Avg LOS unit vector for all datasets (sort of at the center of scene...)
            a_los = []
            d_los = []
            for i, (name, ref_los_vec) in enumerate(zip(blk.names, blk.ref_los_vec_set)):
                if any(np.isnan(ref_los_vec)):
                    continue
                if name.lower().startswith('a'):
                    a_los.append(ref_los_vec)
                if name.lower().startswith('d'):
                    d_los.append(ref_los_vec)


            # a psuedo dataset on the plate grids
            plate = None
            plate = blockModel(name='arabia', print_msg=True)

            if a_los != []:
                a_los = np.mean(np.array(a_los), axis=0)
                inc_a, azi_a = ut.get_angles4unit_vector(a_los)
                print(f'avg ASC LOS geometry: {a_los}, {inc_a}, {azi_a}')
                plate.add_insar(data = None,
                                lats = lats,
                                lons = lons,
                                los_inc_angle = inc_a,
                                los_azi_angle = azi_a,
                                name = 'asc'
                                )

            if d_los != []:
                d_los = np.mean(np.array(d_los), axis=0)
                inc_d, azi_d = ut.get_angles4unit_vector(d_los)
                print(f'avg DSC LOS geometry: {d_los}, {inc_d}, {azi_d}')
                plate.add_insar(data = None,
                                lats = lats,
                                lons = lons,
                                los_inc_angle = inc_d,
                                los_azi_angle = azi_d,
                                name = 'dsc'
                                )

            plate.build_G()
            plate.joint_dG(subtract_ref=False)

            # plate locations G for vlos
            G  = np.array(plate.G_all)
            if plate.bias is False:
                G *= EARTH_RADIUS_A

            # pred d_std (ignore the covariance, simply take the std part)
            sig_a = sig_d = sigA_a = sigA_d = None
            d_std  = np.sqrt(G**2 @  m_std**2)  # for inverted model
            dA_std = np.sqrt(G**2 @ mA_std**2)  # for Altamimi model
            if len(a_los) == 3:
                sig_a  =  d_std[:Npts] * 1e3  # mm/yr
                sigA_a = dA_std[:Npts] * 1e3  # mm/yr
                if len(d_los) == 3:
                    sig_d  =  d_std[Npts:] * 1e3  # mm/yr
                    sigA_d = dA_std[Npts:] * 1e3  # mm/yr
            elif len(d_los) == 3:
                sig_d  =  d_std[:Npts] * 1e3  # mm/yr
                sigA_d = dA_std[:Npts] * 1e3  # mm/yr

            # nominal curves
            if len(a_los) == 3:
                a_curve    = (a_los.reshape(1,-1) @ np.array( pole.get_velocity_enu(lats, lons))).flatten() * 1e3
                apmm_curve = (a_los.reshape(1,-1) @ np.array(poleA.get_velocity_enu(lats, lons))).flatten() * 1e3
                ln1 = ax2.plot(lons, apmm_curve, 'k', ls='--')
                ln2 = ax2.plot(lons, a_curve, colorDict['a'], ls='-')

            if len(d_los) == 3:
                d_curve    = (d_los.reshape(1,-1) @ np.array( pole.get_velocity_enu(lats, lons))).flatten() * 1e3
                dpmm_curve = (d_los.reshape(1,-1) @ np.array(poleA.get_velocity_enu(lats, lons))).flatten() * 1e3
                ln1 = ax2.plot(lons, dpmm_curve, 'k', ls='--')
                ln3 = ax2.plot(lons, d_curve, colorDict['d'], ls='-')

            # plot
            hands  = []
            labels = []
            if sig_a is not None:
                er2 = ax2.fill_between(lons, a_curve-2*sig_a, a_curve+2*sig_a, color=colorDict['a'], alpha=0.2, lw=0)
                hands += [(er2,ln2[0])]
                labels += ['InSAR Asc']
            if sig_d is not None:
                er3 = ax2.fill_between(lons, d_curve-2*sig_d, d_curve+2*sig_d, color=colorDict['d'], alpha=0.2, lw=0)
                hands += [(er3,ln3[0])]
                labels += ['InSAR Dsc']
            if sigA_a is not None:
                er1 = ax2.fill_between(lons, apmm_curve-2*sigA_a, apmm_curve+2*sigA_a, color='k', alpha=0.2, lw=0)
                hands += [(er1,ln1[0])]
                labels += [r'Altamimi$^+$']
            if sigA_d is not None:
                er1 = ax2.fill_between(lons, dpmm_curve-2*sigA_d, dpmm_curve+2*sigA_d, color='k', alpha=0.2, lw=0)
                if len(hands) < 3:
                    hands += [(er1,ln1[0])]
                    labels += [r'Altamimi$^+$']


            lg2 = ax3.legend(
                            handles        =  hands,
                            labels         =  labels,
                            loc            = 'lower left',
                            ncols          = 3,
                            bbox_to_anchor = (0,0),
                            title          = 'Nominal LOS prediction',
                            handletextpad  = 0.2,
                            fontsize       = 10,
                            title_fontsize = 11,
                            alignment      = 'left',
                            facecolor      = 'whitesmoke',
                            edgecolor      = 'none',
                            )
        # annotate panels
        ax2.annotate(ax2_label,xy=(0.02, 0.90), xycoords='axes fraction', fontsize=14, weight='bold', annotation_clip=False)
        ax2.set_xlim(extent[2], extent[3])
        ax2.set_ylim(None, None)


    # Save Figure
    if fname:
        plt.savefig(fname, transparent=True, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # output
    if plot_fig2:
        return ax1, ax2
    else:
        return ax1
