""" ---------------
    a PyPMM module
    ---------------

    This module contains several numpy/skimage-based tools.
    It depends on PyPMM EulerPole PyPMM class,
    also pypmm.utils, pypmm.plot_utils, pypmm.models.

    recommend usage:
        from pypmm import utils as ut

    author: Yuan-Kai Liu  2023-2024
"""

import os
import sys
import copy
import pickle, glob
from pathlib import Path

import scipy
import numpy as np
from matplotlib import pyplot as plt

# cartopy plots
import pyproj
from shapely import geometry
from cartopy import crs as ccrs, feature as cfeature

# PyPMM modules
from pypmm.euler_pole import EulerPole
from pypmm.plot_utils import plot_imshow
from pypmm.models import MAS2RAD, MASY2DMY, EARTH_RADIUS_A
from pypmm import utils as ut

plt.rcParams.update({'font.size'  : 14,
                     'font.family':'Helvetica',
                     'figure.dpi' : 100,
                    })

global vprint
vprint = print


###########################################
#             related stuff
###########################################

def DC_from_block(block, bComp=1, priorDC=True):
    """
    DCs based on the block biases (they are equivalent if parameterized, otherwise nan)
    * Convention: vlos + DC = vlos_ITRF
                  vlos      = vlos_ITRF - DC
                  d         = G  m
    * bias should be inside your block.m_bias if parameterized
    * otherwise, you get the default nans
    * priorDC=True, replace nans with priorDC
    """
    # get DC from model params
    if bComp in [1, 'los']:
        dc  = block.m_bias.flatten()
        std = block.m_std[3:].flatten()
    elif bComp in [3, 'enu']:
        dc  = block.m_bias.reshape(-1,3)
        std = block.m_std[3:].reshape(-1,3)

    # initialize std
    dc_std = np.full_like(dc, np.nan)
    dc_std[block.bias_dset] = std

    # put back G scaling on std only
    dc_std *= block.bias_fac

    # replace nans with priorDC
    if priorDC:
        if block.DCs.size == block.m_bias.size:
            dc = np.array(block.DCs)
        else:
            sys.exit('block.DCs.size != block.m_bias.size')

    return dc, dc_std


def DC_from_pole(pole, lat, lon, los_vec, bComp=1):
    """
    DCs forward predicted by pole
    * Convention: vlos + DC = vlos_ITRF
                  DC        = vlos_ITRF - vlos
                  DC        = ITRF plate motion at reference point (where vlos=0)
    * need latlon and los vector at the ref point
    """
    enu = np.array( pole.get_velocity_enu(lat, lon) )  # 3 by N

    if bComp in [1, 'los']:
        dc      = np.full(len(lat), np.nan)
        dc_std  = np.full(len(lat), 0.)
        dc      = np.diag(los_vec @ enu)  # dot product = diag(Nx3 @ 3xN)

    elif bComp in [3, 'enu']:
        dc      = np.full([len(lat),3], np.nan)
        dc_std  = np.full([len(lat),3], 0.)
        dc[:,:] = enu.T

    # TODO: assume pmm_pole has zero uncertainty. we can propagate that later

    return dc, dc_std


###########################################
### Class: Rotation linear operators    ###
###########################################
# Refernece:
#   Meade, B. J., & Loveless, J. P. (2009). Block Modeling with Connected Fault-Network Geometries and a Linear Elastic Coupling Estimator in Spherical Coordinates.
#       BSSA, 99(6), 3124–3139. https://doi.org/10.1785/0120090088

class blockModel:

    def __init__(self, name=None, print_msg=True):
        global vprint
        vprint = print if print_msg else lambda *args, **kwargs: None

        # basics
        self.names        = []          # name of dataset
        self.N_set        = []          # number of samples
        self.Comp_set     = []          # 'en','los','azi','azimuth','rg','range'
        self.dComp_set    = []          #   -> number of components
        self.subtract_ref = False       # reference the G matrix at ref point
        self.bias         = False       # estimate the bias term (DC shift)
        self.bias_comp    = 1           # number of bias components

        # least-squares matrices
        self.Gc_set   = []              # cross-product matrix
        self.T_set    = []              # XYZ to ENU transform matrix
        self.L_set    = []              # ENU to LOS projection matrix
        self.G_set    = []              # Combine the above three
        self.d_set    = []              # dataset 1-D array

        # 1d array like
        self.std_set  = []              # data standard dev 1-D array
        self.lats_set = []              # latitude of samples
        self.lons_set = []              # longitude of samples
        self.los_inc_angle_set = []     # incidence angle
        self.los_azi_angle_set = []     # azimuth angle
        self.ref_los_vec_set   = []     # ref point LOS unit vector
        self.ref_lalo_set      = []     # ref point lat and lon
        self.ref_yx_set        = []     # ref point x and y
        self.ref_row_set       = []     # ref point row in its G matrix

        # 2D matrix as roi masks
        self.roi_set  = []              # roi mask
        self.Obs_set  = []              # data as image
        self.Std_set  = []              # std as image
        self.Lats_set = []              # lat as image
        self.Lons_set = []              # lon as image

        # model params
        self.m_all = 0                  # model params
        self.e2    = 0                  # ||Gm-d||2^2
        self.rank  = 0                  # rank of G
        self.singv = 0                  # singular values of G
        self.cond  = 0                  # condition number of G
        self.Cm    = 0                  # covariance matrix of model params
        self.m_std = 0                  # std of model params

        # covariance matrix
        self.Cds_set = []               # data spatial covariance matrix
        self.Cx = []                    # total covariance for inversion

        # user input priors (nans will be filled with inversion results)
        self.DCs   = []

    def add_insar(self, data, lats, lons, std=None, los_inc_angle=None, los_azi_angle=None, roi=None, name=None, comp='los', **kwargs):
        """Add 2D InSAR data, LOS geometry, roi masking, std, track name
        """
        # add track name if given
        if name is not None: self.names.append(name); vprint(f'add dataset {name}')

        # initalize null inputs
        shapes = lats.shape
        if data is None: data = np.full(shapes, np.nan); vprint('No data is defined, set to nan')
        if roi  is None:  roi = np.full(shapes,   True); vprint('No  roi is defined, set all to True')
        if std  is None:  std = np.full(shapes,    1.0); vprint('No  std is defined, set all to unity 1.0')

        # single float
        if isinstance(los_inc_angle, float): los_inc_angle = np.full(shapes, los_inc_angle)
        if isinstance(los_azi_angle, float): los_azi_angle = np.full(shapes, los_azi_angle)

        vprint(f'flatten data {shapes} to a 1D array')
        self.add_data_array(data[roi], lats[roi], lons[roi], std[roi], los_inc_angle[roi], los_azi_angle[roi], comp=comp)
        self.roi_set.append(roi)
        self.Obs_set.append(data)
        self.Std_set.append(std)
        self.Lats_set.append(lats)
        self.Lons_set.append(lons)

        # ref. point los info
        ref_yx      = kwargs.get('ref_yx')      if kwargs.get('ref_yx')      is not None else (np.nan, np.nan)
        ref_lalo    = kwargs.get('ref_lalo')    if kwargs.get('ref_lalo')    is not None else (np.nan, np.nan)
        ref_los_vec = kwargs.get('ref_los_vec') if kwargs.get('ref_los_vec') is not None else (np.nan, np.nan, np.nan)
        self.ref_yx_set.append(ref_yx)
        self.ref_lalo_set.append(ref_lalo)
        self.ref_los_vec_set.append(ref_los_vec)
        if ref_yx==(np.nan, np.nan) or len(shapes)==1:
            self.ref_row_set.append(np.nan)
        else:
            ref_idx = ref_yx[0]*shapes[1] + ref_yx[1]
            ref_row = ut.get_masked_index(roi.flatten(), ref_idx)
            self.ref_row_set.append(ref_row)


    def add_gps(self, data, lats, lons, std=None, name=None, comp='en'):
        """Add 2-comp or 3-comp GPS data, std, site names
        data : dim=(N by C) , N=num of stations , C={e,n}
        """

        # add site names if given
        if name is not None: self.names.append(name); vprint(f'add gps sites {name}')

        # add gps data
        self.add_data_array(data, lats, lons, std, comp=comp)


    def add_data_array(self, data, lats, lons, std=None, los_inc_angle=None, los_azi_angle=None, comp='los'):
        # check data dimension
        if   comp in ['los','azi','azimuth','rg','range']: dim_c = 1
        elif comp == 'en' : dim_c = 2
        vprint(f' data component type: {comp}')
        self.Comp_set.append(comp)

        # make sure dim = (N by C)
        data = data.reshape(-1, dim_c)
        std  =  std.reshape(-1, dim_c)

        # num of sites (e.g. num of gps sites or insar pixels)
        dim_s = len(data)

        # num of total samples (e.g. one gps site has 2 or 3 samples)
        dim_n = int(dim_s * dim_c)

        # check std
        if std is None:  std = np.full(dim_n, 1.0);   vprint('No std specified, set all as 1.0')

        # append 1D data to whole set
        self.N_set.append(dim_n)
        self.dComp_set.append(dim_c)
        self.d_set.append(data)
        self.std_set.append(std)
        self.lats_set.append(lats)
        self.lons_set.append(lons)

        # append 1D geom to whole set
        if all(_los is not None for _los in [los_inc_angle, los_azi_angle]):
            self.los_inc_angle_set.append(los_inc_angle)
            self.los_azi_angle_set.append(los_azi_angle)

        vprint(f'added data ({dim_s} sites, {dim_c} components) = {dim_n} samples')
        vprint('~')


    def build_Gc(self):
        # rotation cross product: v_xyz = Ω x r = Gc @ Ω
        # Reference: Equation (2) in Meade & Loveless 2009
        for k, (N, dComp) in enumerate(zip(self.N_set, self.dComp_set)):
            Gc   = np.full((N//dComp, 3, 3), 0.)
            lats = self.lats_set[k]
            lons = self.lons_set[k]
            x, y, z = ut.T_llr2xyz(lats, lons)
            Gc = ut.R_crossProd_xyz(x, y, z)
            self.Gc_set.append(Gc)
            vprint('built Gc shape', Gc.shape)
        vprint('~')


    def build_T(self):
        # cart2enu: v_enu = T @ v_xyz
        # where T rotates the cartesian coord to local ENU coord
        for k, (N, dComp) in enumerate(zip(self.N_set, self.dComp_set)):
            T    = np.full((N//dComp, 3, 3), 0.)
            lats = self.lats_set[k]
            lons = self.lons_set[k]
            T = ut.R_xyz2enu(lat=lats, lon=lons)
            self.T_set.append(T)
            vprint('built T shape', T.shape)
        vprint('~')


    def build_L(self):
        # enu2los: v_los = L @ v_enu
        # where L is the line-of-sight projection operation
        # can also be other projections
        for k, (N, comp) in enumerate(zip(self.N_set, self.Comp_set)):
            los_inc_angle = self.los_inc_angle_set[k]
            los_azi_angle = self.los_azi_angle_set[k]

            if comp.lower().startswith('az'):
                comp = 'en2az'
            elif comp.lower().startswith('los'):
                comp = 'enu2los'
            else:
                sys.exit(f'cannot recognize the component string: {comp}')

            L = np.array(ut.get_unit_vector4component_of_interest(los_inc_angle = los_inc_angle,
                                                                  los_az_angle  = los_azi_angle,
                                                                  comp          = comp)).T
            self.L_set.append(L)
            vprint(f'built L shape {L.shape} {comp}')
        vprint('~')


    def build_G(self):
        # The overall forward G matrix: v_los = G @ Ω = (L @ T @ Gc) @ Ω
        for k, (N, dComp) in enumerate(zip(self.N_set, self.dComp_set)):
            T  = self.T_set[k]
            Gc = self.Gc_set[k]
            G  = np.full((N,3), 0.)

            # G will proj v_enu --> v_los
            if dComp==1:
                L  = self.L_set[k]
                for i in range(N):
                    G[i,:] = L[i].reshape(-1,3) @ T[i] @ Gc[i]

            # G will only take v_en from v_enu (vstack G for each one of the N)
            else:
                for i in range(N//dComp):
                    G[i*dComp:(i+1)*dComp, :] = (T[i] @ Gc[i])[:dComp]

            # normalize the G matrix with radius
            if not self.bias:
                self.Gnorm = EARTH_RADIUS_A
                G /= self.Gnorm

            self.G_set.append(G)
            vprint('built G shape', G.shape)
        vprint('~')


    def build_bias(self, fac=1e6, comp='los'):
        """ add one column for est. of a constant offset parameter
        Convention:
                 vlos + DC = vlos_ITRF  --- (1) insar data need DC to adjust to ITRF
                                                DC is simply the ITRF plate motion at insar ref. point
            vlos_ITRF - DC = vlos       --- (2) this is our Gm = d  (-DC so we need negative unity in G)
                                                DC is estimated as the bias term(s) in model params
        Inputs
        fac     : scaling factor for the bias term unity in G
                  nominally should be 1.0,
                  but compare to other entries in G (~1e6), 1 is a super small number
                  so have to use a big number factor here to avoid rank deficiency
        comp    : bias compoenent vector
                  'los'  -  a single offset in vlos
                  'enu'  -  bias projection in e, n, u, respectively (e.g., los_unit_vec at ref_point)
        """
        self.bias      = True
        self.bias_fac  = fac
        K = len(self.N_set)  # number of independent datasets

        if (comp=='los') or (comp==1):
            self.bias_comp = 1
            for k, (N, G) in enumerate(zip(self.N_set, self.G_set)):
                for col in range(K):
                    if col in self.bias_dset:
                        if col == k:
                                # should be negative unity due to above Convention!
                                G = np.hstack([G, np.full((N, 1), -1.*fac)])
                        else:
                            G = np.hstack([G, np.full((N, 1), 0.)])
                self.G_set[k] = np.array(G)
                vprint(f'G added {comp} comp. bias, scaled {fac}, shape {G.shape}')

        elif (comp=='enu') or (comp==3):
            self.bias_comp = 3
            for k, (N, G, ref_los_vec) in enumerate(zip(self.N_set, self.G_set, self.ref_los_vec_set)):
                for col in range(K):
                    if col in self.bias_dset:
                        if col == k:
                                # should be negative unity due to above Convention!
                                G = np.hstack([G, np.full((N, 3), -1.*fac*ref_los_vec)])
                        else:
                            G = np.hstack([G, np.full((N, 3), 0.)])
                self.G_set[k] = np.array(G)
                vprint(f'G added {comp} comp. biases with vector {ref_los_vec}, scaled {fac}, shape {G.shape}')
        else:
            sys.exit('wrong input {comp}')

        vprint('~')


    def stack_data_operators(self, subtract_ref=False):
        """Stack G, d, and std arrays vertically (several datasets)
        + can subtract out the reference row from the rest of the rows [default=False]
          if False, you need to account for the bias term in model params
          if True,  you might not need the bias term (?)

        + can do this step several times if you modify the dataset
          it will just overwrite the *_all matrices
        """
        self.subtract_ref = subtract_ref

        for k, (G, d, std) in enumerate(zip(self.G_set,
                                            self.d_set,
                                            self.std_set)):
            try:
                vprint(f'reference G matrix: {subtract_ref}  REF_YX: {self.ref_yx_set[k]}')
            except:
                vprint('no reference info found')

            # reference Gm=d at the reference point
            if (subtract_ref) and (self.ref_yx_set[k]!=(np.nan,np.nan)):
                refyx   = self.ref_yx_set[k]
                ref_row = self.ref_row_set[k]
                G  -= G[ref_row, :]
                G   = np.delete(G,   ref_row, axis=0)
                d   = np.delete(d,   ref_row, axis=0)
                std = np.delete(std, ref_row, axis=0)

            # stack the matrices
            if k == 0:
                G_all   = np.array(G)
                d_all   = np.array(d.reshape(-1,1))
                std_all = np.array(std.reshape(-1,1))
            else:
                G_all   = np.vstack([G_all  , G])
                d_all   = np.vstack([d_all  , d.reshape(-1,1)])
                std_all = np.vstack([std_all, std.reshape(-1,1)])

        self.G_all   = np.array(G_all)
        self.d_all   = np.array(d_all)
        self.std_all = np.array(std_all)
        vprint('overall G shape'  , G_all.shape)
        vprint('overall d shape'  , d_all.shape)
        vprint('overall std shape', std_all.shape)
        vprint('~')


    def add_gps2insar(self, lat, lon, ve, vn, vu=None, se=None, sn=None, su=None, name='gps'):
        # add site names if given
        if name is not None: self.names.append(name); vprint(f'add gps sites {name}')

        N = len(lat)

        if vu is None:
            dComp = 2
            comp = 'en'
            if se is None: se = np.zeros_like(ve)
            if sn is None: sn = np.zeros_like(vn)
            d = np.column_stack([ve, vn]).reshape(-1,1)
            s = np.column_stack([se, sn]).reshape(-1,1)

            L = np.array([[1,0,0],   # dummy LOS projection vector
                          [0,1,0],
                          ])
        else:
            dComp = 3
            comp = 'enu'
            if se is None: se = np.zeros_like(ve)
            if sn is None: sn = np.zeros_like(vn)
            if su is None: su = np.zeros_like(vu)
            d = np.column_stack([ve, vn, vu]).reshape(-1,1)
            s = np.column_stack([se, sn, su]).reshape(-1,1)

            L = np.array([[1,0,0],   # dummy LOS projection vector
                          [0,1,0],
                          [0,0,1]
                          ])

        L = np.stack([L]*N)

        # cross product to vxyz
        x, y, z = ut.T_llr2xyz(lat, lon)
        C = ut.R_crossProd_xyz(x, y, z)

        # xyz transf to ENU
        T = ut.R_xyz2enu(lat=lat, lon=lon)


        # G matrix for gps
        G = L @ T @ C

        # reshape the en components to 1d
        G = G.reshape(-1,3)

        # normalize the G matrix with radius
        if not self.bias: G /= self.Gnorm

        # add to dataset
        self.N_set.append(N)
        self.Comp_set.append(comp)
        self.dComp_set.append(dComp)
        self.d_set.append(d.flatten())
        self.std_set.append(s.flatten())
        self.lats_set.append(lat)
        self.lons_set.append(lon)
        self.los_inc_angle_set.append(None)
        self.los_azi_angle_set.append(None)

        # add to the big system
        nbias = self.G_all.shape[1] - G.shape[1]
        G = np.hstack([G, np.zeros([G.shape[0], nbias])])
        self.G_all   = np.vstack([self.G_all,   G])
        self.d_all   = np.vstack([self.d_all,   d])
        self.std_all = np.vstack([self.std_all, s])
        vprint(f'augment GNSS site at lat/lon = {lat}/{lon}')
        vprint('overall G shape'  , self.G_all.shape)
        vprint('overall d shape'  , self.d_all.shape)
        vprint('overall std shape', self.std_all.shape)
        vprint('~')


    def insert_Cds(self, Cds_set, scaling_facs=None, save=False):
        """
        input a spatial covariance matrix dataset
        """
        # insert Cd_s from your input Cds_set
        if scaling_facs is None: scaling_facs = np.ones(len(Cds_set))
        for i, (Cds, fac, roi, refyx, ref_row) in enumerate(zip(Cds_set,
                                                                scaling_facs,
                                                                self.roi_set,
                                                                self.ref_yx_set,
                                                                self.ref_row_set)):
            print(f'reference Cov matrix: {self.subtract_ref}  REF_YX: {refyx}')

            if (self.subtract_ref) and (refyx!=(np.nan,np.nan)):
                Cds = np.delete(Cds, ref_row, axis=0)   # remove refpoint row
                Cds = np.delete(Cds, ref_row, axis=1)   # remove refpoint col

            self.Cds_set.append(Cds * fac)
            print(f'appended Cd_s for dataset {i+1}/{len(Cds_set)}, scaling fac={fac}')

        print('done~')



    def insert_Cps(self, mc_dir, m0=None, subset=None, savefile=False):
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from rotation import read_pole_files, full_Autocorrelation


        print(f'Read model from realizations under : {mc_dir}')
        path = Path(mc_dir)
        in_files = sorted(glob.glob(str(path / 'ref*' / 'block.out')))
        ms = read_pole_files(in_files, error='propagate')[6]

        # sample mean model
        if m0 is not None:
            print(f'set sample prior model as: {m0}')
        else:
            m0 = np.mean(ms, axis=0)
            print(f'compute sample mean model as: {m0}')

        # allocate Cp for each dataset
        Cp_set = []
        for i, dd in enumerate(self.d_set):
            Nsamp = len(dd)
            Cp = np.full((Nsamp-1, Nsamp-1), 0.0)
            Cp_set.append(Cp)
        print(f'allocate Cp for dataset: {len(Cp_set)}')


        # subset realizations
        if subset is not None:
            ms       =       ms[subset[0]:subset[1]]
            in_files = in_files[subset[0]:subset[1]]



        # realization #
        L = 0
        for l, (mi, file) in enumerate(zip(ms, in_files)):

            # Read G from realization i
            G_path = Path(file).parent
            with open(G_path/'G.pkl', 'rb') as f:
                G_reali = pickle.load(f)

            # skip realization if G has missing samples
            print(f'realization #{l}, N={len(G_reali)}')
            L += 1


            # get the original input G matrix (assume order is identical)
            if self.subtract_ref:
                ref_rows  = np.array(self.ref_row_set)
                if len(self.N_set) > 1:
                    ref_rows += np.cumsum(np.insert(np.array(self.N_set[:len(ref_rows)-1]), 0, 0))
                ref_rows  = ref_rows[~np.isnan(ref_rows)].astype(int)
                G_reali = np.insert(G_reali, ref_rows, [0.,0.,0.], axis=0)
            else:
                G_reali = G_reali


            # each dataset, calc Cp indepentend from other datasets #
            idx0 = 0
            for i, (dd, G0, ref_row) in enumerate(zip(self.d_set,
                                                    self.G_set,
                                                    self.ref_row_set)):

                if len(dd) == len(G0):
                    pass
                else:
                    print('Sizes of data and G not matiching. Something wrong!')

                # G_reali for single dataset
                Nsamp = len(dd)

                Gi = G_reali[idx0:idx0+Nsamp]
                idx0 += Nsamp

                print(' ', l, i, len(dd), len(G0), len(Gi))

                # scale G back to original order of magnitude
                Gi = np.array(Gi) * self.Gnorm
                G0 = np.array(G0) * self.Gnorm

                # take out ref_row
                Gi -= Gi[ref_row]
                G0 -= G0[ref_row]
                Gi = np.delete(Gi, ref_row, axis=0)
                G0 = np.delete(G0, ref_row, axis=0)

                # sample prediction
                di = Gi @ mi
                d0 = G0 @ m0

                # deviation from the mean model
                din = di.flatten() - d0.flatten()

                # compute autocorrelation, add to dataset Cp
                Cp_set[i] += full_Autocorrelation(din)


                # clean
                del dd, G0, Gi, d0, di, din


        print(f'Averaing the Cp across {L} realizations')
        for i in np.arange(len(Cp_set)):
            print(f' :: averaging dataset {i+1}')
            Cp_set[i] =  Cp_set[i] / L

        self.Cp_set = Cp_set

        if savefile:
            # Save the list of arrays to a .pkl file
            with open(savefile, 'wb') as f:
                pickle.dump(Cp_set, f)

        return



    def Covariance(self, errname='Cdt', plot=False):
        """build the covariance matrix
        errname   - type of the error model assumption
                Cdt  : 1/sigma from diagonals of the temporal covariance
                        (timeseries scatters at each pixel, i.e., velocityStd in MintPy)
                Cds  : full spatial covariance
                        (structural function on deramped velocity field)
                Cdts : Cd = Cdt + Cds

                Cx   : Cx = Cd  + Cp  (TBA)

        plot   - to plot the cov matrix
        """
        N = self.G_all.shape[0]

        if errname == 'Cdt':
            # basic Cd (Cd temporal from insar and gps)
            self.Cx = self.std_all**2
            pass

        elif errname in ['Cds','Cdts','Cx']:
            if self.subtract_ref:
                del_row = 1
            else:
                del_row = 0

            self.Cx = []
            n1 = 0
            for i, (n, dComp) in enumerate(zip(self.N_set, self.dComp_set)):
                if i <= len(self.Cds_set)-1:
                    Cds = self.Cds_set[i]
                    if len(self.Cds_set[i]) == n-del_row:
                        n -= del_row   # referenced insar data, update n
                else:
                    # ignore spatial covariance when not finding the corresponding one: gps or synthetic sar
                    print('ignore spatial covariance, fill with zeros')
                    n *= dComp
                    if dComp==1 and del_row!=0:
                        # reference the sar data
                        n -= del_row
                    Cds = np.zeros((n,n), dtype=np.float64)

                if errname == 'Cds':
                    self.Cx.append(Cds)

                elif errname == 'Cdts':
                    Cdt = np.diag(self.std_all[n1:n1+n].flatten())**2
                    self.Cx.append(Cdt + Cds)

                elif errname == 'Cx':
                    print('TESTING NOW, be careful!')

                    Cdt = np.diag(self.std_all[n1:n1+n].flatten())**2
                    Cp = self.Cp_set[i]

                    self.Cx.append(Cdt + Cds + Cp)

                n1 += n

        if plot:
            ax = self.plot_Cov()
            return ax
        else:
            return


    def invert(self, errform='no', diagonalize=False, gpu_device=None, save=False, load=False):
        """invert for model params
        INPUTS:

            errform - use of error model from reading the self.Cx
                        'no'   - no errors
                        'diag' - diag uncorrelated error
                        'full' - full covariance
            diagonalize - operation on the covariance matrix
            sub     - subsample
            save    - save the matrices
        """
        timer = ut.Timer()
        timer.start()

        # simple least-squares
        if errform == 'no':
            vprint(' :: Ordinary least-squares, no weights')
            res = ut.weighted_LS(self.G_all, self.d_all)


        # diagonal of covariance
        elif errform == 'diag':
            vprint(' :: only use the diagonals of the covariance')

            if isinstance(self.Cx, list):
                # turn a list of Cx to a full array of Cx
                self.Cx = ut.matrix_block_diagonal(*self.Cx)

            std_all = np.sqrt(self.Cx)

            if std_all.shape[0] == std_all.shape[1]:
                # get the diags of the full covariance
                std_all = np.diag(std_all).reshape(-1,1)

            res = ut.weighted_LS(self.G_all, self.d_all, std_all)


        # full covariance
        elif errform == 'full':
            N = self.G_all.shape[0]
            vprint(f' :: use the full covariance, shape: ({N},{N})')

            # *****************
            if diagonalize:
                vprint(f' :: eigenvalue decomposition on covariance')
                vprint(f'    C = ɸ Λ ɸ^T  --> Λ = ɸ^T C ɸ')

                # for easier computation
                vprint(f'    decompose block-by-block, {len(self.Cx)} blocks')
                L, D = ut.matrix_diagonalization_blockwise(*self.Cx, save=save, load=load, names=self.names)

                # takes more memory?
                # vprint('   decompose Cov all at once')
                # L, D = ut.matrix_diagonalization(ut.matrix_block_diagonal(self.Cx))

                # transform G and d to that diagonalized space
                vprint(' :: decorrelate and normalize d and G')
                vprint(f'    L size: {L.nbytes/(1024**2):.2f} MB; D size: {D.nbytes/(1024**2):.2f} MB')

                Gt, dt = ut.decorrelate_normalize_Gd(L, D, self.G_all, self.d_all)
                self.G_tilde = Gt
                self.d_tilde = dt
                del Gt, dt, L, D

                vprint(' :: solve weighted least-squares with new d and G')
                res = ut.weighted_LS(self.G_tilde, self.d_tilde)

            # *****************

            else:
                vprint('no diagonalization, use full C, block-by-block')
                #C = ut.matrix_block_diagonal(*self.Cx)

                if gpu_device is not None:
                    vprint(f'use gpu {gpu_device}')
                    res = ut.fullCov_cuda_LS_blockwise(self.G_all, self.d_all, self.Cx, gpu_device)
                else:
                    vprint(f'no gpu')
                    res = ut.fullCov_cuda_LS_blockwise(self.G_all, self.d_all, self.Cx)

        self.m_all  = res[0]
        self.Cm     = res[1]
        self.m_std  = res[2]
        self.e2     = res[3]
        self.rank   = res[4]
        self.singv  = res[5]
        self.cond   = res[6]

        # normalize the G matrix with radius, scale back m
        if not self.bias:
            self.m_all /=  self.Gnorm
            self.Cm    /= (self.Gnorm**2)
            self.m_std /=  self.Gnorm

        timer.stop()
        print(timer.readable_elapsed())
        vprint('~')
        return


    def print_info(self, outfile=False, action='w'):
        def _show_content():
            #********************************
            #       * report results *
            print(' m    =\n' , self.m_all     )
            print(' e2   =  ' , self.e2        )
            print(' n_samps=' , len(self.G_all))
            print(' rank =  ' , self.rank      )
            print(' singv=  ' , self.singv     )
            print(' cond =  ' , self.cond      )
            print(' Cm    =\n', self.Cm        )
            print(' m_std =\n', self.m_std     )
            print(' ref_los_vec =\n', self.ref_los_vec_set)
            #********************************
            return

        # 1) display content
        print()
        _show_content()
        print()

        # 2) save to outfile
        if outfile:
            from contextlib import redirect_stdout
            with open(outfile, action) as f:
                with redirect_stdout(f):
                    _show_content()
                f.write('\n')

        return


    def get_model_pred(self, print_model=True, model_out=False, imgInput=True):
        """Get model prediction
        """
        if print_model:
            self.print_info(outfile=model_out)

        bComp = self.bias_comp

        # get the original input G matrix (assume order is identical)
        if self.subtract_ref:
            ref_rows  = np.array(self.ref_row_set)
            if len(self.N_set) > 1:
                ref_rows += np.cumsum(np.insert(np.array(self.N_set[:len(ref_rows)-1]), 0, 0))
            ref_rows  = ref_rows[~np.isnan(ref_rows)].astype(int)
            G_all = np.insert(self.G_all, ref_rows, [0.,0.,0.], axis=0)
        else:
            G_all = self.G_all

        # de-normalize the G matrix with radius
        if not self.bias: G_all = np.array(G_all) * self.Gnorm

        # forward predict from model params
        self.v_pred_all = G_all @ self.m_all
        vprint(f'model prediction on all samples: {self.v_pred_all.shape}')

        # get pole & bias params & prediction for each dataset
        self.m = self.m_all[:3]   # the rotation pole vector

        # initialize estimated bias
        if self.bias:
            # N dsets have N biases, if not estimated, remains nan
            # shape = (N,bComp) = (N,1) for 'los' = (N,3) for 'enu'
            self.m_bias = np.full((len(self.N_set), bComp), np.nan)  # init nan

            # check any input DCs, replace as default
            if len(self.DCs) != len(self.m_bias):
                sys.exit('input DCs length does not match m_bias (datasets) length!')

        # data prediction array in each dset
        self.v_pred_set = []

        for k, (N, dComp) in enumerate(zip(self.N_set, self.dComp_set)):
            r0 = int(np.sum(self.N_set[:k]))       # starting G row
            r1 = int(np.sum(self.N_set[:k+1]))     # ending   G row
            r1 = (r1-r0)*dComp + r0                # considering dComp in N points is actually dComp*N samples
            vprint(f'  set{k+1}: {r0} to {r1} = {N*dComp} samples')
            self.v_pred_set.append(self.v_pred_all[r0:r1].reshape(-1,dComp))

            # get the bias predictions either in {LOS, ENU->LOS}
            # these m_bias has the bias_fac reverted back to physical scale
            if self.bias:
                if k < len(self.bias_dset):
                    dset = int(self.bias_dset[k])           # which dataset

                    # estimated bias params (revert scaling)
                    r0 = int(np.sum(self.N_set[:dset]))     # starting G row
                    c0 = int(3 +  k    * bComp)             # starting G col   int(3 + k)
                    c1 = int(3 + (k+1) * bComp)             # ending   G col   int(3 +  k    * bComp)

                    # raw bias(s)
                    los   = G_all[r0, c0:c1].flatten()
                    mbias = self.m_all[c0:c1].flatten()
                    self.m_bias[dset] = np.linalg.norm(los) * mbias

                    # projected los bias --> fill back to DCs
                    self.DCs[dset] = - ( los @ mbias )

        self.DCs = np.array(self.DCs)

        vprint(f'split model prediction into {k+1} sets')
        vprint('~')

        self.V_pred_set = []

        # reshape all prediction back to 2D image if it is InSAR data
        #  (for residual calc and plotting!)
        if imgInput and len(self.roi_set) > 0:
            # reshape 1D arrays back to 2D matrices as ROI masks
            for k, (roi, vpred) in enumerate(zip(self.roi_set, self.v_pred_set)):
                V_pred      = np.full(roi.shape, np.nan).flatten()
                idx         =    roi.flatten().nonzero()[0]
                V_pred[idx] =  vpred.flatten()
                V_pred      = V_pred.reshape((roi.shape))
                self.V_pred_set.append(V_pred)
            vprint(f'reshape {k+1} sets of prediction into 2D array')

        else:
            for k, vpred in enumerate(self.v_pred_set):
                self.V_pred_set.append(vpred)
        vprint('~')


    def get_residRMS(self, demedian=True):
        self.res_set  = []
        self.rms_set  = []
        self.wrms_set = []
        self.chi2_set = []
        vprint(f'Name   RMS   WRMS   ReducedChi-2')
        for name, Obs, Std, Pred, dComp in zip(self.names, self.Obs_set, self.Std_set, self.V_pred_set, self.dComp_set):
            if dComp==1 and demedian:
                vprint('remove median for residual calc')
                resid = (Obs-np.nanmedian(Obs)) - (Pred-np.nanmedian(Pred))
            else:
                resid = Obs - Pred
            rms   = ut.calc_wrms(resid)
            wrms  = ut.calc_wrms(resid, w=1/Std)
            chi2, achi2 = ut.calc_reduced_chi2(resid, Std)[:2]
            self.res_set.append(resid)
            self.rms_set.append(rms)
            self.wrms_set.append(wrms)
            self.chi2_set.append(chi2)
            vprint(name, rms, wrms, chi2)
        vprint(f'compute and store residual & wrms')
        vprint('~')


    def create_pole(self, name=None, dTx=0, dTy=0, dTz=0):
        omega_cart = self.m.flatten()   # wx, wy, wz in [rad/time]
        pole = EulerPole(name=name,
                         wx=omega_cart[0], wy=omega_cart[1], wz=omega_cart[2], unit='rad/yr',
                         dTx=dTx, dTy=dTy, dTz=dTz)
        vprint(f'create a pole from block\n{pole}')
        return pole


    def copy_from_pole(self, inpole, poleDC=True):
        """Create a new block from an input pole
        + self block object here will be a reference object (for G and data)
        """
        # create model param from pole
        m_make = np.array([inpole.wx, inpole.wy, inpole.wz]) * MAS2RAD

        # copy block
        block_new = copy.deepcopy(self)

        N      = len(block_new.N_set)
        bComp  = int(block_new.bias_comp)

        # have you ever added biases in self.obj?
        if block_new.bias:
            # if so, recreate G with full bias terms
            block_new.G_set = []
            block_new.build_G()
            block_new.bias_dset = np.arange(N)
            block_new.build_bias(fac=1e6, comp=bComp)
            block_new.stack_data_operators()

            # pole forward predict all biases
            if poleDC:
                los_vecs = np.array( block_new.ref_los_vec_set )
                lats     = np.array( block_new.ref_lalo_set )[:,0]
                lons     = np.array( block_new.ref_lalo_set )[:,1]
                dc_pmm, dc_pmm_std = DC_from_pole(inpole, lats, lons, los_vecs, bComp=bComp)
                dc_pmm = dc_pmm.flatten()
                dc_pmm_std = dc_pmm_std.flatten()
                m_make = np.concatenate([m_make, dc_pmm/block_new.bias_fac])

            # no bias (set all DC shift to 0)
            else:
                m_make = np.concatenate([m_make, np.zeros(N * bComp)])


        else:
            # do nothing, use normal G and m_make
            pass

        # create params
        block_new.m_all = m_make.reshape(-1,1)

        # no inversion quality
        block_new.e2    = None
        block_new.rank  = None
        block_new.Cm    = None
        block_new.m_std = None

        # get new model prediction & post-fit residual
        block_new.get_model_pred(print_model=False)
        block_new.get_residRMS()

        print(f'create a block from pole\n{inpole}')
        return block_new


    def plot_G(self, quantile=95):
        """
        plot the G matrix if you like
        """
        A = np.array(self.G_all[:,:])
        q = np.nanpercentile(A, q=quantile)
        fig, ax = plt.subplots()
        im = ax.pcolorfast(A, cmap='coolwarm', vmin=-q, vmax=q)
        ax.invert_yaxis()
        plt.colorbar(im, ax=ax)

        # report unweighted G matrix info
        rank = np.linalg.matrix_rank(A)
        cond = np.linalg.cond(A)
        plt.title(fr'$rank(G)={rank}$, $cond(G)={cond:.2f}$', fontsize=12)
        return fig, ax


    def plot_Cov(self, sub=20, vmax=0.6, title='covariance'):
        """
        plot the subsampled std from Covariance matrix
        """
        if isinstance(self.Cx, list):
            As = []
            for C in self.Cx:
                _A = C[::sub,::sub]**0.5
                As.append(_A)
            A = ut.matrix_block_diagonal(*As)

        elif isinstance(self.Cx, np.ndarray):
            if len(self.Cx.shape)==2:
                A = self.Cx[::sub,::sub]**0.5
            elif len(self.Cx.shape)==1:
                A = np.diag(self.Cx[::sub]**0.5)

        A[A==0] = np.nan
        fig, ax = plt.subplots()
        im = ax.imshow(A*1e3, vmax=vmax)
        plt.colorbar(im, ax=ax, label=r'$C_{d_s}^{0.5}$ [mm/yr]')
        plt.title(title)
        return ax


    def plot_post_fit(self, plot_tks, block2=None, u_fac=1e3, cmap='RdYlBu_r', clabel='mm/year',
                        vlim1=[None,None], vlim2=[None,None], demean=False,
                        figsize=None, fontsize=10, aspect=None, shrink=0.65, **kwargs):
        """Parameters:
        *   plot_tks    insar datasets (tracks) to plot                         [str]
        *   block2      another block object to compare with the current block  [block obj]
        *   u_fac       value scaling factor (1e3 for mm)                       [float]
        *   cmap        colormap                                                [str; colorpmap]
        *   clabel      colormap label                                          [str]
        """
        # num of datasets shown
        N_show = 4       # {obs, std, est_model_pred, postfit_residual}
        if block2 is not None:
            N_show += 2  # {model_pred, model_diff}

        nrows = N_show * 2
        ncols = len(plot_tks)+1
        width_ratios = [1]*len(plot_tks)+[0.06]

        if not figsize:
            figsize = [ncols*0.9, N_show*2]

        fig   = plt.figure(figsize=figsize)
        gspec = fig.add_gridspec(nrows=nrows, ncols=ncols, width_ratios=width_ratios, **kwargs)
        axs   = []

        # make subplots
        chi2_tot = 0
        self.model_diff = []

        for i, k in enumerate(plot_tks):
            ax1 = fig.add_subplot(gspec[0:2,i])
            ax2 = fig.add_subplot(gspec[2:4,i])
            ax3 = fig.add_subplot(gspec[4:6,i])
            ax4 = fig.add_subplot(gspec[6:8,i])

            ki    = self.names.index(k)             # find the index of each dataset

            # get values                            # --- units ---
            nd    = len(self.d_set[ki])
            vobs  = self.Obs_set[ki]    * u_fac     #     mm/yr
            vstd  = self.Std_set[ki]    * u_fac     #     mm/yr
            vpred = self.V_pred_set[ki] * u_fac     #     mm/yr
            resid = self.res_set[ki]    * u_fac     #     mm/yr
            rms   = self.rms_set[ki]    * u_fac     #     mm/yr
            wrms  = self.wrms_set[ki]   * u_fac     #     mm/yr
            chi2  = self.chi2_set[ki]   * 1         # (dimensionless chi-2)

            achi2 = chi2 / nd   # averaged chi-squares per pixel (sample)
            chi2_tot += chi2

            # Determine the maximum absolute value
            max_abs_val = max(abs(np.nanpercentile(vpred, 5)), abs(np.nanpercentile(vpred, 95)))
            if vlim1 == [None,None]: vlim1 = [-max_abs_val, max_abs_val]
            max_abs_val = max(abs(np.nanpercentile(resid, 5)), abs(np.nanpercentile(resid, 95)))
            if vlim2 == [None,None]: vlim2 = [-max_abs_val, max_abs_val]

            # use plot_utils.py module
            im1 = plot_imshow(ax1,  vobs,   cbar=False, cmap=cmap,      vlim=vlim1,  aspect=aspect)[1]
            im2 = plot_imshow(ax2,  vstd,   cbar=False, cmap='magma_r', vlim=[0,1], aspect=aspect)[1]
            im3 = plot_imshow(ax3, vpred,   cbar=False, cmap=cmap,      vlim=vlim1,  aspect=aspect)[1]
            im4 = plot_imshow(ax4, resid,   cbar=False, cmap=cmap,      vlim=vlim2,  aspect=aspect)[1]

            ax1.set_title(k, fontsize=fontsize)
            ax4.text(0.1, 0, f'RMS={rms:.2f}\n'+f'WRMS={wrms:.2f}\n'+fr'$\chi_n^2$={achi2:.2f}',
                    va='bottom', ha='left', transform=ax4.transAxes, fontsize=fontsize*shrink, zorder=99)

            # show Euler pole models comparison = block - block2
            if block2 is not None:
                vpred2 = u_fac * block2.V_pred_set[ki]
                if demean:
                    # demean to ignore the DC shift here!
                    vprint(f'remove median when compares block (med={np.nanmedian(vpred)}) - block2 (med={np.nanmedian(vpred2)})')
                    diff   = (vpred-np.nanmedian(vpred)) - (vpred2-np.nanmedian(vpred2))
                else:
                    diff   = vpred - vpred2

                self.model_diff.append(diff/u_fac)
                diffrms= np.sqrt(np.nanmean(diff**2))

                ax5 = fig.add_subplot(gspec[8:10,i])
                ax6 = fig.add_subplot(gspec[10:12,i])
                im5 = plot_imshow(ax5, vpred2, cbar=False, cmap=cmap,       vlim=vlim1,               aspect=aspect)[1]
                im6 = plot_imshow(ax6, diff,   cbar=False, cmap='coolwarm', vlim=0.2*np.array(vlim2), aspect=aspect)[1]
                ax6.text(0.1, 0, f'RMS={diffrms:.2f}', va='bottom', ha='left', transform=ax6.transAxes, fontsize=fontsize*shrink, zorder=99)
                axs.append([ax1, ax2, ax3, ax4, ax5, ax6])
            else:
                axs.append([ax1, ax2, ax3, ax4])
        axs = np.array(axs).T

        # add reference point symbols
        for ki, k in enumerate(plot_tks):
            refy, refx  = self.ref_yx_set[ki]
            if all(x not in (None,np.nan) for x in [refx,refy]):
                vobs   = u_fac *      self.Obs_set[ki][refy,refx]
                vpred  = u_fac *   self.V_pred_set[ki][refy,refx]
                vstd   = u_fac *      self.Std_set[ki][refy,refx]
                resid  = u_fac *      self.res_set[ki][refy,refx]
                for i, ax in enumerate(axs[:,ki]): # for all rows
                    ax.scatter(refx, refy, s=8, c='k', marker='s')
                axs[0,ki].text(refx, refy, f'{vobs  :.2f}', fontsize=fontsize*shrink, zorder=99)
                axs[1,ki].text(refx, refy, f'{vstd  :.2f}', fontsize=fontsize*shrink, zorder=99)
                axs[2,ki].text(refx, refy, f'{vpred :.2f}', fontsize=fontsize*shrink, zorder=99)
                axs[3,ki].text(refx, refy, f'{resid :.2f}', fontsize=fontsize*shrink, zorder=99)
                if block2 is not None:
                    vpred2 = u_fac * block2.V_pred_set[ki][refy,refx]
                    diff   = vpred - vpred2
                    axs[4,ki].text(refx, refy, f'{vpred2:.2f}', fontsize=fontsize*shrink, zorder=99)
                    axs[5,ki].text(refx, refy, f'{diff  :.2f}', fontsize=fontsize*shrink, zorder=99)

        cax1 = fig.add_subplot(gspec[1:2,-1])
        cax2 = fig.add_subplot(gspec[3:4,-1])
        cax3 = fig.add_subplot(gspec[5:6,-1])
        cax4 = fig.add_subplot(gspec[7:8,-1])
        cbar1 = fig.colorbar(im1, cax=cax1)
        cbar2 = fig.colorbar(im2, cax=cax2)
        cbar3 = fig.colorbar(im3, cax=cax3)
        cbar4 = fig.colorbar(im4, cax=cax4)

        for cbar in [cbar1, cbar2, cbar3, cbar4]:
            cbar.ax.tick_params(labelsize=fontsize*shrink)
            cbar.set_label(label=clabel, size=fontsize*shrink)

        rchi2 = chi2_tot / (nd-len(self.m_all))  # reduced chi-squares (chi2 per degree of freedom)
        fits  = fr' ($\chi_{{\nu}}^2$={rchi2:.2f})'
        plt.text(-0.2, 0.5, 'Obs.',          va='center', fontsize=fontsize, rotation=90, transform=axs[0,0].transAxes)
        plt.text(-0.2, 0.5, 'Std.',          va='center', fontsize=fontsize, rotation=90, transform=axs[1,0].transAxes)
        plt.text(-0.2, 0.5, 'Est. rotation', va='center', fontsize=fontsize, rotation=90, transform=axs[2,0].transAxes)
        plt.text(-0.2, 0.5, 'Residual'+fits, va='center', fontsize=fontsize, rotation=90, transform=axs[3,0].transAxes)

        if block2 is not None:
            cax5 = fig.add_subplot(gspec[9:10,-1])
            cax6 = fig.add_subplot(gspec[11:12,-1])
            cbar5 = fig.colorbar(im5, cax=cax5, label=clabel)
            cbar6 = fig.colorbar(im6, cax=cax6, label=clabel)
            for cbar in [cbar5, cbar6]:
                cbar.ax.tick_params(labelsize=fontsize*shrink)
                cbar.set_label(label=clabel, size=fontsize*shrink)
            plt.text(-0.2, 0.5, 'PMM rotation'  , va='center', fontsize=fontsize, rotation=90, transform=axs[4,0].transAxes)
            plt.text(-0.2, 0.5, 'Model discrep.', va='center', fontsize=fontsize, rotation=90, transform=axs[5,0].transAxes)
        return fig, axs
