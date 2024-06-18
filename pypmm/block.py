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

import sys
import copy
import scipy
import numpy as np
from matplotlib import pyplot as plt

from skimage.transform import downscale_local_mean

# cartopy plots
import pyproj
from shapely import geometry
from cartopy import crs as ccrs, feature as cfeature

# PyPMM modules
from pypmm.euler_pole import EulerPole
from pypmm.plot_utils import plot_imshow
from pypmm.models import MAS2RAD, MASY2DMY
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
        self.names        = []
        self.N_set        = []
        self.dComp_set    = []
        self.subtract_ref = False
        self.bias         = False
        self.bias_comp    = 1

        # design matrices
        self.Gc_set   = []
        self.T_set    = []
        self.L_set    = []
        self.G_set    = []
        self.d_set    = []

        # 1d array like
        self.std_set  = []
        self.lats_set = []
        self.lons_set = []
        self.los_inc_angle_set = []
        self.los_azi_angle_set = []
        self.ref_los_vec_set   = []
        self.ref_lalo_set      = []
        self.ref_yx_set        = []

        # 2D matrix as roi masks
        self.roi_set  = []
        self.Obs_set  = []
        self.Std_set  = []
        self.Lats_set = []
        self.Lons_set = []

        # model params
        self.m_all = 0
        self.e2    = 0
        self.rank  = 0
        self.singv = 0
        self.cond  = 0
        self.Cm    = 0
        self.m_std = 0

        # covariance matrix
        self.Cds_set = []
        self.Cov = []

        # user input priors (nans will be filled with inversion results)
        self.DCs   = []

    def add_insar(self, data, lats, lons, std=None, los_inc_angle=None, los_azi_angle=None, roi=None, name=None, comp='los', **kwargs):
        """Add 2D InSAR data, LOS geometry, roi masking, std, track name
        """
        # add track name if given
        if name is not None: self.names.append(name); vprint(f'add dataset {name}')

        # num of comp. of input data
        self.comp = comp

        # initalize null inputs
        shapes = lats.shape
        if data is None: data = np.full(shapes, np.nan); vprint('No data is supplied, set to nan')
        if roi  is None:  roi = np.full(shapes,   True); vprint('No  roi is supplied, set all to True')
        if std  is None:  std = np.full(shapes,    1.0); vprint('No  std is supplied, set all to unity 1.0')

        # single float
        if isinstance(los_inc_angle, float): los_inc_angle = np.full(shapes, los_inc_angle)
        if isinstance(los_azi_angle, float): los_azi_angle = np.full(shapes, los_azi_angle)

        vprint(f'flatten data {shapes} to a 1D array')
        self.add_data_array(data[roi], lats[roi], lons[roi], std[roi], los_inc_angle[roi], los_azi_angle[roi])
        self.roi_set.append(roi)
        self.Obs_set.append(data)
        self.Std_set.append(std)
        self.Lats_set.append(lats)
        self.Lons_set.append(lons)

        # ref. point los info
        ref_los_vec = kwargs.get('ref_los_vec')
        ref_lalo    = kwargs.get('ref_lalo')
        ref_yx      = kwargs.get('ref_yx')
        self.ref_los_vec_set.append(np.array(ref_los_vec))
        self.ref_lalo_set.append(np.array(ref_lalo))
        self.ref_yx_set.append(np.array(ref_yx))


    def add_gps(self, data, lats, lons, std=None, name=None, comp='en'):
        """Add 2-comp or 3-comp GPS data, std, site names
        data : dim=(N by C) , N=num of stations , C={e,n}
        """
        self.comp = comp

        # add site names if given
        if name is not None: self.names.append(name); vprint(f'add gps sites {name}')

        # add gps data
        self.add_data_array(data, lats, lons, std)


    def add_data_array(self, data, lats, lons, std=None, los_inc_angle=None, los_azi_angle=None):
        # check data dimension
        if   self.comp == 'los': dim_c = 1
        elif self.comp == 'en' : dim_c = 2

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
        for k, N in enumerate(self.N_set):
            los_inc_angle = self.los_inc_angle_set[k]
            los_azi_angle = self.los_azi_angle_set[k]
            L = np.array(ut.get_unit_vector4component_of_interest(los_inc_angle=los_inc_angle,
                                                                  los_az_angle=los_azi_angle)).T
            self.L_set.append(L)
            vprint('built L shape', L.shape)
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
        for k, (G, d, std, roi, refyx) in enumerate(zip(self.G_set,
                                                        self.d_set,
                                                        self.std_set,
                                                        self.roi_set,
                                                        self.ref_yx_set)):
            # reference Gm=d at the reference point
            self.subtract_ref = subtract_ref
            if subtract_ref:
                width = roi.shape[1]
                ref_idx = refyx[0] * width + refyx[1]
                ref_row = ut.get_masked_index(roi.flatten(), ref_idx)
                print(f'putting Gm=d referenced to {refyx}, data={d[ref_row]}')
                G -= G[ref_row, :]
                G   = np.delete(G,   ref_row, axis=0)
                d   = np.delete(d,   ref_row, axis=0)
                std = np.delete(std, ref_row, axis=0)


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


    def add_GNSS2insar(self, lalo, venu, senu):
        # coordinates
        lat, lon = lalo

        # 2-comp GNSS data
        d = np.array(venu).reshape(-1,1)
        s = np.array(senu).reshape(-1,1)

        # cross product to vxyz
        x, y, z = ut.T_llr2xyz(lat, lon)
        C = ut.R_crossProd_xyz(x, y, z)

        # xyz transf to ENU
        T = ut.R_xyz2enu(lat=lat, lon=lon)

        # LOS geometry for ENU comps
        if len(venu) == len(senu) == 2:
            L = np.array([[1,0,0],
                          [0,1,0],
                          ])

        elif len(venu) == len(senu) == 3:
            L = np.array([[1,0,0],
                          [0,1,0],
                          [0,0,1]
                          ])

        # G
        G = L.reshape(-1,3) @ T @ C

        # add to system
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
        for i, (Cds, fac, roi, refyx) in enumerate(zip(Cds_set,
                                                       scaling_facs,
                                                       self.roi_set,
                                                       self.ref_yx_set)):
            if self.subtract_ref:
                width = roi.shape[1]
                ref_idx = refyx[0] * width + refyx[1]
                ref_row = ut.get_masked_index(roi.flatten(), ref_idx)
                Cds = np.delete(Cds, ref_row, axis=0)   # remove refpoint row
                Cds = np.delete(Cds, ref_row, axis=1)   # remove refpoint col
                print(f'putting Gm=d and Covariance referenced to {refyx}')

            self.Cds_set.append(Cds * fac)
            print(f'appended Cd_s for dataset {i+1}/{len(Cds_set)}, scaling fac={fac}')

        print('done~')


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
            self.Cov = self.std_all**2
            pass

        elif errname in ['Cds','Cdts','Cx']:
            if self.subtract_ref: del_row = 1
            self.Cov = []
            n1 = 0
            for i, n in enumerate(self.N_set):
                n -= del_row
                if (i <= len(self.Cds_set)-1) and (len(self.Cds_set[i])==n):
                    Cds = self.Cds_set[i]
                else:
                    Cds = np.zeros((n,n), dtype=np.float32)

                if errname == 'Cds':
                    self.Cov.append(Cds)
                elif errname == 'Cdts':
                    Cdt = np.diag(self.std_all[n1:n1+n].flatten())**2
                    self.Cov.append(Cds + Cdt)
                elif errname == 'Cx':
                    sys.exit('TBA, not supported now...')

                n1 += n

        if plot:
            ax = self.plot_Cov()
            return ax
        else:
            return


    def invert(self, errform='no', diagonalize=False, save=False):
        """invert for model params
        INPUTS:

            errform - use of error model from reading the self.Cov
                        'no'   - no errors
                        'diag' - diag uncorrelated error
                        'full' - full covariance
            diagonalize - operation on the covariance matrix
            sub     - subsample
            save    - save the matrices
        """

        # simple least-squares
        if errform == 'no':
            vprint(' :: Ordinary least-squares, no weights')
            res = ut.weighted_LS(self.G_all, self.d_all)


        # diagonal of covariance
        elif errform == 'diag':
            vprint(' :: only use the diagonals of the covariance')
            std_all = np.sqrt(self.Cov)
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
                vprint('   decompose block-by-block')
                L, D = ut.matrix_diagonalization_blockwise(*self.Cov)

                # takes more memory?
                # vprint('   decompose Cov all at once')
                # L, D = ut.matrix_diagonalization(ut.matrix_block_diagonal(self.Cov))

                # transform G and d to that diagonalized space
                vprint(' :: decorrelate and normalize d and G')
                vprint(f'    L size:{L.nbytes/(1024**2):.3f}MB; D size:{D.nbytes/(1024**2):.3f}MB')
                self.L = L
                self.D = D
                del L, D

                Gt, dt = ut.decorrelate_normalize_Gd(self.L, self.D, self.G_all, self.d_all)
                self.G_tilde = Gt
                self.d_tilde = dt
                del Gt, dt

                vprint(' :: solve weighted least-squares with new d and G')
                res = ut.weighted_LS(self.G_tilde, self.d_tilde)

            # *****************

            else:
                vprint(' Might run into memory/computation issue!')
                res = ut.fullCov_LS(self.G_all, self.d_all, self.Cov)

        self.m_all  = res[0]
        self.Cm     = res[1]
        self.m_std  = res[2]
        self.e2     = res[3]
        self.rank   = res[4]
        self.singv  = res[5]
        self.cond   = res[6]

        vprint('~')
        return


    def print_info(self, outfile=False):
        def _show_content():
            #********************************
            #       * report results *
            print(' m    =\n' , self.m_all)
            print(' e2   =  ' , self.e2   )
            print(' rank =  ' , self.rank )
            print(' singv=  ' , self.singv)
            print(' Cm    =\n', self.Cm   )
            print(' m_std =\n', self.m_std)
            print(' ref_los_vec =\n', np.array(self.ref_los_vec_set))
            #********************************
            return

        # 1) display content
        print()
        _show_content()
        print()

        # 2) save to outfile
        if outfile:
            from contextlib import redirect_stdout
            with open(outfile, 'w') as f:
                with redirect_stdout(f):
                    _show_content()

        return


    def get_model_pred(self, print_model=True, model_out=False, imgInput=True):
        """Get model prediction
        """
        if print_model:
            self.print_info(outfile=model_out)

        bComp = self.bias_comp

        # forward predict from model params
        self.v_pred_all = self.G_all @ self.m_all
        vprint(f'model prediction on all samples: {self.v_pred_all.shape}')

        # get pole & bias params & prediction for each dataset
        self.m = self.m_all[:3]   # the rotation pole vector

        # initialize estimated bias
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
            vprint(f'  set{k+1}: {r0} to {r1} = {N} samples')
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
                    los   = self.G_all[r0, c0:c1].flatten()
                    mbias = self.m_all[c0:c1].flatten()
                    self.m_bias[dset] = np.linalg.norm(los) * mbias

                    # projected los bias --> fill back to DCs
                    self.DCs[dset] = - ( los @ mbias )

        self.DCs = np.array(self.DCs)

        vprint(f'split model prediction into {k+1} sets')
        vprint('~')

        self.V_pred_set = []

        # reshape all prediction back to 2D image if it is InSAR data
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
            if dComp == 1:
                # remove mean for insar data (ref_point)
                vprint('remove median for residual calc')
                Obs  -= np.nanmedian(Obs)
                Pred -= np.nanmedian(Pred)
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
        print(block_new.m_all.shape, block_new.G_all.shape)

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
        return ax


    def plot_Cov(self, sub=20, vmax=0.6, title='covariance'):
        """
        plot the subsampled std from Covariance matrix
        """
        if isinstance(self.Cov, list):
            As = []
            for C in self.Cov:
                _A = C[::sub,::sub]**0.5
                As.append(_A)
            A = ut.matrix_block_diagonal(*As)

        elif isinstance(self.Cov, np.ndarray):
            if len(self.Cov.shape)==2:
                A = self.Cov[::sub,::sub]**0.5
            elif len(self.Cov.shape)==1:
                A = np.diag(self.Cov[::sub]**0.5)

        A[A==0] = np.nan
        fig, ax = plt.subplots()
        im = ax.imshow(A*1e3, vmax=vmax)
        plt.colorbar(im, ax=ax, label=r'$C_{d_s}^{0.5}$ [mm/yr]')
        plt.title(title)
        return ax


    def plot_post_fit(self, plot_tks, block2=None, u_fac=1e3, cmap='RdYlBu_r', clabel='mm/year',
                        vlim1=[None,None], vlim2=[None,None], demean=False,
                        figsize=(10,6), fontsize=12, aspect=None, **kwargs):
        """Parameters:
        *   plot_tks    insar datasets (tracks) to plot                         [str]
        *   block2      another block object to compare with the current block  [block obj]
        *   u_fac       value scaling factor (1e3 for mm)                       [float]
        *   cmap        colormap                                                [str; colorpmap]
        *   clabel      colormap label                                          [str]
        """
        N_show = 4       # {obs, std, est_model_pred, postfit_residual}
        if block2 is not None:
            N_show += 2  # {model_pred, model_diff}
        fig   = plt.figure(figsize=figsize)
        gspec = fig.add_gridspec(nrows=N_show*2, ncols=len(plot_tks)+1, width_ratios=[1]*len(plot_tks)+[0.06], **kwargs)
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
                    va='bottom', ha='left', transform=ax4.transAxes, fontsize=8, zorder=99)

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
                ax6.text(0.1, 0, f'RMS={diffrms:.2f}', va='bottom', ha='left', transform=ax6.transAxes, fontsize=8, zorder=99)
                axs.append([ax1, ax2, ax3, ax4, ax5, ax6])
            else:
                axs.append([ax1, ax2, ax3, ax4])
        axs = np.array(axs).T

        # add reference point symbols
        for ki, k in enumerate(plot_tks):
            refy, refx  = self.ref_yx_set[ki]
            if (refy!=None) and (refx!=None):
                vobs   = u_fac *      self.Obs_set[ki][refy,refx]
                vpred  = u_fac *   self.V_pred_set[ki][refy,refx]
                vstd   = u_fac *      self.Std_set[ki][refy,refx]
                resid  = u_fac *      self.res_set[ki][refy,refx]
                for i, ax in enumerate(axs[:,ki]): # for all rows
                    ax.scatter(refx, refy, s=8, c='k', marker='s')
                axs[0,ki].text(refx, refy, f'{vobs  :.2f}', fontsize=8, zorder=99)
                axs[1,ki].text(refx, refy, f'{vstd  :.2f}', fontsize=8, zorder=99)
                axs[2,ki].text(refx, refy, f'{vpred :.2f}', fontsize=8, zorder=99)
                axs[3,ki].text(refx, refy, f'{resid :.2f}', fontsize=8, zorder=99)
                if block2 is not None:
                    vpred2 = u_fac * block2.V_pred_set[ki][refy,refx]
                    diff   = vpred - vpred2
                    axs[4,ki].text(refx, refy, f'{vpred2:.2f}', fontsize=8, zorder=99)
                    axs[5,ki].text(refx, refy, f'{diff  :.2f}', fontsize=8, zorder=99)

        cax1 = fig.add_subplot(gspec[1:2,-1])
        cax2 = fig.add_subplot(gspec[3:4,-1])
        cax3 = fig.add_subplot(gspec[5:6,-1])
        cax4 = fig.add_subplot(gspec[7:8,-1])
        fig.colorbar(im1, cax=cax1, label=clabel)
        fig.colorbar(im2, cax=cax2, label=clabel)
        fig.colorbar(im3, cax=cax3, label=clabel)
        fig.colorbar(im4, cax=cax4, label=clabel)

        rchi2 = chi2_tot / (nd-len(self.m_all))  # reduced chi-squares (chi2 per degree of freedom)
        fits  = fr' ($\chi_{{\nu}}^2$={rchi2:.2f})'
        plt.text(-0.2, 0.5, 'Obs.',          va='center', rotation=90, transform=axs[0,0].transAxes)
        plt.text(-0.2, 0.5, 'Std.',          va='center', rotation=90, transform=axs[1,0].transAxes)
        plt.text(-0.2, 0.5, 'Est. rotation', va='center', rotation=90, transform=axs[2,0].transAxes)
        plt.text(-0.2, 0.5, 'Residual'+fits, va='center', rotation=90, transform=axs[3,0].transAxes)

        if block2 is not None:
            cax5 = fig.add_subplot(gspec[9:10,-1])
            cax6 = fig.add_subplot(gspec[11:12,-1])
            fig.colorbar(im5, cax=cax5, label=clabel)
            fig.colorbar(im6, cax=cax6, label=clabel)
            plt.text(-0.2, 0.5, 'PMM rotation'  , va='center', rotation=90, transform=axs[4,0].transAxes)
            plt.text(-0.2, 0.5, 'Model discrep.', va='center', rotation=90, transform=axs[5,0].transAxes)
        return fig, axs
