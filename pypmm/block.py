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

# still reply on mintpy reading & los_vector utilities
from mintpy.utils import readfile, utils as ut

# shift to pypmm now
from pypmm.euler_pole import (EulerPole,
                              cart2sph, cart2sph_err, sph2cart_err,
                              MAS2RAD, MASY2DMY,
                              )
from pypmm.plot_utils import (read_plate_outline,
                              plot_plate_motion,
                              plot_imshow,
                              )


plt.rcParams.update({'font.size'  : 14,
                     'font.family':'Helvetica',
                     'figure.dpi' : 100,
                    })

EARTH_RADIUS = 6371.0088e3         # the arithmetic mean radius in meters

###################################################
#########         Data reading          ###########
###################################################

def read_data_geom(dtafile, stdfile, geofile, roifile=None, downscale=False, medStd=False):
    vlos = readfile.read(dtafile, datasetName='velocity')[0]
    atr  = readfile.read(dtafile)[1]
    if roifile is not None: roi  = readfile.read(roifile)[0]
    else:                   roi  = np.full(vlos.shape, 1).astype(bool)
    if stdfile is not None: std  = readfile.read(stdfile, datasetName='velocityStd')[0]
    else:                   std  = np.full(vlos.shape, 1.0)

    inc_angle = readfile.read(geofile, datasetName='incidenceAngle')[0]
    azi_angle = readfile.read(geofile, datasetName='azimuthAngle')[0]
    lats = readfile.read(geofile, datasetName='latitude')[0]
    lons = readfile.read(geofile, datasetName='longitude')[0]

    if downscale: # an integer > 1
        roi       = downscale_local_mean(roi      , (downscale, downscale), cval=np.nan)==1
        vlos      = downscale_local_mean(vlos     , (downscale, downscale), cval=np.nan)
        std       = downscale_local_mean(std      , (downscale, downscale), cval=np.nan)
        inc_angle = downscale_local_mean(inc_angle, (downscale, downscale), cval=np.nan)
        azi_angle = downscale_local_mean(azi_angle, (downscale, downscale), cval=np.nan)
        lats      = downscale_local_mean(lats     , (downscale, downscale), cval=np.nan)
        lons      = downscale_local_mean(lons     , (downscale, downscale), cval=np.nan)

    # update roi from any nans
    nan_bool = np.isnan(vlos)+np.isnan(std)+np.isnan(inc_angle)+np.isnan(azi_angle)+np.isnan(lats)+np.isnan(lons)
    roi[nan_bool]   = False
    vlos[~roi]      = np.nan
    std[~roi]       = np.nan
    inc_angle[~roi] = np.nan
    azi_angle[~roi] = np.nan

    # report only the median of the std (bypass the ref_point effect)
    if medStd:
        print('use constant field veloStd -> a constant diag of Cd')
        std = np.nanmedian(std) * np.ones_like(std)

    return vlos, std, inc_angle, azi_angle, lats, lons, roi


## Unfinished
###################################################
###### predict plate motion at plate boundary #####
###################################################
DST_GRIDS = {'lats' : [31.257, 30.921, 30.517, 30.075, 29.590, 29.091, 28.616, 28.024],
             'lons' : [35.448, 35.425, 35.330, 35.207, 35.034, 34.819, 34.712, 34.509],
            }

def est_velocity_along_dst(pole):
    ## Along the DST plate boundary velocity
    lats = DST_GRIDS['lats']
    lons = DST_GRIDS['lons']
    V_enu = np.round(np.array(pole.get_velocity_enu(lats, lons, ellps=True)) * 1e3, 3)
    rates = np.round(np.sqrt(V_enu[0]**2 + V_enu[1]**2), 3)
    print('East : ', *V_enu[0])
    print('North: ', *V_enu[1])
    print('Total: ', *rates)
    print(f'Mean rates along the boundary: {np.mean(rates)} mm/yr\n')


def est_velocity_across_dst(pole):
    ## across the profile on Sinai about 100 km
    ##  N.     E.     inc.     azi.
    ## 28.69. 33.49.  37.06.  -259.43
    ## 28.49  34.48   42.46   -259.90

    geod = pyproj.Geod(ellps='WGS84')
    dist = geod.inv(33.49,28.69, 34.48,28.49)[-1]
    print(f'Distance = {dist/1e3:.3f} km')

    unit_vec1 = ut.get_unit_vector4component_of_interest(37.06, -259.43, comp='enu2los')    # near range
    unit_vec2 = ut.get_unit_vector4component_of_interest(42.46, -259.90, comp='enu2los')    # far range

    v1 = np.array(pole.get_velocity_enu(28.69, 33.49, alt=0.0, ellps=True))
    vlos1 = (v1[0]*unit_vec1[0] + v1[1]*unit_vec1[1] + v1[2]*unit_vec1[2])

    v2 = np.array(pole.get_velocity_enu(28.49, 34.48, alt=0.0, ellps=True))
    vlos2 = (v2[0]*unit_vec2[0] + v2[1]*unit_vec2[1] + v2[2]*unit_vec2[2])

    ramp = (vlos2 - vlos1) * 1e3
    print(f'Estimated ramp = {ramp:.4f} mm/yr\n')

#######################################################


# Matrix & Rotation tools
# ****************************************
def make_symm_mat(xx, xy, xz, yy, yz, zz):
    mat = np.array([[xx, xy, xz],
                    [xy, yy, yz],
                    [xz, yz, zz]])
    return mat


def xyz2enu_mat(lat, lon):
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    mat = np.array([[-np.sin(lon)            ,  np.cos(lon)            , 0          ],   # East
                    [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],   # North
                    [ np.cos(lat)*np.cos(lon),  np.cos(lat)*np.sin(lon), np.sin(lat)]])  # Up
    return mat


def rotate2xyz_cross_mat(lat, lon):
    R   = float(EARTH_RADIUS)      # meter
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)

    x1 = R * np.cos(lat) * np.cos(lon)
    x2 = R * np.cos(lat) * np.sin(lon)
    x3 = R * np.sin(lat)

    mat = np.array([[  0,  x3, -x2],
                    [-x3,   0,  x1],
                    [ x2, -x1,  0]])
    return mat


# Statistic measures
# ****************************************
def calc_reduced_chi2(r, sig, p):
    """Reduced chi-squares statistic
    https://en.wikipedia.org/wiki/Reduced_chi-squared_statistic
    *   r   :   residuals
    *   sig :   data standard deviations
    *   p   :   number of model params
    RETURN:
    *   reduced chi-2   :   <1        1      >1     >>1
                        overfitting  good   okay   poor fit
    """
    # number of valid samples
    nanidx = np.isnan(r) + np.isnan(sig)
    r   = r[~nanidx]
    sig = sig[~nanidx]
    n   = len(r)

    chi2  = np.sum( (r**2) / (sig**2) )
    rchi2 = chi2 / (n-p)
    return rchi2


def calc_wms(r, w=1):
    """(Weighted) Root-mean-squares
    *   r   :   residuals
    *   w   :   weights
    """
    # unweighted RMS if w=const.
    w *= np.ones_like(r)

    # number of valid samples
    nanidx = np.isnan(r)
    r = r[~nanidx]
    w = w[~nanidx]

    # normalize the weight; sum up to 1
    w /= np.sum(w)

    # root mean squares
    wrms = np.sqrt( np.sum(w * (r**2)) )
    return wrms
# ****************************************



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

        self.names    = []
        self.roi_set  = []
        self.N_set    = []
        self.comp_set = []
        self.Gc_set   = []
        self.T_set    = []
        self.L_set    = []
        self.G_set    = []
        self.d_set    = []
        self.std_set  = []
        self.lats_set = []
        self.lons_set = []
        self.los_inc_angle_set = []
        self.los_azi_angle_set = []
        self.bias = False

        self.m_all = 0
        self.e2    = 0
        self.rank  = 0
        self.Cm    = 0
        self.m_std = 0


    def add_insar(self, data, lats, lons, std=None, los_inc_angle=None, los_azi_angle=None, roi=None, name=None, comp='los'):
        """Add 2D InSAR data, LOS geometry, roi masking, std, track name
        """
        self.comp = comp

        # add track name if given
        if name is not None: self.names.append(name); vprint(f'add dataset {name}')

        # flatten the array (N,); N = number of valid pixels based on roi
        if roi is None:  roi = np.full(data.shape, True);  vprint('No roi specified, set all as True')
        vprint(f'flatten data {data.shape} to a 1D array')
        self.add_data_array(data[roi], lats[roi], lons[roi], std[roi], los_inc_angle[roi], los_azi_angle[roi])
        self.roi_set.append(roi)


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
        self.comp_set.append(dim_c)
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
        for k, (N, comp) in enumerate(zip(self.N_set, self.comp_set)):
            Gc   = np.full((N//comp, 3, 3), 0.)
            lats = self.lats_set[k]
            lons = self.lons_set[k]
            for i, (lat, lon) in enumerate(zip(lats, lons)):
                Gc[i,:,:] = rotate2xyz_cross_mat(lat, lon)
            self.Gc_set.append(Gc)
            vprint('built Gc shape', Gc.shape)
        vprint('~')


    def build_T(self):
        # cart2enu: v_enu = T @ v_xyz
        # where T rotates the cartesian coord to local ENU coord
        for k, (N, comp) in enumerate(zip(self.N_set, self.comp_set)):
            T    = np.full((N//comp, 3, 3), 0.)
            lats = self.lats_set[k]
            lons = self.lons_set[k]
            for i, (lat, lon) in enumerate(zip(lats, lons)):
                T[i,:,:] = xyz2enu_mat(lat, lon)
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
        for k, (N, comp) in enumerate(zip(self.N_set, self.comp_set)):
            T  = self.T_set[k]
            Gc = self.Gc_set[k]
            G  = np.full((N,3), 0.)

            # G will proj v_enu --> v_los
            if comp==1:
                L  = self.L_set[k]
                for i in range(N):
                    G[i,:] = L[i].reshape(-1,3) @ T[i] @ Gc[i]

            # G will only take v_en from v_enu (vstack G for each one of the N)
            else:
                for i in range(N//comp):
                    G[i*comp:(i+1)*comp, :] = (T[i] @ Gc[i])[:comp]

            self.G_set.append(G)
            vprint('built G shape', G.shape)
        vprint('~')


    def build_bias(self, fac=1e6):
        # add one column for est. of a constant offset parameter
        self.bias = True
        # this constant term kernel should be 1 in G,
        # but compare to other entries in G matrix (~1e6), 1 is a super small number
        # So have to use a big number factor here to avoid rank deficiency
        self.bias_fac = fac
        K = len(self.N_set)  # number of independent datasets
        for k, N in enumerate(self.N_set):
            G = self.G_set[k]
            for col in range(K):
                if col == k:
                    G = np.hstack([G, np.full((N, 1), 1.*fac)])
                else:
                    G = np.hstack([G, np.full((N, 1), 0.)])
            self.G_set[k] = np.array(G)
            vprint(f'G added bias, scaled {fac}, shape {G.shape}')
        vprint('~')


    def stack_data_operators(self):
        # Can do this step several times if you modify the dataset
        # Will just overwrite the *_all matrices
        K       = len(self.N_set)
        G_all   = self.G_set[0]
        d_all   = self.d_set[0].reshape(-1,1)
        std_all = self.std_set[0].reshape(-1,1)

        if K > 1:
            for k, (G, d, std) in enumerate(zip(self.G_set[1:],
                                                self.d_set[1:],
                                                self.std_set[1:])):
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


    def insert_Cp(self, Cp_set, scaling_facs=None):
        if scaling_facs is None:
            scaling_facs = np.ones(len(Cp_set))

        N = 0
        for Cp in Cp_set:
            N += len(Cp)
        self.Cp_all = np.zeros([N,N])
        print(f'total Cp size: {(N,N)}')

        n1 = 0
        n  = 0
        for i, (Cp, fac) in enumerate(zip(Cp_set, scaling_facs)):
            n1 += n
            n = len(Cp)
            self.Cp_all[n1:n1+n, n1:n1+n] = np.array(Cp) * fac
            print(f'appended Cp for dataset {i+1}/{len(Cp_set)}, {n} samples, scaling fac={fac}')
        print('done~')


    def invert(self, cond=1e-5, option=0, sub=5, diagOnly=True):
        # invert for model params
        # options :  0 : simple least squares
        #            1 : diag weight of 1/sigma
        #            2 : full covariance matrix
        #            3 : combine diag + full covariance
        # fallback to option=0 if not identified
        if option not in [0,1,2,3]: option = 0

        # 0. simple least-squares
        if option==0:
            vprint(' :: Ordinary least-squares, no weights')
            self.m_all, self.e2, self.rank = scipy.linalg.lstsq(self.G_all, self.d_all, cond=cond)[:3]
            Cm         = scipy.linalg.inv(self.G_all.T @ self.G_all)  # model resolution matrix
            self.Cm    = np.array(Cm)
            self.m_std = np.array(np.sqrt(np.diag(Cm)))

        # 1. weighted least squares with diag 1/std as weights
        elif option==1:
            vprint(' :: weighted least-squares: W = diag(1/std); W.T W = Cd^-1')
            self.w_diag = 1. / (self.std_all+1e-6)  # 1e-6 to stabalize
            self.m_all, self.e2, self.rank = scipy.linalg.lstsq(np.multiply(self.G_all, self.w_diag), np.multiply(self.d_all, self.w_diag), cond=cond)[:3]

            # propagate errors, calculate inversion quality
            # Eq. 2.23 in Aster et al., 2013 can be simplified as
            # Cm = ( G^T Cd^-1 G )^-1 = (G^-1 Cd G^-T); where G^-1 is the psuedo-inverse of the unweighted G
            if self.rank==len(self.m_all):
                iG       = scipy.linalg.pinv(self.G_all)
                Cd_diag  = self.std_all **2
                Cm       = np.multiply(Cd_diag.T, iG) @ iG.T # avoid sparse diagonal cov matrix
                self.Cm  = np.array(Cm)
                self.m_std = np.array(np.sqrt(np.diag(Cm)))

        # 2 & 3. unregularized least squares with a full cov matrix (C_chi, called Cx in code syntax)
        elif option==2 or option==3:
            Cx = np.array(self.Cp_all[::sub, ::sub])
            print(f' :: further downsample (x 1/{sub}), cov dimension = {Cx.shape}')
            if diagOnly:
                print(' :: only use the diagonal Cp')
                Cx = np.diag(np.diag(Cx))
            else:
                print(' :: use the full Cp')

            if option==3:
                print(f' :: C_chi = Cd + Cp')
                # add data Cd (std_all) and semivariogram Cp (characterize model incorrectness)
                # Now I have: C_chi = Cd + Cp
                Cx += np.diag(self.std_all**2)

            elif option==2:
                print(f' :: C_chi = Cp only')
            G  = np.array( self.G_all)[::sub,  :]
            d  = np.array( self.d_all)[::sub,  :]

            print(' :: m_est  =  (G^T W^T W G)^-1 G^T W^T W d  =  (G^T Cx^-1 G)^-1 G^T Cx^-1 d')
            # See also: csi.multifaultsolve.UnregularizedLeastSquareSoln()
            # See Tarantola, 2005
            # m_est  =  (G^T W^T W G)^-1 G^T W^T W d  =  (G^T Cx^-1 G)^-1 G^T Cx^-1 d
            iCx   = scipy.linalg.inv(Cx)                                ; print(' :: got Cx^-1')
            One   = scipy.linalg.inv(np.dot(  np.dot(G.T, iCx), G ) )   ; print(' :: got 1st term')
            Two   = np.dot( np.dot( G.T, iCx ), d )                     ; print(' :: got 2nd term')
            mpost = np.dot( One, Two )                                  ; print(' :: got m_post')
            self.m_all = mpost

            # propagate errors, calculate inversion quality
            # Eq. 2.23 in Aster et al., 2013 can be simplified as
            # Cm = ( G^T Cx^-1 G )^-1 = (G^-1 Cx G^-T); where G^-1 is the psuedo-inverse of the unweighted G
            Cm = np.array(One)
            self.Cm = np.array(Cm)
            self.m_std = np.array(np.sqrt(np.diag(Cm)))

        vprint('~')


    def get_model_pred(self, print_model=True, los=True):
        """Get model prediction
        """
        if print_model:
            #********************************
            #       * report results *
            vprint(' m    =\n' , self.m_all)
            vprint(' e2   =  ' , self.e2   )
            vprint(' rank =  ' , self.rank )
            vprint(' Cm    =\n', self.Cm   )
            vprint(' m_std =\n', self.m_std)
            #********************************


        self.v_pred_all = self.G_all @ self.m_all
        vprint(f'model prediction on all samples: {self.v_pred_all.shape}')
        self.m          = self.m_all[:3]   # the rotation pole vector
        self.bias_set   = []               # each bias offset term
        self.v_pred_set = []               # each data prediction array
        start = 0
        for k, (N, comp) in enumerate(zip(self.N_set, self.comp_set)):
            vprint(f'  set{k+1}: {start} to {start+N} = {N} samples')
            self.v_pred_set.append(self.v_pred_all[start : start+N].reshape(-1,comp))
            start = int(start+N)
            if self.bias:
                self.bias_set.append(self.m_all[3+k]*self.bias_fac)
        vprint(f'split model prediction into {k+1} sets')
        vprint('~')


        self.V_pred_set = []
        self.Obs_set    = []
        self.Std_set    = []
        if los and len(self.roi_set) > 0:
            # reshape 1D arrays back to 2D matrices as ROI masks
            for k, (roi, vpred, obs, std) in enumerate(zip(self.roi_set, self.v_pred_set, self.d_set, self.std_set)):
                V_pred      = np.full(roi.shape, np.nan).flatten()
                Obs         = np.full(roi.shape, np.nan).flatten()
                Std         = np.full(roi.shape, np.nan).flatten()
                idx         =    roi.flatten().nonzero()[0]
                V_pred[idx] =  vpred.flatten()
                Obs[idx]    =    obs.flatten()
                Std[idx]    =    std.flatten()
                V_pred      = V_pred.reshape((roi.shape))
                Obs         =    Obs.reshape((roi.shape))
                Std         =    Std.reshape((roi.shape))
                self.V_pred_set.append(V_pred)
                self.Obs_set.append(Obs)
                self.Std_set.append(Std)
            vprint(f'reshape {k+1} sets of prediction into 2D array')

        else:
            for k, (vpred, obs, std) in enumerate(zip(self.v_pred_set, self.d_set, self.std_set)):
                self.V_pred_set.append(vpred)
                self.Obs_set.append(obs)
                self.Std_set.append(std)
        vprint('~')


    def get_residRMS(self, demedian=True):
        self.res_set  = []
        self.rms_set  = []
        self.wrms_set = []
        self.chi2_set = []
        vprint(f'Name   RMS   WRMS   ReducedChi-2')
        for name, Obs, Std, Pred, comp in zip(self.names, self.Obs_set, self.Std_set, self.V_pred_set, self.comp_set):
            if comp == 1:
                # remove mean for insar data (ref_point)
                vprint('remove median for residual calc')
                Obs  -= np.nanmedian(Obs)
                Pred -= np.nanmedian(Pred)
            resid = Obs - Pred
            rms   = calc_wms(resid)
            wrms  = calc_wms(resid, w=1/Std)
            chi2  = calc_reduced_chi2(resid, Std, p=len(self.m))
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


    def copy_from_pole(self, inpole):
        # create a new block from an input pole
        # self block object here will be a reference object (for G and data)
        m_list = [inpole.wx, inpole.wy, inpole.wz]
        if self.bias:
            N = len(self.N_set)
            m_list += [0.0]*N
        m = np.array( m_list ) * MAS2RAD
        block_new = copy.deepcopy(self)
        block_new.m_all = np.array(m)
        block_new.get_model_pred(print_model=False)
        print(f'create a block from pole\n{inpole}')
        return block_new


    def plot_G(self):
        # plot the G matrix
        fig, ax = plt.subplots()
        im = ax.pcolorfast(self.G_all[:,:], cmap='coolwarm')#, vmin=-1e10, vmax=1e10)
        ax.invert_yaxis()
        plt.colorbar(im)
        plt.show()


    def plot_post_fit(self, plot_tks, block2=None, u_fac=1e3, cmap='RdYlBu_r', clabel='mm/year', vlim=[None,None], figsize=(10,6), fontsize=12, aspect=None, **kwargs):
        """Parameters:
        *   plot_tks    insar datasets (tracks) to plot                         [str]
        *   block2      another block object to compare with the current block  [block obj]
        *   u_fac       value scaling factor (1e3 for mm)                       [float]
        *   cmap        colormap                                                [str; colorpmap]
        *   clabel      colormap label                                          [str]
        """
        Ndset = 3       # {obs, model pred, residual}
        if block2 is not None:
            Ndset += 1  # {model diff}
        fig   = plt.figure(figsize=figsize)
        gspec = fig.add_gridspec(nrows=Ndset*2, ncols=len(plot_tks)+1, width_ratios=[1]*len(plot_tks)+[0.06], **kwargs)
        axs   = []
        # make subplots
        for i, k in enumerate(plot_tks):
            ax1 = fig.add_subplot(gspec[0:2,i])
            ax2 = fig.add_subplot(gspec[2:4,i])
            ax3 = fig.add_subplot(gspec[4:6,i])

            ki    = self.names.index(k)             # -- useful units --
            vobs  = self.Obs_set[ki]    * u_fac     # mm/yr
            vpred = self.V_pred_set[ki] * u_fac     # mm/yr
            resid = self.res_set[ki]    * u_fac     # mm/yr
            rms   = self.rms_set[ki]    * u_fac     # mm/yr
            wrms  = self.wrms_set[ki]   * u_fac     # mm/yr
            chi2  = self.chi2_set[ki]   * 1         # (dimensionless reduced chi-2)

            im1 = plot_imshow(ax1, vobs,    cbar=False, cmap=cmap, vlim=vlim, aspect=aspect)[1]
            im2 = plot_imshow(ax2, vpred,   cbar=False, cmap=cmap, vlim=vlim, aspect=aspect)[1]
            im3 = plot_imshow(ax3, resid,   cbar=False, cmap=cmap, vlim=vlim, aspect=aspect)[1]

            ax1.set_title(k, fontsize=fontsize)
            ax3.text(0.1, 0, f'RMS={rms:.2f}\n'+f'WRMS={wrms:.2f}\n'+r'$\chi_{\nu}^2$='+f'{chi2:.2f}',
                    va='bottom', ha='left', transform=ax3.transAxes, fontsize=8, zorder=99)

            if block2 is not None:
                ax4    = fig.add_subplot(gspec[6:8,i])
                vpred2 = u_fac * block2.V_pred_set[ki]
                diff   = vpred - vpred2
                diffrms= np.sqrt(np.nanmean(diff**2))
                im4    = plot_imshow(ax4, diff, cbar=False, cmap='coolwarm', vlim=0.2*np.array(vlim), aspect=aspect)[1]
                ax4.text(0.1, 0, f'RMS={diffrms:.2f}', va='bottom', ha='left', transform=ax4.transAxes, fontsize=8, zorder=99)

            if block2 is not None:
                axs.append([ax1, ax2, ax3, ax4])
            else:
                axs.append([ax1, ax2, ax3])
        axs = np.array(axs).T

        cax1 = fig.add_subplot(gspec[1:2,-1])
        cax2 = fig.add_subplot(gspec[3:4,-1])
        cax3 = fig.add_subplot(gspec[5:6,-1])
        fig.colorbar(im1, cax=cax1, label=clabel)
        fig.colorbar(im2, cax=cax2, label=clabel)
        fig.colorbar(im3, cax=cax3, label=clabel)
        plt.text(-0.2, 0.5, 'Obs.',          va='center', rotation=90, transform=axs[0,0].transAxes)
        plt.text(-0.2, 0.5, 'Est. rotation', va='center', rotation=90, transform=axs[1,0].transAxes)
        plt.text(-0.2, 0.5, 'Residual',      va='center', rotation=90, transform=axs[2,0].transAxes)

        if block2 is not None:
            cax4 = fig.add_subplot(gspec[7:8,-1])
            fig.colorbar(im4, cax=cax4, label=clabel)
            plt.text(-0.2, 0.5, 'Model diff.', va='center', rotation=90, transform=axs[3,0].transAxes)
        return fig, axs
