""" ---------------
    a PyPMM module
    ---------------

    This module contains several numpy-based tools.
    It does not depends on other PyPMM classes.
    But it needs constants from pypmm.models.

    recommend usage:
        from pypmm import utils as ut

    author: Yuan-Kai Liu  2022-2024
"""

import os
import sys
import copy
import numpy as np
import scipy

import pyproj

import pickle

# constants
from pypmm.models import (EARTH_RADIUS_A,
                          EARTH_RADIUS_B,
                          EARTH_ECCENT,
                          MAS2RAD, MASY2DMY,
                          )

import decimal
import time

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.end_time = time.perf_counter()

    def elapsed(self):
        if self.start_time is None or self.end_time is None:
            raise ValueError("Timer has not been started or stopped.")
        return self.end_time - self.start_time

    def readable_elapsed(self):
        hours, rem = divmod(self.elapsed(), 3600)
        minutes, seconds = divmod(rem, 60)
        msg = f'Execution Time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.4f} seconds'
        return msg


def round_precision(val, prec=3):
    val = float(val)
    with decimal.localcontext() as ctx:
        ctx.prec = prec
        out_val  = decimal.Decimal(val) * decimal.Decimal(1)
        return float(out_val)

def as_sci_fmt(x, ndp):
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    if m.startswith('-'):
        m = r'$-$' + m[1:]
    return fr'{m:s}$\times$10$^{{{int(e):d}}}$'

# **************************************************
#                   linear alg
# **************************************************
def is_PD(matrix):
    """
    check positive definite
    """
    try:
        # Attempt to perform Cholesky decomposition
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


def is_PSD(matrix):
    """
    check positive semi-definite
    """
    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvalsh(matrix)
    # Check if all eigenvalues are non-negative
    return np.all(eigenvalues >= 0)


def matrix_block_diagonal(*matrices):
    """
    call scipy.linalg
    """
    return scipy.linalg.block_diag(*matrices)


def sort_diagonal_matrix(D):
    """
    sort diagonal matrix D by the elements in ascending order
    """
    # Extract the diagonal elements
    diags = np.diag(D)

    # Sort the diagonal elements in ascending order
    sorted_idx = np.argsort(diags)
    sorted_diags = diags[sorted_idx]

    # Create a new diagonal matrix with the sorted diagonal elements
    D_sorted = np.diag(sorted_diags)

    return D_sorted, sorted_idx


def matrix_diagonalization(C):
    """
    diagonalize a symmetric positive-definite matrix C
    C = L   @ D @ L.T
    D = L.T @ C @ L
    """
    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = scipy.linalg.eigh(C)

    # L is the matrix of eigenvectors
    L = eigenvectors

    # D in theory is the diagonal matrix of eigenvalues
    # here I only take eigen values for memory sake
    D = eigenvalues

    # Verify that C = L D L^T
    # C_reconstructed = L @ D @ L.T
    return L, D


def matrix_diagonalization_blockwise(*As, save=False, load=False, names=None):
    """
    diagonalize sevaral symmetric positive-definite matrix As (A1, A2, A3, ...)
    and get the diagonalized matrix D_block and eigenvector L_block
    ex:
    A = scipy.linalg.block_diag(*As) = [A1   0   0]
                                       [ 0  A2   0]
                                       [ 0   0  A3] ...
    """
    if names is None:
        names = [None] * len(As)

    # Diagonalize each As
    Ls, Ds = [], []
    for i, (_A, name) in enumerate(zip(As, names)):
        if name is None:
            name = str(i+1)
        out_file = f'diagonalized_block_{name}.pkl'

        if load and os.path.isfile(out_file):
            print(f' read block D and L from {out_file}')
            with open(out_file, 'rb') as fin:
                in_dict = pickle.load(fin)
            _L = in_dict['L']
            _D = in_dict['D']
            del in_dict

        else:
            print(f' diagonalize matrix {name}, shape: {_A.shape} size: {_A.nbytes/(1024**2):.2f} MB')
            _L, _D = matrix_diagonalization(_A)

            if save:
                with open(out_file, 'wb') as fout:
                    pickle.dump({'L': _L, 'D': _D}, fout)

        Ls.append(_L)
        Ds.append(_D)
        del _L, _D

    # concatenate D block and sort (1d array to save mem)
    D_block = np.concatenate([*Ds])
    sorted_idx = np.argsort(D_block)
    D_block = D_block[sorted_idx]
    del Ds

    # concatenate L block and sort (2d matrix)
    L_block = matrix_block_diagonal(*Ls)
    L_block = L_block[:,sorted_idx]
    del Ls

    return L_block, D_block


def decorrelate_normalize_Gd(L, D, G, d):
    """
    D is the diagonals array of some matrix C via eigen decomposition
    L is the corresponding eigenvectors
    C = L   @ D @ L.T
    D = L.T @ C @ L

    decorrelate and normalize the      G @ m = d
                             --> G_tilde @ m = d_tilde
    where
        G_tilde = D^(-1/2) @ L.T @ D
        d_tilde = D^(-1/2) @ L.T @ d
    """
    D_inv_sqrt = (1 / np.sqrt(D)).reshape(-1,1)

    #d_tilde = D_inv_sqrt @ L.T @ d   # wasting mem
    #G_tilde = D_inv_sqrt @ L.T @ G   # wasting mem
    d_tilde = np.multiply(L.T @ d, D_inv_sqrt)
    G_tilde = np.multiply(L.T @ G, D_inv_sqrt)
    return G_tilde, d_tilde


def weighted_LS(G, d, std=None, cond_thres=1e-9):
    """weighted least squares by an weight array, w
    For the purpose of avoiding a huge sparse covariance matrix

    W = diag(1/std);  W.T @ W = Cd^-1

    Gw = W*G;  dw = W*d

    Gw @ m = dw

    Eq. 2.23 in Aster et al., 2013 can be simplified as

    Cm = ( G^T Cd^-1 G )^-1
       = (G^-1 Cd G^-T)
        where G^-1 is the psuedo-inverse of the unweighted G

    INPUTS:
    G          - design matrix              ; (N, M)
    d          - data array                 ; (N, 1)
    std        - data uncorrelated error    ; (N, 1)
    cond_thres - condition number threshold ; float
    """
    # weighting
    if std is None:
        std = np.ones_like(d)

    w = 1. / (std + 1e-8)  #  1e-8 to stabalize
    Gw = np.array( np.multiply(G, w) )
    dw = np.array( np.multiply(d, w) )

    # call lstsq
    m, e2, rank, singv = scipy.linalg.lstsq(Gw, dw, cond=cond_thres)


    if rank < len(m):
        print(f'!! rank deficient rank = {rank} < {len(m)} num of params')

    # propagate errors (inversion quality)
    cond     = singv[0] / singv[-1]
    iG       = scipy.linalg.pinv(G)
    Cd_diag  = std**2
    Cm       = np.multiply(Cd_diag.T, iG) @ iG.T
    m_std    = np.array(np.sqrt(np.diag(Cm)))

    return m, Cm, m_std, e2, rank, singv, cond


def fullCov_LS(G, d, Cx=None, Cm0=None):
    """least squares with a full covariance matrix

    See: + Tarantola, 2005
         + csi.multifaultsolve.UnregularizedLeastSquareSoln()

    m_est  =  (G^T W^T W G)^-1 G^T W^T W d
           =  (G^T Cx^-1 G)^-1 G^T Cx^-1 d
           =        Cm         G^T Cx^-1 d

    Eq. 2.23 in Aster et al., 2013 can be simplified as

    Cm = ( G^T  Cx^-1 G   )^-1
       = ( G^-1 Cx    G^-T)
        where G^-1 is the psuedo-inverse of the unweighted G

    INPUTS:
    G          - design matrix              ; (N, M)
    d          - data array                 ; (N, 1)
    Cx         - data uncorrelated error    ; (N, N)
    """
    # linalg
    iCx   = scipy.linalg.inv(Cx)                                ; print(' :: got Cx^-1')
    if Cm0 is None:
        Cm = scipy.linalg.inv(np.dot(  np.dot(G.T, iCx), G ) )  ; print(' :: got 1st term, Cm')
    else:
        Cm = scipy.linalg.inv(np.dot(  np.dot(G.T, iCx), G ) + scipy.linalg.inv(Cm0)) ; print(' :: got 1st term, Cm')
    Two   = np.dot( np.dot( G.T, iCx ), d )                     ; print(' :: got 2nd term')
    mpost = np.dot( Cm, Two )                                   ; print(' :: got m_post')
    m_std = np.sqrt(np.diag(Cm))

    # 2_norm^2 residuals
    resid = d - (G @ mpost)
    e2    = np.sum(resid**2)

    # Perform SVD
    singv = scipy.linalg.svdvals(G)
    cond = singv[0] / singv[-1]
    rank = np.linalg.matrix_rank(G)

    return mpost, Cm, m_std, e2, rank, singv, cond


def fullCov_cuda_LS(G, d, Cx=None, Cm0=None, gpu_device=0):
    """least squares with a full covariance matrix

    INPUTS:
    G          - design matrix              ; (N, M)
    d          - data array                 ; (N, 1)
    Cx         - data uncorrelated error    ; (N, N)
    """
    import cuda

    device    = cuda.manager.device(gpu_device)

    precision = 'float32'
    print(' pyre cuda matrix...')
    gd  = cuda.matrix(source=d , dtype=precision)           ; print(f' cuda d matrix')
    gG  = cuda.matrix(source=G , dtype=precision)           ; print(f' cuda G matrix')
    gCi = cuda.matrix(source=Cx, dtype=precision).inverse() ; print(f' cuda Cx inv matrix')
    if Cm0 is not None:
        gCm0i = cuda.matrix(source=Cm0, dtype=precision).inverse() ; print(f' cuda Cm0 inv matrix')

    print(f' cublas linalg: 1st term, Cm')
    if Cm0 is None:
        gCm  = cuda.cublas.gemm(cuda.cublas.gemm(gG, gCi, transa=1), gG).inverse()
    else:
        gCm  = (cuda.cublas.gemm(cuda.cublas.gemm(gG, gCi, transa=1), gG)+gCm0i).inverse()

    print(f' cublas linalg: 2nd term')
    gTwo = cuda.cublas.gemm(cuda.cublas.gemm(gG, gCi, transa=1), gd)
    print(f' cublas linalg: m')
    gm   = cuda.cublas.gemm(gCm, gTwo)

    print(f' cuda copy to host')
    Cm    = gCm.copy_to_host(type='numpy')
    mpost = gm.copy_to_host(type='numpy')

    m_std = np.sqrt(np.diag(Cm))

    # 2_norm^2 residuals
    resid = d - (G @ mpost)
    e2    = np.sum(resid**2)

    # perform SVD on G
    singv = scipy.linalg.svdvals(G)
    cond = singv[0] / singv[-1]
    rank = np.linalg.matrix_rank(G)

    return mpost, Cm, m_std, e2, rank, singv, cond


def fullCov_cuda_LS_blockwise(G, d, Cs, Cm0=None, gpu_device=None):
    """least squares with a full covariance matrix

    fullCov_cuda_LS() with block-by-block covariance matrices

    To save mem :
        - set precision to float32 rather than float64
        - use cuda.matrix.inverse_cholesky() rather than cuda.matrix.inverse()
            but this end up with tiny model error, why?
    """
    if gpu_device is not None:
        import cuda
        device    = cuda.manager.device(gpu_device)
        precision = 'float32'
        print(f'gpu device {gpu_device}, precision {precision}')

    # C inverse
    Cs_inv = []
    for C in Cs:
        if gpu_device is not None:
            Ci = cuda.matrix(source=C, dtype=precision).inverse().copy_to_host(type='numpy')
        else:
            Ci = scipy.linalg.inv(C)
        Cs_inv.append(Ci)
    print(' :: got blockwise Cx^-1')


    # One: G.T @ C^-1
    One = np.full(G.T.shape, np.nan)
    n0 = 0
    for Ci in Cs_inv:
        n = len(Ci)
        One[:,n0:n0+n] = G.T[:,n0:n0+n] @ Ci
        n0 += n
    print(' :: got G.T Cx^-1')


    # Two: Cm = (One @ G)^-1            (without prior model covariance)
    if Cm0 is None:
        Cm    = scipy.linalg.inv(One @ G)

    # Two: Cm = (One @ G + Cm0^-1)^-1   (with prior model covariance)
    else:
        Cm    = scipy.linalg.inv(One @ G + scipy.linalg.inv(Cm0))

    m_std = np.sqrt(np.diag(Cm))
    print(' :: got Cm = ( G.T Cx^-1 G )^-1')


    # mpost
    mpost = Cm @ One @ d
    print(' :: got m_post')


    # 2_norm^2 residuals
    resid = d - (G @ mpost)
    e2    = np.sum(resid**2)


    # Perform SVD
    singv = scipy.linalg.svdvals(G)
    cond = singv[0] / singv[-1]
    rank = np.linalg.matrix_rank(G)

    return mpost, Cm, m_std, e2, rank, singv, cond


def multivariate_normal_centroid(ms, cs):
    """ Linear regression to find mean
    * Propagate multivariate models and covariances to the centroid (like taking the mean)

    Parameters:
    ms (np.ndarray, shape=(N,M)  ): N samples of m vectors with dimentsion M.
    cs (np.ndarray, shape=(N,M,M)): N covariance matrices corresponding to the samples.

    Returns:
    centroid   (np.array): The computed centroid vector.
    covariance (np.array): The covariance matrix of the centroid.
    """
    # n: num of samples; m: multivariate params dimension
    n, m = ms.shape

    # allocate the full covariance matrix, and fill-in cs in block diagonals
    C = matrix_block_diagonal(*cs)
    print(f'Full covariance matrix shape: {C.shape}')

    # allocate system matrix G
    I = np.eye(m)         # identity matrix for m-dim model params
    G = np.vstack([I]*n)  # block for averaging the n-samples

    # G m = d
    mpost, Cmpost, m_std, e2, rank, singv, cond = fullCov_LS(G=G, d=ms.flatten(), Cx=C)

    return mpost, Cmpost, G



# **************************************************
#                arbitrary functions
# **************************************************

def split_string(s):
    i = 0
    # Find the index where the first numeric character occurs
    while i < len(s) and s[i].isalpha():
        i += 1
    # Split the string at the found index
    part1 = s[:i]
    part2 = s[i:]
    return part1, part2


def get_track_name(name, style='1'):
    head, tail = split_string(name)

    if style == '1':
        if   head.lower().startswith('a'): cap = 'A'
        elif head.lower().startswith('d'): cap = 'D'
    elif style == '3':
        if   head.lower().startswith('a'): cap = 'Asc'
        elif head.lower().startswith('d'): cap = 'Dsc'
    elif style == 'full':
        if   head.lower().startswith('a'): cap = 'Ascending'
        elif head.lower().startswith('d'): cap = 'Descending'
    else:
        cap = str(head)

    return cap+tail


def get_array4csi( dataDict : dict   | None = None,  # option 1
                   name     : str    | None = None,  # option 1
                   block    : object | None = None,     # option 2
                   k        : int    | None = None,     # option 2
                   dset     : str    | None = 'res_set' # option 2
                   ) -> tuple:
    if all([dataDict, name]) is not None:
        print('read original input data from [dataDict]')
        dsets = dataDict[name]
        (vlos, vstd, los_inc_angle, los_azi_angle, lat, lon, roi, ref_los_vec, refyx, reflalo, bbox, std_scl, paths, comp, ramp_rate_err) = dsets
        arr = np.array(vlos[roi])
        lat = np.array(lat[roi])
        lon = np.array(lon[roi])
        inc = np.array(los_inc_angle[roi])
        azi = np.array(los_azi_angle[roi])

    elif all([block, k, dset]) is not None:
        print(f'read from block object, dataset=[{dataset}]')
        roi = block.roi_set[k]

        # 1d arrays
        arr = vars(block)[dset][k][roi]
        lat = block.lats_set[k]
        lon = block.lons_set[k]
        inc = block.los_inc_angle_set[k]
        azi = block.los_azi_angle_set[k]

    else:
        sys.exit('you made a mistake, either inputing option1 or option2')

    los = np.array(get_unit_vector4component_of_interest(inc, azi, comp='enu2los')).T

    return arr, lat, lon, los


def get_image_from_arr(arr, roi):
    image      = np.full(roi.shape, np.nan).flatten()
    idx        = roi.flatten().nonzero()[0]
    image[idx] = arr.flatten()
    image      = image.reshape((roi.shape))
    return image


def get_masked_index(mask, original_index):
    if np.isnan(original_index):
        return np.nan

    if mask[original_index]:
        return np.where(mask[:original_index+1])[0].size - 1
    else:
        return -1  # Or handle appropriately if the element is masked out


# **************************************************
#                     statistics
# **************************************************
def calc_cov(*variables):
    nrow = len(variables)
    ncol = len(variables[0])
    m    = np.full((nrow, ncol), np.nan)
    for i, v in enumerate(variables):
        m[i] = v
    cov = np.cov(m)
    return cov


def calc_reduced_chi2( r   : float | np.ndarray,
                       sig : float | np.ndarray,
                       p   : float | np.ndarray | None = 0,
                       )  -> float | np.ndarray:
    """Reduced chi-squares statistic
    The chi-square statistic per degree-of-freedom
    https://en.wikipedia.org/wiki/Reduced_chi-squared_statistic
    *   r   :   residuals
    *   sig :   data standard deviations
    *   p   :   number of model params
    RETURN:
    *   chi-2
    *   averaged chi-2
    *   reduced chi-2   :   <1        1      >1     >>1
                        overfitting  good   okay   poor fit
    *   number of data
    """
    # number of valid samples
    nanidx = np.isnan(r) + np.isnan(sig)
    r   = r[~nanidx]
    sig = sig[~nanidx]
    n   = len(r)

    # chi-sqaures
    chi2  = np.sum( (r**2) / (sig**2) )

    # averaged chi-sqaures over n number of data
    achi2 = chi2 / (n)

    # reduced chi-squares over (n-p) degree of freedom
    rchi2 = chi2 / (n-p)

    return chi2, achi2, rchi2, n


def calc_wrms( r : float | np.ndarray,
               w : float | np.ndarray | None = 1
              ) -> float | np.ndarray :
    """(Weighted) Root-mean-squares
    Note that
    + weight array is apply to each r^2 term, i.e. w * r^2
    + weights are normalized, thus is relative weighting
    + if w = None | const, assume uniform, wrms simply falls back to rms.
    INPUTS:
    *   r   :   values, residuals, whatever...
    *   w   :   weights
    """
    # unweighted RMS if w=const.
    w *= np.ones_like(r)

    # get only valid samples
    nanidx = np.isnan(r)
    r = r[~nanidx]
    w = w[~nanidx]

    # normalize the weight; w sum up to 1
    w /= np.sum(w)

    # root mean squares
    wrms = np.sqrt( np.sum(w * (r**2)) )
    return wrms



# **************************************************
#                   geodesy
# **************************************************
def project_vector_2d(vector, theta):
    """
    Project a 2D vector onto a direction specified by the angle theta (clockwise from north).

    Parameters:
    vector (array-like): The 2D vector to be projected, in the form [x, y].
    theta (float): The angle in degrees, measured clockwise from north.

    Returns:
    np.ndarray: The projected vector.
    """
    # deg to rad
    theta_rad = np.deg2rad(theta)

    # Calculate the unit vector
    unit_vector = np.array([np.sin(theta_rad), np.cos(theta_rad)])

    # Compute the dot product of the original vector with the unit vector
    if len(unit_vector.shape) > 1:
        magnitude = np.einsum('ij,ij->j', unit_vector, vector)
    else:
        magnitude = np.dot(unit_vector, vector)

    # Scale the unit vector by the magnitude to get the projected vector
    projected_vector = magnitude * unit_vector

    return projected_vector


# Radius of curvature in the prime vertical
def radius_of_curvature(lat, A, ecc):
    N = A / np.sqrt(1 - ecc**2 * np.sin(lat)**2)
    return N


def haversine_distance( coord1 : np.ndarray | list | tuple,
                        coord2 : np.ndarray | list | tuple,
                        R      : float | None = EARTH_RADIUS_A,
                        ecc    : float | None = None,
                        ) -> np.ndarray :
    '''
    Calculate distance using the Haversine Formula
    coord1  -   lat, lon [deg]
    coord2  -   lat, lon [deg]
    R       -   Earth's radius [meters], default = semi-major axis (equatorial radius)
    ecc     -   eccentricity [-]
    '''
    # Coordinates in decimal degrees (e.g. 2.89078, 12.79797)
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)
    lon1 = np.deg2rad(lon1)
    lon2 = np.deg2rad(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2

    # geodesic degrees in [radian]
    c = 2 * np.arcsin(np.sqrt(a))

    # you can use atan2 instead of arcsin and get same result
    # c = 2 * arctan2(sqrt(a), sqrt(1-a))

    # consider ellipsoid Earth
    if ecc is not None:
        R1 = radius_of_curvature(lat1, R, ecc)
        R2 = radius_of_curvature(lat2, R, ecc)
        R = (R1 + R2) / 2  # Average radius of curvature

    # length scale in meters
    d = c * R                # meter

    return np.rad2deg(c), d


def get_angles4unit_vector( unit_vec : list | tuple | np.ndarray,
                            comp     : str  | None = 'enu',
                            ) -> float | np.ndarray :
    unit_vec = np.array(unit_vec)
    unit_vec = unit_vec / np.linalg.norm(unit_vec)
    e, n, u = unit_vec

    if comp in ['enu']:
        inc_angle = np.arccos(u)
        azi_angle = np.arctan2(e, n)
        inc_angle =      np.rad2deg( inc_angle )
        azi_angle = -1 * np.rad2deg( azi_angle ) % 360

        return inc_angle, azi_angle


def los2orbit_azimuth_angle(los_az_angle, look_direction='right'):
    """Convert the azimuth angle of the LOS vector to the one of the orbit flight vector.
    Parameters: los_az_angle - np.ndarray or float, azimuth angle of the LOS vector from the ground to the SAR platform
                               measured from the north with anti-clockwise direction as positive, in the unit of degrees
    Returns:    orb_az_angle - np.ndarray or float, azimuth angle of the SAR platform along track/orbit direction
                               measured from the north with anti-clockwise direction as positive, in the unit of degrees
    """
    if look_direction == 'right':
        orb_az_angle = los_az_angle - 90
    else:
        orb_az_angle = los_az_angle + 90
    orb_az_angle -= np.round(orb_az_angle / 360.) * 360.
    return orb_az_angle


def get_unit_vector4component_of_interest( los_inc_angle : float | np.ndarray,
                                           los_az_angle  : float | np.ndarray,
                                           comp          : str   | None = 'enu2los',
                                           horz_az_angle : float | np.ndarray | None = None,
                                           ) -> np.ndarray :
    """Get the unit vector for the component of interest (in courtesy of MintPy).
    Parameters: los_inc_angle - np.ndarray or float, incidence angle from vertical, in the unit of degrees
                los_az_angle  - np.ndarray or float, azimuth angle of the LOS vector from the ground to the SAR platform
                                measured from the north with anti-clockwise direction as positive, in the unit of degrees
                comp          - str, component of interest, choose among the following values:
                                enu2los, en2los, hz2los, u2los, up2los, orb(it)_az, vert, horz
                horz_az_angle - np.ndarray or float, azimuth angle of the horizontal direction of interest
                                measured from the north with anti-clockwise direction as positive, in the unit of degrees
    Returns:    unit_vec      - list(np.ndarray/float), unit vector of the ENU component for the component of interest
    """
    # check input arguments
    comps = [
        'enu2los', 'en2los', 'hz2los', 'horz2los', 'u2los', 'vert2los',   # radar LOS / cross-track
        'en2az', 'hz2az', 'orb_az', 'orbit_az',                           # radar azimuth / along-track
        'vert', 'vertical', 'horz', 'horizontal',                         # vertical / arbitrary horizontal
    ]

    if comp not in comps:
        raise ValueError(f'un-recognized comp input: {comp}.\nchoose from: {comps}')

    if comp == 'horz' and horz_az_angle is None:
        raise ValueError('comp=horz requires horz_az_angle input!')

    # initiate output
    unit_vec = None

    if comp in ['enu2los']:
        unit_vec = [
            np.sin(np.deg2rad(los_inc_angle)) * np.sin(np.deg2rad(los_az_angle)) * -1,
            np.sin(np.deg2rad(los_inc_angle)) * np.cos(np.deg2rad(los_az_angle)),
            np.cos(np.deg2rad(los_inc_angle)),
        ]

    elif comp in ['en2los', 'hz2los', 'horz2los']:
        unit_vec = [
            np.sin(np.deg2rad(los_inc_angle)) * np.sin(np.deg2rad(los_az_angle)) * -1,
            np.sin(np.deg2rad(los_inc_angle)) * np.cos(np.deg2rad(los_az_angle)),
            np.zeros_like(los_inc_angle),
        ]

    elif comp in ['u2los', 'vert2los']:
        unit_vec = [
            np.zeros_like(los_inc_angle),
            np.zeros_like(los_inc_angle),
            np.cos(np.deg2rad(los_inc_angle)),
        ]

    elif comp in ['en2az', 'hz2az', 'orb_az', 'orbit_az']:
        orb_az_angle = los2orbit_azimuth_angle(los_az_angle)
        unit_vec = [
            np.sin(np.deg2rad(orb_az_angle)) * -1,
            np.cos(np.deg2rad(orb_az_angle)),
            np.zeros_like(orb_az_angle),
        ]

    elif comp in ['vert', 'vertical']:
        unit_vec = [0, 0, 1]

    elif comp in ['horz', 'horizontal']:
        unit_vec = [
            np.sin(np.deg2rad(horz_az_angle)) * -1,
            np.cos(np.deg2rad(horz_az_angle)),
            np.zeros_like(horz_az_angle),
        ]

    return unit_vec


def project_synthetic_motion(pole, lats, lons, inc_angles, azi_angles, roi=None, comp='en2az'):
    """
    pole : euler pole object for forward model plate motion

    comps = [
        'enu2los', 'en2los', 'hz2los', 'horz2los', 'u2los', 'vert2los',   # radar LOS / cross-track
        'en2az', 'hz2az', 'orb_az', 'orbit_az',                           # radar azimuth / along-track
        'vert', 'vertical', 'horz', 'horizontal',                         # vertical / arbitrary horizontal
    ]
    """
    # (N, )
    if roi is not None:
        lats       = lats[roi]
        lons       = lons[roi]
        inc_angles = inc_angles[roi]
        azi_angles = azi_angles[roi]

    # (3, N)
    v_enu = np.array(pole.get_velocity_enu(lats, lons))

    # (3, N)
    unit_vec = np.array(get_unit_vector4component_of_interest(los_inc_angle = inc_angles,
                                                              los_az_angle  = azi_angles,
                                                              comp          = comp
                                                             ))
    # (N, )
    v_proj = np.sum(v_enu * unit_vec, axis=0)

    if roi is not None:
        V_proj      = np.full(roi.shape, np.nan).flatten()
        idx         = roi.flatten().nonzero()[0]
        V_proj[idx] = v_proj
        V_proj      = V_proj.reshape((roi.shape))
        return V_proj

    else:
        return v_proj


# **************************************************
#            matrix transform/rotation
# **************************************************
# Reference:
#   1. Euler pole forward formulation:
#       + https://yuankailiu.github.io/assets/docs/Euler_pole_doc.pdf
#         (need a proper reference here...)
#   2. Uncertainty propagation: from cartesian pole (w_x, w_y, w_z) to spherical pole (lat, lon, rate)
#       + https://github.com/tobiscode/disstans
#       + Goudarzi, M. A., Cocard, M., & Santerre, R. (2014),*EPC: Matlab software to estimate Euler pole parameters*,GPS Solutions, 18(1), 153–162,
#         doi:`10.1007/s10291-013-0354-4 <https://doi.org/10.1007/s10291-013-0354-4

def make_symm_mat( xx : float,
                   xy : float,
                   xz : float,
                   yy : float,
                   yz : float,
                   zz : float,
                   ) -> np.ndarray :
    """Build a symmetric 3-by-3 matrix"""
    mat = np.array([[xx, xy, xz],
                    [xy, yy, yz],
                    [xz, yz, zz]])
    return mat


def R_crossProd_xyz( x : float | np.ndarray,
                     y : float | np.ndarray,
                     z : float | np.ndarray,
                     ) -> np.ndarray :
    """Rotation matrix
    rotate matrix for cross product with a location at {x, y, z}
    """
    def _mat_crossProd(x,y,z):
        mat = np.array([[ 0,   z, -y],
                        [-z,   0,  x],
                        [ y,  -x,  0]])
        return mat

    if all(isinstance(inp, (list, np.ndarray)) for inp in [x,y,z]):
        xs = np.array(x)
        ys = np.array(y)
        zs = np.array(z)
        mat = np.zeros((len(xs), 3, 3))
        for i, (x,y,z) in enumerate(zip(xs, ys, zs)):
            mat[i,:,:] = _mat_crossProd(x,y,z)

    else:
        mat = _mat_crossProd(x,y,z)

    return mat


def R_xyz2enu( lat : float | np.ndarray,
               lon : float | np.ndarray,
               inv : bool  | None = False,
               ) -> np.ndarray :
    """Rotation matrix
    rotate xyz to enu at a given {lat, lon}
    """
    def _mat_xyz2enu(lat, lon, inv=False):
        mat = np.array([[-np.sin(lon)            ,  np.cos(lon)            , 0          ],   # East
                        [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],   # North
                        [ np.cos(lat)*np.cos(lon),  np.cos(lat)*np.sin(lon), np.sin(lat)]])  # Up
        # return the inverse matrix?
        if inv:
            mat = np.linalg.inv(mat)
        return mat

    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)

    if all(isinstance(inp, (list, np.ndarray)) for inp in [lat,lon]):
        lats = np.array(lat)
        lons = np.array(lon)
        mat = np.zeros((len(lats), 3, 3))
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            mat[i,:,:] = _mat_xyz2enu(lat, lon, inv=inv)

    else:
        mat = _mat_xyz2enu(lat, lon, inv=inv)

    return mat


def T_llr2xyz( lat : float | np.ndarray,
               lon : float | np.ndarray,
               h   : float | np.ndarray | None = 0.,
               R   : float | None = EARTH_RADIUS_A,
               e   : float | None = EARTH_ECCENT,
               ) -> tuple[float, float, float] :
    """Transform
    * sph2cart()
    transform {lat, lon, radius} to {x, y, z}

    Convert spherical coordinates to cartesian.

    Parameters: lat/lon - float / np.ndarray, latitude / longitude [degree]
                r       - float / np.ndarray, radius [any units of angular distance]
    Returns:    rx/y/z  - float / np.ndarray, angular distance in X/Y/Z direction [same unit as r]
    Examples:
        # convert spherical coord to xyz coord
        x, y, z = T_llr2xyz(lat, lon, R=radius)
        # convert Euler pole (in spherical) to Euler vector (in cartesian)
        wx, wy, wz = T_llr2xyz(pole_lat, pole_lon, R=rot_rate, e=0)
    """
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    e2 = e**2
    N = R / np.sqrt(1 - e2*np.sin(lat)**2)
    x =        (N + h) * np.cos(lat) * np.cos(lon)
    y =        (N + h) * np.cos(lat) * np.sin(lon)
    z = ((1-e2)*N + h) * np.sin(lat)  # be careful here, typo in Meade+Loveless 2009, should be sin(lat)
    return x, y, z


def T_xyz2llr( x : float,
               y : float,
               z : float,
               R : float | None = EARTH_RADIUS_A,
               e : float | None = EARTH_ECCENT,
               ) -> tuple[float, float, float] :
    """Convert cartesian coordinates to spherical.
    * cart2sph()
    REFERENCE: Subirana, J. S., Zornoza, J. J., & Hernández-Pajares, M. (2016). Ellipsoidal and cartesian coordinates conversion.

    Parameters: x/y/z  - float / np.ndarray, angular distance in X/Y/Z direction [any units of distance]
    Returns:    lat/lon - float / np.ndarray, latitude / longitude  [degree]
                r       - float / np.ndarray, radius [same unit as rx/y/z]
    Examples:
        # convert xyz coord to spherical coord
        lat, lon, r = T_xyz2llr(x, y, z)
        # convert Euler vector (in cartesian) to Euler pole (in spherical)
        pole_lat, pole_lon, rot_rate = T_xyz2llr(wx, wy, wz, e=0)
    """
    # perfect sphere (closed form)
    if float(e) == 0.:
        r   = np.sqrt(x**2 + y**2 + z**2)
        lat = np.rad2deg(np.arctan2(z, np.sqrt(x**2 + y**2)))
        lon = np.rad2deg(np.arctan2(y, x))

    # ellipsoid (iterative algorithm)
    else:
        p   = np.sqrt(x**2 + y**2)
        e2  = e**2
        lon = np.rad2deg(np.arctan2(y, x))

        delta = 1
        lat_i = np.arctan2(z, (1-e2)*p)
        while delta > 1e-9:
            N_i   = R / np.sqrt(1 - e2*np.sin(lat_i)**2)
            h_i   = (p / np.cos(lat_i)) - N_i
            _tmp  = np.arctan2(z, ((1 - (e2 * N_i)/(N_i + h_i)) * p))
            delta = np.abs(_tmp - lat_i)
            lat_i = float(_tmp)

        lat = np.rad2deg(lat_i)
        r   = R + h_i

    return lat, lon, r


def R_xyz2llr_err( w_x : float, # rad/yr
                   w_y : float, # rad/yr
                   w_z : float, # rad/yr
                   cartesian_covariance : np.ndarray,
                   ) -> np.ndarray :
    """
    * R_cart2sph_err()
    Adapted from disstans/disstans/tools.py
    https://github.com/tobiscode/disstans/blob/656c8be6d3d948f66fe091c7e3982e85ee6604cb/disstans/tools.py#L1688
    Goudarzi et al., 2014, equation 18; swap the rows to {lat, lon, rate} order; there is a "sign typo" at entry (2,3) in eq. 18
    """
    w_xy_mag = np.linalg.norm([w_x, w_y])
    w_mag    = np.linalg.norm([w_x, w_y, w_z])
    G        = np.array([
                    [-w_x*w_z / (w_xy_mag * w_mag**2), -w_y*w_z / (w_xy_mag * w_mag**2),  w_xy_mag / w_mag**2], # latitude
                    [-w_y / w_xy_mag**2              ,  w_x / w_xy_mag**2              ,  0                  ], # longitude
                    [ w_x / w_mag                    ,  w_y / w_mag                    ,  w_z / w_mag        ], # rate
                    ])
    spherical_covariance = G @ cartesian_covariance @ G.T
    return spherical_covariance


def R_llr2xyz_err( lat  : float, # rad
                   lon  : float, # rad
                   rate : float, # rad/yr
                   spherical_covariance : np.ndarray,
                   ) -> np.ndarray :
    """
    * R_sph2cart_err()
    Inverse of the previous function

    Goudarzi et al., 2014, equation. 5 & 6 (no scaling here); cols in {lat, lon, rate} order
    """
    #                        latitude                           longitude                        rate
    J    = np.array([
                    [-rate*np.sin(lat)*np.cos(lon)  ,  -rate*np.cos(lat)*np.sin(lon)  ,  np.cos(lat)*np.cos(lon)],
                    [-rate*np.sin(lat)*np.sin(lon)  ,   rate*np.cos(lat)*np.cos(lon)  ,  np.cos(lat)*np.sin(lon)],
                    [ rate*np.cos(lat)              ,               0                 ,  np.sin(lat)            ],
                    ])
    cartesian_covariance = J @ spherical_covariance @ J.T
    return cartesian_covariance


def helmert_transform( x : float | np.ndarray,
                       y : float | np.ndarray,
                       z : float | np.ndarray,
                       vx: float | np.ndarray,
                       vy: float | np.ndarray,
                       vz: float | np.ndarray,
                       helmert : dict
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helmert transformation with full 7 params and their rates = 14 params

    ******************
    ITRF params URL : https://itrf.ign.fr/en/solutions/transformations

    REFERENCE       : http://geoweb.mit.edu/gg/courses/201706_UNAVCO/pdf/21-ref_frames.pdf (slide 13)
                      https://www.epncb.oma.be/_productsservices/coord_trans/TUTORIAL_Coordinate_Transformation.pdf (slide 10-17)
    Online velocity
      conversion    : https://www.epncb.oma.be/_productsservices/coord_trans/
      calculator      https://www.unavco.org/software/geodetic-utilities/plate-motion-calculator/plate-motion-calculator.html
    ******************

    INPUT
        x, y, z     : 1d array locations
        vx, vy, vz  : 1d array displacement|velocity before transformation
        helmert     : a dictionary with full 14 params (translation + scale + rotation and all their rates)
    OUTPUT
        vx, vy, vz  : 1d array after transformation
    """
    T1, dT1 = helmert['T1'] * 1e-3   , helmert['dT1'] * 1e-3    # translation & rate in x-axis  [m]   [m/yr]
    T2, dT2 = helmert['T2'] * 1e-3   , helmert['dT2'] * 1e-3    # translation & rate in y-axis  [m]   [m/yr]
    T3, dT3 = helmert['T3'] * 1e-3   , helmert['dT3'] * 1e-3    # translation & rate in z-axis  [m]   [m/yr]
    R1, dR1 = helmert['R1'] * MAS2RAD, helmert['dR1'] * MAS2RAD # rotation    & rate in x-axis  [rad] [rad/yr]
    R2, dR2 = helmert['R2'] * MAS2RAD, helmert['dR2'] * MAS2RAD # rotation    & rate in y-axis  [rad] [rad/yr]
    R3, dR3 = helmert['R3'] * MAS2RAD, helmert['dR3'] * MAS2RAD # rotation    & rate in z-axis  [rad] [rad/yr]
    D ,  dD = helmert['D']  * 1e-9   , helmert['dD']  * 1e-9    # dilatation  & rate            [-]   [-/yr]

    # position matrix
    pxyz = np.stack([x, y, z])

    # velocity matrix
    vxyz = np.stack([vx, vy, vz])

    # translation vector
    dT = np.array([dT1, dT2, dT3]).reshape(3,1)

    # Helmert matrix
    A = np.array([[  D,   -R3,    R2],
                  [ R3,     D,   -R1],
                  [-R2,    R1,     D]])

    # Helmert derivitives
    B = np.array([[  dD,   -dR3,    dR2],
                  [ dR3,     dD,   -dR1],
                  [-dR2,    dR1,     dD]])

    # apply the full transformation
    vxyz_trans = vxyz + dT + (A @ vxyz) + (B @ pxyz)

    return vxyz_trans[0,:], vxyz_trans[1,:], vxyz_trans[2,:]


def T_llh2xyz_pyproj( llh : np.ndarray | list | tuple | None=None,
                      xyz : np.ndarray | list | tuple | None=None,
                      ) -> tuple[float, float, float] :

    """Convert coordinates from WGS84 lat/long/hgt to ECEF x/y/z.
    * coord_llh2xyz()
    Parameters:
            (lat, lon, h) - each is {float | list(float) | np.ndarray}       [deg, deg, meter]
            (x, y, z)     - each is {float | list(float) | np.ndarray}   [meter, meter, meter]
    Returns
            (x, y, z)     - each is {float | list(float) | np.ndarray}   [meter, meter, meter]
            (lat, lon, h) - each is {float | list(float) | np.ndarray}       [deg, deg, meter]
    """
    # llh to xyz
    if llh is not None:
        lat, lon, h = llh

        # ensure same type between alt and lat/lon
        if isinstance(lat, np.ndarray) and not isinstance(h, np.ndarray):
            h *= np.ones_like(lat)
        elif isinstance(lat, list) and not isinstance(h, list):
            h = [h] * len(lat)

        # construct pyproj transform object
        transformer = pyproj.Transformer.from_crs(
            {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
            {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        )

        # apply coordinate transformation
        x, y, z = transformer.transform(lon, lat, h, radians=False)

        return x, y, z

    # xyz to llh
    elif xyz is not None:
        x, y, z = xyz

        # construct pyproj transform object
        transformer = pyproj.Transformer.from_crs(
            {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
            {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
        )

        # apply coordinate transformation
        lon, lat, h = transformer.transform(x, y, z)

        return lat, lon, h


def T_xyz2enu(lat, lon, xyz=None, enu=None):
    """Transform a vector at given locations (lat, lon)
      between ECEF (xyz) and the local ENU coordinate

    Reference:
        Navipedia, https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates
        Cox, A., and Hart, R.B. (1986) Plate tectonics: How it works. Blackwell Scientific Publications,
          Palo Alto, doi: 10.4236/ojapps.2015.54016. Page 145-156

    Notes: This transformation can also be acheived in matrix operation.
    Transformation via matrix rotation:
          R   = your rotation/transformation matrix (xyz -> enu) you somehow get it somewhere
        V_enu = R * V_xyz           --- Eq. (1)
        V_xyz = R^-1 * V_enu        --- Eq. (2)

    Example code (with multiple points, npts):
        Eq. (1)
        R = R_xyz2enu(lat:float|np.ndarray, lon:float|np.ndarray)
        V_enu = np.diagonal(
            np.matmul(
                R.reshape([-1,3]),
                V_xyz.T,
            ).reshape([3, npts, npts], order='F'),
            axis1=1,
            axis2=2,
        ).T

        Eq. (2)
        R = R_xyz2enu(lat:float|np.ndarray, lon:float|np.ndarray, inv=True)
        V_xyz = np.diagonal(
            np.matmul(
                R.reshape([-1,3]),
                V_enu.T,
            ).reshape([3, npts, npts], order='F'),
            axis1=1,
            axis2=2,
        ).T

    Parameters: lat/lon - float / np.ndarray, latitude/longitude      at location(s) [degree]
                xyz     - np.ndarray, x/y/z         vector component at location(s) [e.g., displacement, velocity]
                enu     - np.ndarray, east/north/up vector component at location(s) [e.g., displacement, velocity]
    Returns:    e,n,u     if given xyz
                x,y,z     if given enu
    """
    # convert the unit from degree to radian
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)

    if xyz is not None:
        # cart2enu
        x, y, z = xyz
        e = - np.sin(lon) * x \
            + np.cos(lon) * y
        n = - np.sin(lat) * np.cos(lon) * x \
            - np.sin(lat) * np.sin(lon) * y \
            + np.cos(lat) * z
        u =   np.cos(lat) * np.cos(lon) * x \
            + np.cos(lat) * np.sin(lon) * y \
            + np.sin(lat) * z
        return e, n, u

    elif enu is not None:
        # enu2cart
        e, n, u = enu
        x = - np.sin(lon) * e \
            - np.cos(lon) * np.sin(lat) * n \
            + np.cos(lon) * np.cos(lat) * u
        y =   np.cos(lon) * e \
            - np.sin(lon) * np.sin(lat) * n \
            + np.sin(lon) * np.cos(lat) * u
        z =   np.cos(lat) * n \
            + np.sin(lat) * u
        return x, y, z

    else:
        raise ValueError('Input (x,y,z) or (e,n,u) is NOT complete!')



# **************************************************
#            other geodesy  (TBA)
# **************************************************

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

    unit_vec1 = get_unit_vector4component_of_interest(37.06, -259.43, comp='enu2los')    # near range
    unit_vec2 = get_unit_vector4component_of_interest(42.46, -259.90, comp='enu2los')    # far range

    v1 = np.array(pole.get_velocity_enu(28.69, 33.49, alt=0.0, ellps=True))
    vlos1 = (v1[0]*unit_vec1[0] + v1[1]*unit_vec1[1] + v1[2]*unit_vec1[2])

    v2 = np.array(pole.get_velocity_enu(28.49, 34.48, alt=0.0, ellps=True))
    vlos2 = (v2[0]*unit_vec2[0] + v2[1]*unit_vec2[1] + v2[2]*unit_vec2[2])

    ramp = (vlos2 - vlos1) * 1e3
    print(f'Estimated ramp = {ramp:.4f} mm/yr\n')

