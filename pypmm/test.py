#%%
import numpy as np
import scipy

from importlib import reload
from pypmm import utils as ut
reload(ut)

import psutil

def memory_usage_psutil():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss  # in bytes


def test_least_squares_covariance_diagonalization(M=5, Ns=[100,150,200], random_seed=None):
    """a testing function
    * try different ways to run G m = d with covariance matrix C
    * make sure the solution consistency
    * why covariance diagonalization and block-wise diag?
       because we want memory-amenable tradeoff with computation time
    """
    import time

    if random_seed is not None:
        np.random.seed(random_seed)

    start_mem  = memory_usage_psutil()

    Cs = []
    for n in Ns:
        C = np.random.rand(n, n) * 1e-1
        C = np.dot(C, C.T)  # ensure C1 is symmetric positive definite
        Cs.append(C)
        del C

    print(f"Covariance matrices memory used: {(memory_usage_psutil()-start_mem) / (1024*1024)} MB\n")

    # full covariance matrix from all datasets
    start_mem  = memory_usage_psutil()
    C = ut.matrix_block_diagonal(*Cs)
    print(f"Covariance matrix shape: {C.shape}")
    print(f"Covariance matrix memory used: {(memory_usage_psutil()-start_mem) / (1024*1024)} MB\n")

    # number of params
    N = len(C)
    if M is None: M = len(C)
    del C

    # G & d
    G = np.random.rand(N, M)
    d = np.arange(N).reshape(-1,1)

    print("Method 0 : ")
    # solve G m = d, no wieghts, fast
    start_time = time.time()
    start_mem  = memory_usage_psutil()
    m = scipy.linalg.lstsq(G, d)[0]
    print(m.flatten())
    print(f"--- {time.time()-start_time} seconds ---")
    print(f"memory used: {(memory_usage_psutil()-start_mem) / (1024*1024)} MB\n")

    print("Method 1 : ")
    # solve G m = d, with full covariance C, big C, heavy computation
    start_time = time.time()
    start_mem  = memory_usage_psutil()
    C = ut.matrix_block_diagonal(*Cs)
    iCx   = scipy.linalg.inv(C)
    One   = scipy.linalg.inv(np.dot(  np.dot(G.T, iCx), G ) )
    Two   = np.dot( np.dot( G.T, iCx ), d )
    mpost = np.dot( One, Two )
    print(mpost.flatten())
    print(f"--- {time.time()-start_time} seconds ---")
    print(f"memory used: {(memory_usage_psutil()-start_mem) / (1024*1024)} MB\n")
    del C


    print("Method 2.1 : ")
    # solve G m = d, diagonalized C -> D, full C, still memory non-efficient
    start_time = time.time()
    start_mem  = memory_usage_psutil()
    C = ut.matrix_block_diagonal(*Cs)
    L, D = ut.matrix_diagonalization(C)
    Gt, dt = ut.decorrelate_normalize_Gd(L, D, G, d)
    mt  = scipy.linalg.lstsq(Gt, dt)[0]
    print(mt.flatten())
    print(f"--- {time.time()-start_time} seconds ---")
    print(f"memory used: {(memory_usage_psutil()-start_mem) / (1024*1024)} MB\n")
    del C, L, D

    print("Method 2.2 : ")
    # solve G m = d, diagonalized block-wise C -> D, memory efficient
    start_time = time.time()
    start_mem  = memory_usage_psutil()
    L_block, D_block = ut.matrix_diagonalization_blockwise(*Cs)
    Gtb, dtb = ut.decorrelate_normalize_Gd(L_block, D_block, G, d)
    mtb = scipy.linalg.lstsq(Gtb, dtb)[0]
    print(mtb.flatten())
    print(f"--- {time.time()-start_time} seconds ---")
    print(f"memory used: {(memory_usage_psutil()-start_mem) / (1024*1024)} MB\n")
    del L_block, D_block

    return

#%%%

# PMM as data input
enu_b = np.array([-1.27117780e-08,-2.28230064e-09, 1.47707815e-08])
los_v = np.array([-0.64787394, -0.11632071,  0.75281399])
print(np.dot(los_v, enu_b))

# Real A087 data
enu_b = np.array([-1.35234686e-08, -2.42803336e-09, 1.57139466e-08])
los_v = np.array([-0.64787394, -0.11632071,  0.75281399])
print(np.dot(los_v, enu_b))







### other tests
# %%
ut.R_crossProd_xyz(1,2,3)

#%%
x = np.array([1,1,1,1,1])
y = np.array([2,2,2,2,2])
z = np.array([3,3,3,3,3])

ut.R_crossProd_xyz(x, y, z)

# %%
ut.R_xyz2enu(lat=20, lon=30)

# %%
ut.T_llr2xyz(20, 30, 0)
# %%
ut.T_xyz2llr(5192546.626742392, 2997918.1927294023, 2167696.7786782286)

# %%
ut.T_llh2xyz_pyproj(llh=(20,30,0))
# %%
ut.T_llh2xyz_pyproj(xyz=(5192546.625389262, 2997918.1919481726, 2167696.7878287574))

# %%
x = np.array([1,1,1,1,1])
y = np.array([1,1,1,1,1])
z = np.array([1,1,1,1,1])
lats = np.array([45, 46, 47, 48, 49])
lons = np.array([20, 21, 22, 23, 24])
ut.T_xyz2enu(lat=lats, lon=lons, xyz=(x,y,z))
# %%
e = np.array([0.59767248, 0.57521248, 0.55257726, 0.52977372, 0.50680881])
n = np.array([-0.19920101, -0.23469152, -0.2700709 , -0.30530763, -0.34037052])
u = np.array([1.61341457, 1.61680255, 1.61917265, 1.62053295, 1.62089257])
ut.T_xyz2enu(lat=lats, lon=lons, enu=(e,n,u))

# %%
R = ut.R_xyz2enu(lat=lats, lon=lons)
V_xyz = np.tile(np.array([1,1,1]), (5,1))

V_enu = np.diagonal(
    np.matmul(
        R.reshape([-1,3]),
        V_xyz.T,
    ).reshape([3, len(V_xyz), len(V_xyz)], order='F'),
    axis1=1,
    axis2=2,
).T
#V_enu = R @ V_xyz
print(V_enu)

#%%
R = ut.R_xyz2enu(lat=lats, lon=lons, inv=True)
V_xyz = np.diagonal(
    np.matmul(
        R.reshape([-1,3]),
        V_enu.T,
    ).reshape([3, len(V_enu), len(V_enu)], order='F'),
    axis1=1,
    axis2=2,
).T
#V_xyz = np.linalg.inv(R) @ V_enu
print(V_xyz)


# %%
