""" ---------------
    a PyPMM module
    ---------------

    This module contains a PyPMM Euler Pole class definition
    It needs constants & functions in pypmm.models and pypmm.utils.

    recommend usage:
        from pypmm.euler_pole import EulerPole

    Reference:
     Pichon, X. L., Francheteau, J. & Bonnin, J. Plate Tectonics; Developments in Geotectonics 6;
       Hardcover - January 1, 1973. Page 28-29
     Cox, A., and Hart, R.B. (1986) Plate tectonics: How it works. Blackwell Scientific Publications,
       Palo Alto. DOI: 10.4236/ojapps.2015.54016. Page 145-156.
     Navipedia, Transformations between ECEF and ENU coordinates. [Online].
       https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates
     Goudarzi, M. A., Cocard, M. & Santerre, R. (2014), EPC: Matlab software to estimate Euler
       pole parameters, GPS Solutions, 18, 153-162, doi: 10.1007/s10291-013-0354-4

    author: Yuan-Kai Liu  Oct, 2022
"""

import collections
import os

import numpy as np
import pyproj
from shapely import geometry

# models & global constants
from pypmm.models import (ITRF2014_PMM,
                          ITRF2020_PMM,
                          EARTH_RADIUS_A,
                          EARTH_RADIUS_MEAN,
                          EARTH_ECCENT,
                          MAS2RAD, MASY2DMY,
                          )
# pypmm tools
from pypmm import utils as ut


####################################  EulerPole class begin  #############################################
# Define the Euler pole class
EXAMPLE = """Define an Euler pole:
  Method 1 - Use an Euler vector [wx, wy, wz]
             wx/y/z   - float, angular velocity in x/y/z-axis [mas/yr or deg/Ma]
  Method 2 - Use an Euler Pole lat/lon and rotation rate [lat, lon, rot_rate]
             lat/lon  - float, Euler pole latitude/longitude [degree]
             rot_rate - float, magnitude of the angular velocity [deg/Ma or mas/yr]
             1) define rotation vection as from the center of sphere to the pole lat/lon and outward;
             2) positive for conterclockwise rotation when looking from outside along the rotation vector.

  Example:
    # equivalent ways to describe the Eurasian plate in the ITRF2014 plate motion model
    EulerPole(wx=-0.085, wy=-0.531, wz=0.770, unit='mas/yr')
    EulerPole(wx=-0.024, wy=-0.148, wz=0.214, unit='deg/Ma')
    EulerPole(pole_lat=55.070, pole_lon=-99.095, rot_rate=0.939, unit='mas/yr')
    EulerPole(pole_lat=55.070, pole_lon=-99.095, rot_rate=0.261, unit='deg/Ma')
    EulerPole(pole_lat=-55.070, pole_lon=80.905, rot_rate=-0.939, unit='mas/yr')
"""

class EulerPole:
    """EulerPole object to compute velocity for a given tectonic plate.
        The coordinate system convention:
         cartesian expression - ECEF {x,y,z} coordinate, where (x axis (0°N °E), y axis (0°N 90°E), and z axis (90°N))
         spherical expression - WG84 ellipsoid {lat,lon,rate}
    Example:
        # compute velocity of the Eurasia plate in ITRF2014-PMM from Altamimi et al. (2017)
        pole_obj = EulerPole(pole_lat=55.070, pole_lon=-99.095, rot_rate=0.939, unit='mas/yr')
        pole_obj.print_info()
        vx, vy, vz = pole_obj.get_velocity_xyz(lats, lons, alt=0.0) # in  ECEF xyz coordinate
        ve, vn, vu = pole_obj.get_velocity_enu(lats, lons, alt=0.0) # in local ENU coordinate
    """

    def __init__(self, name=None, itrf='2014',
                    wx=None,        wy=None,        wz=None,
                    wx_sig=0.,      wy_sig=0.,      wz_sig=0.,
                    pole_lat=None,  pole_lon=None,  rot_rate=None, unit='mas/yr',
                    dTx=0.,         dTy=0.,         dTz=0.):

        # check if name is provided
        if name is not None:
            if   itrf == '2014':
                PMM = ITRF2014_PMM
            elif itrf == '2020':
                PMM = ITRF2020_PMM
                # ORB: origin bias rate -- a 3-dimensional translation rate for all ITRF2020_PMM plates
                # REFERENCE:  [Altamimi et al., 2023]
                dTx, dTy, dTz = PMM['ORB'][0], PMM['ORB'][1], PMM['ORB'][2]
            if name in PMM: # if name is given properly, read from table directly
                print(f'an existing plate in ITRF{itrf} table: {name}')
                wx, wy, wz = PMM[name].omega_x, PMM[name].omega_y, PMM[name].omega_z
                wx_sig, wy_sig, wz_sig = PMM[name].omega_x_sig, PMM[name].omega_y_sig, PMM[name].omega_z_sig
            else:
                if name is not None:
                    print(f'a user-defined new plate: {name}')

        # check - unit
        if unit.lower().startswith('mas'):
            unit = 'mas/yr'

        elif unit.lower().startswith('deg'):
            unit = 'deg/Ma'
            # convert input deg/Ma to mas/yr for internal calculation
            wx = wx / MASY2DMY if wx else None
            wy = wy / MASY2DMY if wy else None
            wz = wz / MASY2DMY if wz else None
            wx_sig = wx_sig / MASY2DMY if wx_sig is not None else None
            wy_sig = wy_sig / MASY2DMY if wy_sig is not None else None
            wz_sig = wz_sig / MASY2DMY if wz_sig is not None else None
            rot_rate = rot_rate / MASY2DMY if rot_rate else None
        elif unit.lower().startswith('rad'):
            unit = 'rad/yr'
            # convert input rad/yr to mas/yr for internal calculation
            wx = wx / MAS2RAD if wx else None
            wy = wy / MAS2RAD if wy else None
            wz = wz / MAS2RAD if wz else None
            wx_sig = wx_sig / MAS2RAD if wx_sig is not None else None
            wy_sig = wy_sig / MAS2RAD if wy_sig is not None else None
            wz_sig = wz_sig / MAS2RAD if wz_sig is not None else None
            rot_rate = rot_rate / MAS2RAD if rot_rate else None

        else:
            raise ValueError(f'Unrecognized rotation rate unit: {unit}! Use mas/yr or deg/Ma')

        # calculate Euler vector and pole
        if all(var is not None for var in [wx, wy, wz]):
            # calc Euler pole from vector
            #pole_lat, pole_lon, rot_rate = cart2sph(wx, wy, wz)
            pole_lat, pole_lon, rot_rate = ut.T_xyz2llr(wx, wy, wz, e=0.)

        elif all(var is not None for var in [pole_lat, pole_lon, rot_rate]):
            # calc Euler vector from pole
            #wx, wy, wz = sph2cart(pole_lat, pole_lon, r=rot_rate)
            wx, wy, wz = ut.T_llr2xyz(pole_lat, pole_lon, R=rot_rate, e=0)

        else:
            raise ValueError(f'Incomplete Euler Pole input!\n{EXAMPLE}')

        # save member variables
        self.name = name
        self.poleLat = pole_lat   # Euler pole latitude      [degree]
        self.poleLon = pole_lon   # Euler pole longitude     [degree]
        self.rotRate = rot_rate   # angular rotation rate    [mas/yr]
        self.wx = wx              # angular velocity x       [mas/yr]
        self.wy = wy              # angular velocity y       [mas/yr]
        self.wz = wz              # angular velocity z       [mas/yr]
        self.wx_sig = wx_sig      # angular velocity x sig   [mas/yr]
        self.wy_sig = wy_sig      # angular velocity y sig   [mas/yr]
        self.wz_sig = wz_sig      # angular velocity z sig   [mas/yr]
        self.xyz_cov = None       # full cov in cartesian    [rad/yr]
        self.dTx = dTx            # origin bias rate in x    [mm/yr]
        self.dTy = dTy            # origin bias rate in y    [mm/yr]
        self.dTz = dTz            # origin bias rate in z    [mm/yr]

    def __repr__(self):
        msg = f'{self.__class__.__name__}(name={self.name}, poleLat={self.poleLat}, poleLon={self.poleLon}, '
        msg += f'rotRate={self.rotRate}, wx={self.wx}, wy={self.wy}, wz={self.wz}, unit=mas/yr)'
        return msg


    def __add__(self, other):
        """Add two Euler pole objects.

        Example:
            pole1 = EulerPole(...)
            pole2 = EulerPole(...)
            pole3 = pol2 + pol1
        """
        new_wx = self.wx + other.wx
        new_wy = self.wy + other.wy
        new_wz = self.wz + other.wz
        new_wx_sig = (self.wx_sig**2 + other.wx_sig**2)**0.5
        new_wy_sig = (self.wy_sig**2 + other.wy_sig**2)**0.5
        new_wz_sig = (self.wz_sig**2 + other.wz_sig**2)**0.5
        new_dTx = self.dTx + other.dTx
        new_dTy = self.dTy + other.dTy
        new_dTz = self.dTz + other.dTz
        return EulerPole(wx=new_wx, wy=new_wy, wz=new_wz,
                         wx_sig=new_wx_sig, wy_sig=new_wy_sig, wz_sig=new_wz_sig,
                         dTx=new_dTx, dTy=new_dTy, dTz=new_dTz)


    def __sub__(self, other):
        """Subtract two Euler pole objects.

        Example:
            pole1 = EulerPole(...)
            pole2 = EulerPole(...)
            pole3 = pol2 - pol1
        """
        new_wx = self.wx - other.wx
        new_wy = self.wy - other.wy
        new_wz = self.wz - other.wz
        new_wx_sig = (self.wx_sig**2 + other.wx_sig**2)**0.5
        new_wy_sig = (self.wy_sig**2 + other.wy_sig**2)**0.5
        new_wz_sig = (self.wz_sig**2 + other.wz_sig**2)**0.5
        new_dTx = self.dTx - other.dTx
        new_dTy = self.dTy - other.dTy
        new_dTz = self.dTz - other.dTz
        return EulerPole(wx=new_wx, wy=new_wy, wz=new_wz,
                         wx_sig=new_wx_sig, wy_sig=new_wy_sig, wz_sig=new_wz_sig,
                         dTx=new_dTx, dTy=new_dTy, dTz=new_dTz)


    def __neg__(self):
        """Negative of an Euler pole object.

        Example:
            pole1 = EulerPole(...)
            pole2 = -pol1
        """
        new_wx = -self.wx
        new_wy = -self.wy
        new_wz = -self.wz
        new_wx_sig = self.wx_sig
        new_wy_sig = self.wy_sig
        new_wz_sig = self.wz_sig
        new_dTx = -self.dTx
        new_dTy = -self.dTy
        new_dTz = -self.dTz
        return EulerPole(wx=new_wx, wy=new_wy, wz=new_wz,
                         wx_sig=new_wx_sig, wy_sig=new_wy_sig, wz_sig=new_wz_sig,
                         dTx=new_dTx, dTy=new_dTy, dTz=new_dTz)


    def append_to_pole(self, inDict):
        # assign to the pole object
        for k, v in inDict.items():
            setattr(self, k, v)


    def get_uncertainty(self, block=None, in_err=None, src='tableStd', append=True):
        """Convert the uncertainty between different units
            + Rotational pole/uncertainty can have the following two interchangeable expressions:
                cartesian expression - ECEF {x,y,z} coordinate, where (x axis (0°N °E), y axis (0°N 90°E), and z axis (90°N))
                spherical expression - WG84 ellipsoid {lat,lon,rate}
            + Can take the input uncertainty from
                (1.1) tableStd  -  Euler pole model table (ITRF2014, ITRF2020 PMM): cartesian express.
                (1.2) tableCov  -  Euler pole model table (MORVEL, GSRM PMM): cartesian express.
                (2)   block     -  Block model object: cartesian express.
                (3)   in_err    -  User input error
            + If append==True, will append the unceratinty to the Euler pole object
        """
        xyz_std     = None
        xyz_cov     = None
        xyz_mas_std = None
        xyz_deg_std = None
        sph_lat_std = None
        sph_lon_std = None
        sph_mas_std = None
        sph_deg_std = None

        # do a check
        if src == 'block':
            try:
                block.Cm[:3,:3]
            except:
                src = 'tableStd'

        # 1.1. cartesian expression: sigma from Euler pole table. default [mas/time]
        if src == 'tableStd':
            xyz_mas_std = np.array([self.wx_sig, self.wy_sig, self.wz_sig])
            xyz_std     = xyz_mas_std * MAS2RAD         # rad/year
            xyz_deg_std = np.rad2deg(xyz_std*1e6)       # deg/Ma
            xyz_cov     = np.diag(xyz_std**2)           # rad^2/year^2

        # 1.2. cartesian expression: ULL covariance matrix from Euler pole table. default [mas/time]
        elif src == 'tableCov':
            print('!!! code is not ready...')

        # 2. cartesian expression: FULL covariance matrix from block object. default [rad/time]
        elif src == 'block':
            xyz_cov     = block.Cm[:3,:3]               # rad^2/year^2
            xyz_std     = np.diag(xyz_cov)**0.5         # rad/year
            xyz_mas_std = xyz_std/MAS2RAD               # mas/year
            xyz_deg_std = np.rad2deg(xyz_std*1e6)       # deg/Ma

        # 3. user-input errors
        elif src == 'in_err':
            # cartesian expression: sigma. [mas/year] or [deg/Ma]
            if 'xyz_mas_std' in in_err:
                xyz_mas_std = np.array(in_err['xyz_mas_std']) # mas/year
                xyz_std     = xyz_mas_std * MAS2RAD           # rad/year
                xyz_cov     = np.diag(xyz_std**2)             # rad^2/year^2
                xyz_deg_std = np.rad2deg(xyz_std*1e6)         # deg/Ma
            elif 'xyz_deg_std' in in_err:
                xyz_deg_std = np.array(in_err['xyz_deg_std']) # deg/Ma
                xyz_std     = np.deg2rad(xyz_deg_std)*1e-6    # rad/year
                xyz_mas_std = xyz_std / MAS2RAD               # mas/year
                xyz_cov     = np.diag(xyz_std**2)             # rad^2/year^2
            # spherical expression: sigma. [deg], [deg], [mas/year] or [deg/Ma]
            elif ('sph_lat_std' in in_err) and ('sph_lon_std' in in_err):
                sph_lat_std  = float(in_err['sph_lat_std'])      # deg
                sph_lon_std  = float(in_err['sph_lon_std'])      # deg
                if 'sph_mas_std' in in_err:
                    sph_mas_std  = float(in_err['sph_mas_std'])  # mas/year
                    sph_std      = [np.deg2rad(sph_lat_std),        # rad
                                    np.deg2rad(sph_lon_std),        # rad
                                    sph_mas_std * MAS2RAD  ]        # rad/yr
                if 'sph_deg_std' in in_err:
                    sph_deg_std  = float(in_err['sph_deg_std'])  # deg/Ma
                    sph_std      = [np.deg2rad(sph_lat_std),        # rad
                                    np.deg2rad(sph_lon_std),        # rad
                                    np.deg2rad(sph_deg_std)*1e-6]   # rad/yr
            # cartesian expression: FULL covariance. [rad^2/year^2]
            elif 'xyz_cov' in in_err:
                xyz_cov     = in_err['xyz_cov']             # rad^2/year^2
                xyz_std     = np.diag(xyz_cov)**0.5         # rad/year
                xyz_mas_std = xyz_std/MAS2RAD               # mas/year
                xyz_deg_std = np.rad2deg(xyz_std) * 1e6     # deg/Ma

        # no error source is identified
        else:
            print(f'no source of error measure: {src}')
            errDict = dict(**locals())
            errDict = {k: v for k, v in errDict.items() if any(ele in k for ele in ['cov','std']) and v is not None}
            if append:
                self.append_to_pole(errDict)
            return


        # covariance in spherical space
        # sph_cov_rad:        lat       lon         rate
        #             lat  [[ rad^2     rad^2       rad^2/yr   ]
        #             lon   [ rad^2     rad^2       rad^2/yr   ]
        #            rate   [ rad^2/yr  rad^2/yr    rad^2/yr^2 ]]
        if xyz_cov is not None: # 1. propagate from cartesian xyz_cov
            sph_cov_rad = ut.R_xyz2llr_err(self.wx*MAS2RAD,    #  rad/yr
                                           self.wy*MAS2RAD,    #  rad/yr
                                           self.wz*MAS2RAD,    #  rad/yr
                                           xyz_cov)            # (rad/yr) ^2
        else: # 2. assume diagonal from user input
            sph_std     = np.array(sph_std)
            sph_cov_rad = np.diag(sph_std**2)
            xyz_cov     = ut.R_llr2xyz_err(self.poleLat*np.pi/180,  #  rad
                                           self.poleLon*np.pi/180,  #  rad
                                           self.rotRate*MAS2RAD,    #  rad/yr
                                           sph_cov_rad)
            xyz_std     = np.diag(xyz_cov)**0.5         # rad/year
            xyz_mas_std = xyz_std/MAS2RAD               # mas/year
            xyz_deg_std = np.rad2deg(xyz_std) * 1e6     # deg/Ma

        # covariance in spherical space
        # sph_cov_deg:        lat       lon         rate
        #             lat  [[ deg^2     deg^2       deg^2/yr   ]
        #             lon   [ deg^2     deg^2       deg^2/yr   ]
        #            rate   [ deg^2/yr  deg^2/yr    deg^2/yr^2 ]]
        sph_cov_deg = sph_cov_rad * (180/np.pi)**2

        # sigma in spherical space
        sph_lat_std = (sph_cov_deg[0,0]**0.5)            # deg
        sph_lon_std = (sph_cov_deg[1,1]**0.5)            # deg
        sph_mas_std = (sph_cov_rad[2,2]**0.5) / MAS2RAD  # mas/year
        sph_deg_std = (sph_cov_deg[2,2]**0.5) * 1e6      # deg/Ma

        # return output
        self.xyz_cov = np.array(xyz_cov)
        self.wx_sig  = xyz_mas_std[0]
        self.wy_sig  = xyz_mas_std[1]
        self.wz_sig  = xyz_mas_std[2]
        errDict = dict(**locals())
        errDict = {k: v for k, v in errDict.items() if any(ele in k for ele in ['cov','std']) and v is not None}
        if append:
            self.append_to_pole(errDict)
        return


    def print_info(self, outfile=False):
        """Print the Euler pole information.
        """
        def _show_content():
            # print msg
            print('------------------ Euler Pole ± 1 * std ------------------')
            print(f'Name: {self.name}')
            print(f'Spherical expression:')
            print(f'   Pole Latitude  : {self.poleLat:{md}.4f} ± {sph_lat_std} deg')
            print(f'   Pole Longitude : {self.poleLon:{md}.4f} ± {sph_lon_std} deg')
            print(f'   Rotation rate  : {self.rotRate * MASY2DMY:{md}.4f} ± {sph_deg_std} deg/Ma   = {self.rotRate:{md}.4f} ± {sph_mas_std} mas/yr')
            print(f'Cartesian expression (angular velocity vector):')
            print(f'   wx             : {self.wx * MASY2DMY:{md}.4f} ± {x_deg_std} deg/Ma   = {self.wx:{md}.4f} ± {x_mas_std} mas/yr')
            print(f'   wy             : {self.wy * MASY2DMY:{md}.4f} ± {y_deg_std} deg/Ma   = {self.wy:{md}.4f} ± {y_mas_std} mas/yr')
            print(f'   wz             : {self.wz * MASY2DMY:{md}.4f} ± {z_deg_std} deg/Ma   = {self.wz:{md}.4f} ± {z_mas_std} mas/yr')
            if self.xyz_cov is not None:
                print(f'Full covariance in Cartesian expression (rad^2/yr^2):')
                print(f'   covariance     : {self.xyz_cov[0]}')
                print(f'                    {self.xyz_cov[1]}')
                print(f'                    {self.xyz_cov[2]}')
            print('----------------------------------------------------------')
            return

        # maximum digit
        vals = [self.poleLat, self.poleLon, self.rotRate, self.wx, self.wy, self.wz]
        md = len(str(int(np.max(np.abs(vals))))) + 5
        md += 1 if any(x < 0 for x in vals) else 0
        mde = md-4

        # errors strings
        sph_lat_std = f'{self.sph_lat_std:{mde}.4f}'    if 'sph_lat_std' in vars(self) else '--'
        sph_lon_std = f'{self.sph_lon_std:{mde}.4f}'    if 'sph_lon_std' in vars(self) else '--'
        sph_deg_std = f'{self.sph_deg_std:{mde}.4f}'    if 'sph_deg_std' in vars(self) else '--'
        sph_mas_std = f'{self.sph_mas_std:{mde}.4f}'    if 'sph_mas_std' in vars(self) else '--'
        x_deg_std   = f'{self.xyz_deg_std[0]:{mde}.4f}' if 'xyz_deg_std' in vars(self) else '--'
        y_deg_std   = f'{self.xyz_deg_std[1]:{mde}.4f}' if 'xyz_deg_std' in vars(self) else '--'
        z_deg_std   = f'{self.xyz_deg_std[2]:{mde}.4f}' if 'xyz_deg_std' in vars(self) else '--'
        x_mas_std   = f'{self.xyz_mas_std[0]:{mde}.4f}' if 'xyz_mas_std' in vars(self) else '--'
        y_mas_std   = f'{self.xyz_mas_std[1]:{mde}.4f}' if 'xyz_mas_std' in vars(self) else '--'
        z_mas_std   = f'{self.xyz_mas_std[2]:{mde}.4f}' if 'xyz_mas_std' in vars(self) else '--'

        # 1) display content
        print()
        _show_content()
        print()

        # 2) save to outfile
        if outfile:
            from contextlib import redirect_stdout
            with open(outfile, 'a') as f:
                with redirect_stdout(f):
                    _show_content()

        return


    def get_velocity_xyz(self, lat, lon, alt=0.0, ellps=True, orb=True, helmert=False, print_msg=True):
        """Compute cartesian velocity (vx, vy, vz) of the Euler Pole at point(s) of interest.

        Parameters: lat   - float / 1D/2D np.ndarray, points of interest (latitude)  [degree]
                    lon   - float / 1D/2D np.ndarray, points of interest (longitude) [degree]
                    alt   - float / 1D/2D np.ndarray, points of interest (altitude)  [meter]
                    ellps - bool, consider ellipsoidal Earth projection
        Returns:    vx    - float / 1D/2D np.ndarray, ECEF x linear velocity [meter/year]
                    vy    - float / 1D/2D np.ndarray, ECEF y linear velocity [meter/year]
                    vz    - float / 1D/2D np.ndarray, ECEF z linear velocity [meter/year]
        """
        # check input lat/lon data type (scalar / array) and shape
        poi_shape = lat.shape if isinstance(lat, np.ndarray) else None

        # convert lat/lon into x/y/z
        # Note: the conversion assumes either a spherical or spheroidal Earth, tests show that
        # using a ellipsoid as defined in WGS84 produce results closer to the UNAVCO website
        # calculator, which also uses the WGS84 ellipsoid.
        if ellps:
            if print_msg:
                print(f'assume a spheroidal Earth as defined in WGS84')
                print(f'         semi-major radius : {EARTH_RADIUS_A} m')
                print(f'              eccentricity : {EARTH_ECCENT}')
            #x, y, z = coord_llh2xyz(lat, lon, alt)
            x, y, z = ut.T_llr2xyz(lat, lon, h=alt, R=EARTH_RADIUS_A)

        else:
            if print_msg:
                print(f'assume a perfect spherical Earth')
                print(f'  arithmetic mean radius : {EARTH_RADIUS_MEAN} m')
            #x, y, z = sph2cart(lat, lon, alt+EARTH_RADIUS_MEAN)
            x, y, z = ut.T_llr2xyz(lat, lon, h=alt, R=EARTH_RADIUS_MEAN, e=0)

        # ensure matrix is flattened
        if poi_shape is not None:
            x = x.flatten()
            y = y.flatten()
            z = z.flatten()

        # compute the cartesian linear velocity (i.e., ECEF) in meter/year as:
        #
        #     V_xyz = Omega x R_i
        #
        # where R_i is location vector at point i
        xyz = np.array([x, y, z], dtype=np.float32)
        omega = np.array([self.wx, self.wy, self.wz]) * MAS2RAD
        vx, vy, vz = np.cross(omega, xyz.T).T.reshape(xyz.shape)

        # add origin bias rates (ORB) for ITRF2020 PMM
        if orb and all(_dT is not None for _dT in (self.dTx, self.dTy, self.dTz)):
            if all(float(_dT) != 0.0 for _dT in (self.dTx, self.dTy, self.dTz)):
                print(f'Apply ORB to velocity_xyz : {orb}')
                print(f'   translate velocity_xyz : {self.dTx, self.dTy, self.dTz} meter/year')
                vx += self.dTx * 1e-3  # meter to mm
                vy += self.dTy * 1e-3  # meter to mm
                vz += self.dTz * 1e-3  # meter to mm

        # Helmert transform vxyz to another NNR system
        if helmert:
            vx, vy, vz = ut.helmert_transform(x, y, z, vx, vy, vz, helmert)

        # reshape to the original shape of lat/lon
        if poi_shape is not None:
            vx = vx.reshape(poi_shape)
            vy = vy.reshape(poi_shape)
            vz = vz.reshape(poi_shape)

        return vx, vy, vz


    def get_velocity_enu(self, lat, lon, alt=0.0, ellps=True, orb=True, helmert=False, print_msg=True):
        """Compute the spherical velocity (ve, vn, vu) of the Euler Pole at point(s) of interest.

        Parameters: lat   - float / 1D/2D np.ndarray, points of interest (latitude)  [degree]
                    lon   - float / 1D/2D np.ndarray, points of interest (longitude) [degree]
                    alt   - float / 1D/2D np.ndarray, points of interest (altitude) [meter]
                    ellps - bool, consider ellipsoidal Earth projection
        Returns:    ve    - float / 1D/2D np.ndarray, east  linear velocity [meter/year]
                    vn    - float / 1D/2D np.ndarray, north linear velocity [meter/year]
                    vu    - float / 1D/2D np.ndarray, up    linear velocity [meter/year]
        """
        # calculate ECEF velocity
        vx, vy, vz = self.get_velocity_xyz(lat, lon, alt=alt, ellps=ellps, orb=orb, helmert=helmert, print_msg=print_msg)

        # convert ECEF to ENU velocity via matrix rotation: V_enu = T * V_xyz
        ve, vn, vu = ut.T_xyz2enu(lat, lon, xyz=(vx,vy,vz))
        # enforce zero vertical velocitpy when ellps=False
        # to avoid artifacts due to numerical precision
        if not ellps:
            if isinstance(lat, np.ndarray):
                vu[:] = 0
            else:
                vu = 0

        return ve, vn, vu
