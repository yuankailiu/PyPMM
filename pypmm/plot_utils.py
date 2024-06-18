""" ---------------
    a PyPMM module
    ---------------

    This module contains functions that are not dependent on
    any of PyPMM's classes. But it needs constants from pypmm.models.

    recommend usage:
        from pypmm.plot_utils import ...

    author: Yuan-Kai Liu  2022-2024
"""


import os
import numpy as np
from shapely import geometry

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from cartopy import crs as ccrs, feature as cfeature

from pypmm.models import (GSRM_NNR_V2_1_PMM,
                          NNR_MORVEL56_PMM,
                          PLATE_BOUNDARY_FILE,
                          MAS2RAD,
                          MASY2DMY,
                          )

###############  basic general plotting  ###################
def text_accomodating_xylim(fig, ax):
    # Update limits to ensure all text is within the plot area
    fig.canvas.draw()  # Draw the canvas to update text positions

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    for child in ax.get_children():
        if isinstance(child, plt.Text):
            bbox = child.get_window_extent(renderer=fig.canvas.get_renderer())
            bbox_data_coords = bbox.transformed(ax.transData.inverted())
            x_min = min(x_min, bbox_data_coords.xmin)
            x_max = max(x_max, bbox_data_coords.xmax)
            y_min = min(y_min, bbox_data_coords.ymin)
            y_max = max(y_max, bbox_data_coords.ymax)

    # Apply the new limits with a small margin
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.01
    ax.set_xlim(x_min-x_margin, x_max+x_margin)
    ax.set_ylim(y_min-y_margin, y_max+y_margin)


def update_handles(legend, ms=30, ec='k', lw=1, alpha=1.0):
    for ha in legend.legend_handles:
        ha.set_sizes([ms])
        ha.set_edgecolor(ec)
        ha.set_linewidth(lw)
        ha.set_alpha(alpha)


def lighten_color(color, amount=0.5):
    """
    url: https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def discrete_cmap( N    : int,
                   cmap : str | None = 'viridis',
                   )   -> list:
    """Create an N-bin discrete colormap from the specified input map
    REFERECE: https://gist.github.com/jakevdp/91077b0cae40f8f8244a#file-discrete_cmap-py-L8
    """
    N = int(N)
    base = matplotlib.colormaps.get_cmap(cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


# better image show
def plot_imshow( ax     : matplotlib.axes.Axes,
                 data   : np.ndarray,
                 vlim   : tuple | None = [None,None],
                 cmap   : str   | None = 'RdBu_r',
                 title  : str   | None = None,
                 label  : str   | None = 'mm/yr',
                 intp   : str   | None = 'nearest',
                 cbar   : bool  | None = True,
                 shrink : float | None = 0.5,
                 aspect : str   | None = None,
                 axon   : str   | None = 'off'
                 )     -> tuple :
    vmin, vmax = vlim
    im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, interpolation=intp, aspect=aspect)
    bound = np.argwhere(~np.isnan(data))
    ax.set_xlim(min(bound[:, 1]), max(bound[:, 1]))
    ax.set_ylim(max(bound[:, 0]), min(bound[:, 0]))
    ax.axis(axon)
    if title:
        ax.set_title(title)
    if cbar:
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', shrink=shrink)
        cbar.set_label(label=label, size=10)
        cbar.ax.tick_params(labelsize=10)
    return ax, im, cbar


# visualize the error ellipse (pole & covariance)
def confidence_ellipse( ax        : matplotlib.axes.Axes,
                        x         : np.ndarray | None = None,
                        y         : np.ndarray | None = None,
                        cov       : np.ndarray | None = None,
                        n_std     : float      | None = 2.,
                        facecolor : str        | None = 'none',
                        from_data : bool       | None = False,
                        print_msg : bool       | None = False,
                        **kwargs,
                        ):
    """Create a plot of the covariance confidence ellipse of *x* and *y*.
    https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    x, y : array-like, shape (n, )
        Input data.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    facecolor : str
        face color of the ellipse

    from_data : bool
        whether to estimate the covariance directly from input data x, y

    print_msg : bool
        whether to print message

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    ec=kwargs['ec']               if 'ec'        in kwargs else 'k'
    edgecolor=kwargs['edgecolor'] if 'edgecolor' in kwargs else 'k'
    alpha=kwargs['alpha']         if 'alpha'     in kwargs else 1
    #zorder=kwargs['zorder']       if 'zorder'    in kwargs else 2

    def _transf_ellipse(ax, elps, transf):
        # apply the affine transform and then the map coordinate transform
        if hasattr(ax, 'projection'):
            if print_msg: print('affine + ccrs.PlateCarree')
            elps.set_transform(transf + ccrs.PlateCarree()._as_mpl_transform(ax))
        else:
            if print_msg: print('affine + ax.transData')
            elps.set_transform(transf + ax.transData)
        return elps


    # covariance from data x, y
    if from_data:
        if x.size != y.size:
            raise ValueError("x and y must be the same size")
        else:
            cov = np.cov(x, y)

    # mean of data
    mean_x = np.nanmean(x)
    mean_y = np.nanmean(y)

    # plot the error ellipse
    if cov is not None:
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this 2d dataset.
        pearson = np.clip(pearson, -1.0, 1.0)  # make sure to be bounded for numerical reason
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        kwargs['alpha'] = 1
        ellipse = Ellipse((0, 0), width=ell_radius_x*2, height=ell_radius_y*2, facecolor='none', **kwargs)

        # the ellipse bound with the given number of std (n_std=2 for 95% confid. interval)
        scale_x = np.sqrt(cov[0, 0]) * n_std
        scale_y = np.sqrt(cov[1, 1]) * n_std

        # define a affine transf: rotate, scale, then translate
        transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

        # apply the affine transform and then the map coordinate transform
        ellipse = _transf_ellipse(ax, ellipse, transf)
        ax.add_patch(ellipse)

        if facecolor != 'none':
            ellipsefill = Ellipse((0, 0), width=ell_radius_x*2, height=ell_radius_y*2, facecolor=facecolor, alpha=alpha)
            ellipsefill = _transf_ellipse(ax, ellipsefill, transf)
            ax.add_patch(ellipsefill)

        if print_msg:
            print('--------------------')
            print(f'covariance matrix = \n{cov}')
            print(f'pearson = {pearson}')
            print(f'x_bound = {scale_x}; y_bound = {scale_y}')
            print(ellipse)
            print('--------------------')

        # plot the pole location
        if hasattr(ax, 'projection'):
            ax.scatter(mean_x, mean_y, s=10, marker='o', fc='white', ec=ec, transform=ccrs.PlateCarree())
        else:
            ax.scatter(mean_x, mean_y, s=10, marker='o', fc='white', ec=ec)

    else:
        # plot the pole location, show legend
        kwargs['alpha'] = 1
        if hasattr(ax, 'projection'):
            ax.scatter(mean_x, mean_y, s=10, marker='o', fc='white', transform=ccrs.PlateCarree(), **kwargs)
        else:
            ax.scatter(mean_x, mean_y, s=10, marker='o', fc='white', **kwargs)
    return


###############  block model plotting  ###################

def plot_block_diff( block1   : object,
                     block2   : object,
                     plot_tks : list[str],
                     u_fac    : float | None = 1e3,
                     cmap     : str   | None = 'RdYlBu_r',
                     clabel   : str   | None = 'mm/year',
                     vlim     : list  | None = [-4,4],
                     figsize  : tuple | None = (10,6),
                     fontsize : float | None = 12,
                     ) -> tuple :
    fig   = plt.figure(figsize=figsize)
    gspec = fig.add_gridspec(3*2, len(plot_tks)+1, width_ratios=[1]*len(plot_tks)+[0.06])
    axs   = []
    # make subplots
    for i, k in enumerate(plot_tks):
        ax1 = fig.add_subplot(gspec[0:2,i])
        ax2 = fig.add_subplot(gspec[2:4,i])
        ax3 = fig.add_subplot(gspec[4:6,i])

        ki     = block1.names.index(k)
        vpred1 = u_fac * block1.V_pred_set[ki]
        vpred2 = u_fac * block2.V_pred_set[ki]
        diff   = vpred2 - vpred1

        im1 = plot_imshow(ax1, vpred1, cbar=False, cmap=cmap, vlim=vlim)[1]
        im2 = plot_imshow(ax2, vpred2, cbar=False, cmap=cmap, vlim=vlim)[1]
        im3 = plot_imshow(ax3, diff,   cbar=False, cmap='coolwarm', vlim=0.2*np.array(vlim))[1]
        ax1.set_title(k, fontsize=fontsize)

        axs.append([ax1, ax2, ax3])
    axs = np.array(axs).T

    cax1 = fig.add_subplot(gspec[1:2,-1])
    cax2 = fig.add_subplot(gspec[3:4,-1])
    cax3 = fig.add_subplot(gspec[5:6,-1])
    fig.colorbar(im1, cax=cax1, label=clabel)
    fig.colorbar(im2, cax=cax2, label=clabel)
    fig.colorbar(im3, cax=cax3, label=clabel)
    return fig, axs


def plot_pole_covariance( poles    : object,
                          names    : list[str],
                          colors   : list[str],
                          n_std    : float | None = 2,
                          extent   : str   | list | tuple | None = 'auto',
                          radius   : float | None = 20,
                          axes     : list[matplotlib.axes.Axes] | None = None,
                          grids_on : bool  | None = True,
                          **kwargs,
                          ) -> tuple :
    """Provide the pole objects, plot the covariance in 3 subplots
        1. Lat vs Lon
        2. Lat vs rate
        3. Lon vs rate
    """
    print(f'plot {n_std} sigmas of the covariance as error ellipse')
    if not isinstance(poles,  list): poles  = [poles]
    if not isinstance(names,  list): names  = [names]
    if not isinstance(colors, list): colors = [colors]

    map_proj = ccrs.PlateCarree()
    fig      = plt.figure(figsize=(8,8))
    gspec    = fig.add_gridspec(2,2)

    # initialize axes
    if axes is not None:
        ax1, ax2, ax3 = axes[0], axes[1], axes[2]
    else:
        ax1 = fig.add_subplot(gspec[0,0], projection=map_proj, aspect='auto')
        ax2 = fig.add_subplot(gspec[0,1], sharey=ax1)
        ax3 = fig.add_subplot(gspec[1,0], sharex=ax1)


    # axis 1 - {Lon, Lat}
    if extent is None:
        print('no extent setting, set to global')
        ax1.set_global()
    elif extent == 'share':
        print('share map extent with x/y axis with axis 2/3, map aspect ratio will be stretched!')
    elif extent == 'auto':
        print('auto select map extent based on pole(s)')
        extent = find_extent(poles, radius=radius)[-1]
        ax1.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        print(f'user-input map extent: {extent}')
        ax1.set_extent(extent, crs=ccrs.PlateCarree())
    if grids_on:
        gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.1, linestyle='--', color='k')

    ax1.add_feature(cfeature.BORDERS  , lw=0.6)
    ax1.add_feature(cfeature.STATES   , lw=0.1)
    ax1.add_feature(cfeature.OCEAN    , color='#D4F1F4')
    ax1.add_feature(cfeature.LAKES    , color='#D4F1F4')
    ax1.add_feature(cfeature.LAND     , color='lightgray')
    ax1.add_feature(cfeature.COASTLINE, lw=0.6)

    for pole, color, name in zip(poles, colors, names):
        try:
            cov   = np.flip(pole.sph_cov_deg[:2,:2])  # {lat,lon} flip to {lon,lat}
        except:
            cov   = None
        label = str(name)
        confidence_ellipse(ax1, x=pole.poleLon, y=pole.poleLat, cov=cov, n_std=n_std,
                           edgecolor=color, facecolor=color, label=label, print_msg=False, **kwargs)
    ax1.legend(loc='upper left', fontsize=10)


    # axis 2 - {rate, lat}
    for pole, color, name in zip(poles, colors, names):
        try:
            cov = pole.sph_cov_deg[np.ix_([0,2],[0,2])]  # {lat,rate}
            cov[0,1] *= 1e6     # deg/yr     -> deg/Ma
            cov[1,0] *= 1e6     # deg/yr     -> deg/Ma
            cov[1,1] *= 1e12    # deg^2/yr^2 -> deg^2/Ma^2
            cov   = np.flip(cov)  # flip to {rate, lat}
        except:
            cov   = None
        label = str(name)
        confidence_ellipse(ax2, x=pole.rotRate*MASY2DMY, y=pole.poleLat, cov=cov, n_std=n_std,
                           edgecolor=color, facecolor=color, label=label, **kwargs)
    ax2.tick_params(axis="y",direction="in")
    ax2.tick_params(axis="x",direction="in", pad=-15)
    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')
    ax2.yaxis.set_label_position("right")
    ax2.tick_params(labeltop=False, labelbottom=True, labelleft=False, labelright=True)
    ax2.set_ylabel('Latitude [deg]')
    ax2.set_xlabel('Rate [deg/Ma]', labelpad=-32)
    ax2.xaxis.set_major_formatter('{x:.2f}')


    # axis 3 - {Lon, rate}
    for pole, color, name in zip(poles, colors, names):
        try:
            cov = pole.sph_cov_deg[np.ix_([1,2],[1,2])] # {lon,rate}
            cov[0,1] *= 1e6     # deg/yr     -> deg/Ma
            cov[1,0] *= 1e6     # deg/yr     -> deg/Ma
            cov[1,1] *= 1e12    # deg^2/yr^2 -> deg^2/Ma^2
        except:
            cov   = None
        label = str(name)
        confidence_ellipse(ax3, x=pole.poleLon, y=pole.rotRate*MASY2DMY, cov=cov, n_std=n_std,
                           edgecolor=color, facecolor=color, label=label, **kwargs)
    ax3.tick_params(axis="y",direction="in", pad=-32)
    ax3.tick_params(axis="x",direction="in")
    ax3.xaxis.set_ticks_position('both')
    ax3.yaxis.set_ticks_position('both')
    ax3.yaxis.set_label_position("right")
    ax3.tick_params(labeltop=False, labelbottom=True, labelleft=False, labelright=True)
    ax3.set_xlabel('Longitude [deg]')
    ax3.set_ylabel('Rate [deg/Ma]', labelpad=-48)
    ax3.yaxis.set_major_formatter('{x:.2f}')

    # adjust and show
    plt.subplots_adjust(hspace=.05)
    plt.subplots_adjust(wspace=.05)
    [ll.set_linewidth(1.5) for ll in ax1.spines.values()]
    [ll.set_linewidth(1.5) for ll in ax2.spines.values()]
    [ll.set_linewidth(1.5) for ll in ax3.spines.values()]
    return fig, (ax1, ax2, ax3)


###############  plate motion plotting  ###################
# Utility for plotting the plate motion on a globe
# check usage: https://github.com/yuankailiu/utils/blob/main/notebooks/PMM_plot.ipynb

def read_plate_outline( pmm_name   : str  | None = 'GSRM',
                        plate_name : str  | None = None,
                        order      : str  | None = 'lalo',
                        print_msg  : bool | None = False,
                        ) -> object | dict :
    """Read the plate boundaries for the given plate motion model.

    Parameters: pmm_name   - str, plate motion (model) name
                plate_name - str, plate name of interest, return all plates if None
                order      - str, 'lalo' or 'lola' is the output column order
    Returns:    outline    - dict, a dictionary that contains lists of vertices in lat/lon for all plates
                             OR shapely.geometry.polygon.Polygon object, boundary of the given "plate".
    """
    vprint = print if print_msg else lambda *args, **kwargs: None

    # check input
    if 'GSRM' in pmm_name:
        pmm_name = 'GSRM'
        pmm_dict = GSRM_NNR_V2_1_PMM

    elif 'MORVEL' in pmm_name:
        pmm_name = 'MORVEL'
        pmm_dict = NNR_MORVEL56_PMM

    elif 'PB2002' in pmm_name:
        pmm_name = 'PB2002'
        pmm_dict = None

    else:
        msg = f'Un-recognized plate motion model: {pmm_name}!'
        msg += '\nAvailable models: GSRM, MORVEL.'
        raise ValueError(msg)

    # plate boundary file
    plate_boundary_file = PLATE_BOUNDARY_FILE[pmm_name]
    coord_order = os.path.basename(plate_boundary_file).split('.')[-1]
    if coord_order not in ['lalo', 'lola']:
        raise ValueError(f'Can NOT recognize the lat/lon order from the file extension: .{coord_order}!')

    # dict to convert plate abbreviation to name
    plate_abbrev2name = {}
    if pmm_dict is not None:
        for key, val in pmm_dict.items():
            plate_abbrev2name[val.Abbrev.upper()] = key

    # read the plate outlines file, save them to a dictionary {plate_A: [vertices], ..., ..., ...}
    outlines = {}
    with open(plate_boundary_file) as f:
        lines = f.readlines()
        key, vertices = None, None
        # loop over lines to read
        for line in lines:
            # default comment line
            if line.startswith('***') or line.startswith('##'):
                continue

            # whether we meet a new plate name abbreviation
            if line.startswith('> ') or line.startswith('# ') or (len(line.split())==1 and line.split()[0].isalpha()):
                # whether to add the previous plate to the dictionary
                if key and vertices:
                    if plate_abbrev2name != {}:
                        pname = plate_abbrev2name[key]
                    else:
                        pname = str(key)
                    outlines[pname] = np.array(vertices)
                    vprint(f'getting {key} {pname}')
                # identify the new plate name abbreviation
                if line.startswith('> '):
                    key = line.split('> ')[1]
                elif line.startswith('# '):
                    key = line.split('# ')[1]
                else:
                    key = str(line)
                key = key.splitlines()[0].upper()
                # watch out for some plate_bound_data file has lower, upper indicating different plates
                # new vertices for the new plate
                vertices = []
                if plate_abbrev2name != {}:
                    if key not in plate_abbrev2name:
                        vprint(f' no name {key} in {pmm_name} PMM, ignore')
                        key, vertices = None, None
                        continue

            # get plate outline vertices
            else:
                if ',' in line: delim = ','
                else: delim = None
                if key:
                    vert = np.array(line.split(delim)).astype(float)
                    if coord_order != order.lower():
                        vert = np.flip(vert)
                    vertices.append(vert)

    # outline of a specific plate
    if plate_name is not None:
        if plate_name in outlines.keys():
            # convert list into shapely polygon object for easy use
            outline = geometry.Polygon(outlines[plate_name])
        else:
            if pmm_dict:
                plate_abbrev = pmm_dict[plate_name].Abbrev
            else:
                plate_abbrev = '-'
            raise ValueError(f'Can NOT found plate {plate_name} ({plate_abbrev}) in file: {plate_boundary_file}!')

    else:
        outline = outlines

    return outline


## Map extent for showing two or more poles together
def find_extent( poles  : object     | None = None,
                 radius : float      | None = None,
                 lons   : np.ndarray | None = None,
                 lats   : np.ndarray | None = None,
                 ) -> tuple :
    if poles is not None:
        if not isinstance(poles, list): poles = [poles]
        lats, lons = [], []
        for pole in poles:
            lats.append(pole.poleLat)
            lons.append(pole.poleLon)
        lat0 = np.mean(lats)
        lon0 = np.mean(lons)
        if radius is None:
            dist = np.sqrt( (np.array(lats)-lat0)**2 + (np.array(lons)-lon0)**2 )
            radius = 1.01 * np.max(dist)
            print(f'radius = {radius} deg')
        extent = [lon0-radius, lon0+radius, lat0-radius, lat0+radius]

    elif lons is not None:
        ww = 0.1 * (((np.max(lons)-np.min(lons))**2 + (np.max(lats)-np.min(lats))**2 )**0.5)
        extent = [np.min(lons)-ww, np.max(lons)+ww, np.min(lats)-ww, np.max(lats)+ww]
        lon0, lat0 = np.mean(extent[:2]), np.mean(extent[2:])

    return lon0, lat0, extent


def extent2poly(extent: tuple) -> object:
    x1, x2, y1, y2 = extent
    poly = np.array([[y1,x1],[y1,x2],[y2,x2],[y2,x1]])
    return geometry.Polygon(poly)


def sample_coords_within_polygon( polygon_obj : object,
                                  ny : int | None = 10,
                                  nx : int | None = 10,
                                  ) -> tuple[np.ndarray] :
    """Make a set of points inside the defined sphericalpolygon object.

    Parameters: polygon_obj - shapely.geometry.Polygon, a polygon object in lat/lon.
                ny          - int, number of initial sample points in the y (lat) direction.
                nx          - int, number of initial sample points in the x (lon) direction.
    Returns:    sample_lats - 1D np.ndarray, sample coordinates   in the y (lat) direction.
                sample_lons - 1D np.ndarray, sample coordinates   in the x (lon) direction.
    """
    # generate sample point grid
    poly_lats = np.array(polygon_obj.exterior.coords)[:,0]
    poly_lons = np.array(polygon_obj.exterior.coords)[:,1]
    cand_lats, cand_lons = np.meshgrid(
        np.linspace(np.min(poly_lats), np.max(poly_lats), ny),
        np.linspace(np.min(poly_lons), np.max(poly_lons), nx),
    )
    cand_lats = cand_lats.flatten()
    cand_lons = cand_lons.flatten()

    # select points inside the polygon
    flag = np.zeros(cand_lats.size, dtype=np.bool_)
    for i, (cand_lat, cand_lon) in enumerate(zip(cand_lats, cand_lons)):
        if polygon_obj.contains(geometry.Point(cand_lat, cand_lon)):
            flag[i] = True
    sample_lats = cand_lats[flag]
    sample_lons = cand_lons[flag]

    return sample_lats, sample_lons


def update_kwargs(kwargs : dict) -> dict:
    # default plot settings
    kwargs['c_ocean']     = kwargs.get('c_ocean', 'w')
    kwargs['c_land']      = kwargs.get('c_land', 'lightgray')
    kwargs['c_plate']     = kwargs.get('c_plate', 'mistyrose')
    kwargs['lw_coast']    = kwargs.get('lw_coast', 0.0)
    kwargs['lw_pbond']    = kwargs.get('lw_pbond', 1)
    kwargs['lc_pbond']    = kwargs.get('lc_pbond', 'coral')
    kwargs['alpha_plate'] = kwargs.get('alpha_plate', 0.4)
    kwargs['grid_on']     = kwargs.get('grid_on', '--')
    kwargs['grid_ls']     = kwargs.get('grid_ls', '--')
    kwargs['grid_lw']     = kwargs.get('grid_lw', 0.3)
    kwargs['grid_lc']     = kwargs.get('grid_lc', 'gray')
    kwargs['qnum']        = kwargs.get('qnum', 6)
    kwargs['font_size']   = kwargs.get('font_size', 12)

    # point of interest
    kwargs['pts_lalo']    = kwargs.get('pts_lalo', None)
    kwargs['pts_marker']  = kwargs.get('pts_marker', '^')
    kwargs['pts_ms']      = kwargs.get('pts_ms', 20)
    kwargs['pts_mfc']     = kwargs.get('pts_mfc', 'r')
    kwargs['pts_mec']     = kwargs.get('pts_mec', 'k')
    kwargs['pts_mew']     = kwargs.get('pts_mew', 1)
    return kwargs


def plot_basemap( plate_boundary   : object,
                  pole_lalo        : tuple                | None = None,
                  map_style        : str                  | None = 'globe',
                  center_lalo      : tuple                | None = None,
                  satellite_height : float                | None = 1e6,
                  extent           : list  | tuple        | None = None,
                  ax               : matplotlib.axes.Axes | None = None,
                  figsize          : list  | tuple        | None = [5, 5],
                  **kwargs,
                  ) -> tuple :
    """Plot the globe map wityh plate boundary, quivers on some points.

    Parameters: plate_boundary   - shapely.geometry.Polygon object
                epole_obj        - mintpy.objects.euler_pole.EulerPole object (can be a list of poles)
                map_style        - style of projection {'globe', 'platecarree'}
                center_lalo      - globe projection:       list of 2 float, center the map at this lat, lon
                satellite_height - globe projection:       height of the perspective view looking in meters
                extent           - PlateCarree proection:  map extent [lon0, lon1, lat0, lat1]
                qscale           - float, scaling factor of the quiver
                qunit            - float, length of the quiver legend in mm/yr
                qcolor           - str, quiver color
                unit             - str, {'mm','cm','m'} of the plate motion vector
                figsize          - figure size
                ax               - matplotlib axis
                kwargs           - dict, dictionary for plotting
    Returns:    ax               - matplotlib figure and axes objects
    Examples:
        from matplotlib import pyplot as plt
        from mintpy.objects import euler_pole
        from shapely import geometry

        # build EulerPole object
        plate_pmm = euler_pole.ITRF2014_PMM['Arabia']
        epole_obj = euler_pole.EulerPole(wx=plate_pmm.omega_x, wy=plate_pmm.omega_y, wz=plate_pmm.omega_z)

        # read plate boundary
        plate_boundary = euler_pole.read_plate_outline('GSRM', 'Arabia')

        # plot plate motion
        ax = euler_pole.plot_plate_motion(plate_boundary, epole_obj)
        plt.show()
    """
    import matplotlib.ticker as mticker
    from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                    LatitudeLocator, LongitudeLocator)

    kwargs = update_kwargs(kwargs)


    if plate_boundary:
        bnd_centroid = np.array(plate_boundary.centroid.coords)[0]

    if map_style == 'globe':
        # map projection is based on: map center and satellite_height
        # map center
        if not isinstance(center_lalo, (list, tuple, np.ndarray)):
            if center_lalo == 'point':
                center_lalo = kwargs['pts_lalo']
            elif center_lalo == 'pole' and pole_lalo is not None:
                center_lalo = pole_lalo
                if kwargs['pts_lalo'] is None:
                    kwargs['pts_lalo'] = pole_lalo
            elif center_lalo == 'mid' and pole_lalo is not None:
                if abs(pole_lalo[1] - bnd_centroid[1]) < 90.:
                    center_lalo = np.array([(pole_lalo[0] + bnd_centroid[0])/2,
                                            (pole_lalo[1] + bnd_centroid[1])%360/2])
                else:
                    center_lalo = bnd_centroid
                if kwargs['pts_lalo'] is None:
                    kwargs['pts_lalo'] = pole_lalo
            else:
                center_lalo = bnd_centroid

        print(f'Map center at ({center_lalo[0]:.1f}N, {center_lalo[1]:.1f}E)')
        map_proj = ccrs.NearsidePerspective(center_lalo[1], center_lalo[0], satellite_height=satellite_height)
        map_proj.threshold = map_proj.threshold/500.  # set finer threshold for line segment along the projection
        #https://stackoverflow.com/questions/60685245/plot-fine-grained-geodesic-with-cartopy
        extent   = None
        extentPoly = plate_boundary
    elif map_style == 'platecarree':
        map_proj   = ccrs.PlateCarree()
        extentPoly = extent2poly(extent)


    # make a base map from cartopy
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection=map_proj))

    # map extent & grids
    if map_style == 'globe':
        # make the map global rather than have it zoom in to the extents of any plotted data
        ax.set_global()
        if kwargs['grid_on']:
            gl = ax.gridlines(crs=ccrs.PlateCarree(),
                            color=kwargs['grid_lc'],
                            linestyle=kwargs['grid_ls'],
                            linewidth=kwargs['grid_lw'],
                            xlocs=np.arange(-180,180,10),
                            ylocs=np.arange(-80,81,10),
                            )
    elif map_style == 'platecarree':
        ax.set_extent(extent, crs=map_proj)
        if kwargs['grid_on']:
            gl = ax.gridlines(crs=ccrs.PlateCarree(),
                            color=kwargs['grid_lc'],
                            draw_labels=True,
                            linestyle=kwargs['grid_ls'],
                            linewidth=kwargs['grid_lw'],
                            xlocs=mticker.MaxNLocator(4),
                            ylocs=mticker.MaxNLocator(4),
                            xformatter=LongitudeFormatter(),
                            yformatter=LatitudeFormatter(),
                            )
            gl.top_labels  = False
            gl.left_labels = False

    # cartopy features
    ax.add_feature(cfeature.OCEAN, color=kwargs['c_ocean'])
    ax.add_feature(cfeature.LAND,  color=kwargs['c_land'])
    ax.add_feature(cfeature.COASTLINE, linewidth=kwargs['lw_coast'])

    # add the plate polygon
    if plate_boundary:
        poly_lats = np.array(plate_boundary.exterior.coords)[:, 0]
        poly_lons = np.array(plate_boundary.exterior.coords)[:, 1]
        # ccrs.Geodetic()
        ax.plot(poly_lons, poly_lats, color=kwargs['lc_pbond'], transform=ccrs.PlateCarree(), linewidth=kwargs['lw_pbond'])
        ax.fill(poly_lons, poly_lats, color=kwargs['c_plate'],  transform=ccrs.PlateCarree(), alpha=kwargs['alpha_plate'])
        # overlap the extent of interest and plate
        polygon = plate_boundary.intersection(extentPoly)
    else:
        polygon = extentPoly

    # add custom points (e.g., show some points of interest)
    if kwargs['pts_lalo'] is not None:
        ax.scatter(kwargs['pts_lalo'][1], kwargs['pts_lalo'][0],
                   marker=kwargs['pts_marker'], s=kwargs['pts_ms'],
                   fc=kwargs['pts_mfc'], ec=kwargs['pts_mec'],
                   lw=kwargs['pts_mew'], transform=ccrs.PlateCarree())

    return ax, polygon


def plot_plate_motion( plate_boundary   : object,
                       map_style        : str                  | None = 'globe',
                       center_lalo      : tuple                | None = None,
                       satellite_height : float                | None = 1e6,
                       extent           : list  | tuple        | None = None,
                       ax               : matplotlib.axes.Axes | None = None,
                       figsize          : list  | tuple        | None = [5, 5],
                       epole_obj        : object               | None = None,
                       orb              : bool                 | None = True,
                       helmert          : dict                 | None = False,
                       Ve               : list                 | None = None,
                       Vn               : list                 | None = None,
                       Lats             : list                 | None = None,
                       Lons             : list                 | None = None,
                       qscale           : float                | None = 200,
                       qunit            : float                | None = 50,
                       qwidth           : float                | None = .0075,
                       qcolor           : list[str]   | str    | None = 'coral',
                       qalpha           : list[float] | float  | None = 1.,
                       qname            : list[str]   | str    | None = None,
                       unit             : str                  | None = 'mm',
                       **kwargs,
                       ) -> tuple :


    kwargs = update_kwargs(kwargs)

    # multi src plots:
    if epole_obj is not None:
        if not isinstance(epole_obj, list): epole_obj = [epole_obj]
        pole_lalo = np.array([epole_obj[0].poleLat, epole_obj[0].poleLon])
        N = len(epole_obj)
    else:
        pole_lalo = None
        if Ve is not None: N = len(Ve)
    if not isinstance(helmert, list): helmert = N * [helmert]
    if not isinstance(qcolor,  list): qcolor  = N * [qcolor]
    if not isinstance(qalpha,  list): qalpha  = N * [qalpha]
    if not isinstance(qname,   list): qname   = N * [qname]
    if all(_inp is not None for _inp in (Lats, Lons)):
        if len(Lats) != N: Lats = N * [Lats]
        if len(Lons) != N: Lons = N * [Lons]

    # map extent
    if extent is None:
        lon_0, lat_0, extent = find_extent(poles=epole_obj, lons=Lons, lats=Lats)

    # plot basemap
    ax, polygon = plot_basemap(plate_boundary, pole_lalo=pole_lalo, map_style=map_style, center_lalo=center_lalo,
                               satellite_height=satellite_height, extent=extent, ax=ax, figsize=figsize, **kwargs)


    if all(_inp is not None for _inp in (Ve, Vn, Lats, Lons)):
        # VECTORS from : input Ve Vn lists
        print('you gave me ve vn, plot from input vectors')

    elif epole_obj is not None:
        # VECTORS from : input pole lists
        print('plot from poles')
        Lats = []
        Lons = []
        Ve   = []
        Vn   = []
        for j, (epole, helm) in enumerate(zip(epole_obj, helmert)):
            # select sample points inside the polygon
            _lats, _lons = sample_coords_within_polygon(polygon, ny=kwargs['qnum'], nx=kwargs['qnum'])

            # calculate plate motion on sample points
            _ve, _vn = epole.get_velocity_enu(lat=_lats, lon=_lons, orb=orb, helmert=helm)[:2]

            Ve.append(_ve)
            Vn.append(_vn)
            Lats.append(_lats)
            Lons.append(_lons)


    for j, (ve, vn, lats, lons, qc, qa) in enumerate(zip(Ve, Vn, Lats, Lons, qcolor, qalpha)):
        # scale the vector unit
        if   unit == 'mm':  ve = np.array(ve)*1e3; vn = np.array(vn)*1e3
        elif unit == 'cm':  ve = np.array(ve)*1e2; vn = np.array(vn)*1e2

        if True:
            # correcting for "East" further toward polar region; re-normalize ve, vn
            norm = np.sqrt(ve**2 + vn**2)
            ve /= np.cos(np.deg2rad(lats))
            renorm = np.sqrt(ve**2 + vn**2)/norm
            ve /= renorm
            vn /= renorm

        # ---------- plot inplate vectors --------------
        q = ax.quiver(lons, lats, ve, vn, transform=ccrs.PlateCarree(), scale=qscale, width=qwidth, color=qc, angles="xy", alpha=qa, zorder=3)
        if j==0:
            ax.quiverkey(q, X=0.1, Y=0.1, U=qunit, label=f'{qunit} {unit}/yr', labelpos='S', coordinates='axes', color='k', fontproperties={'size':kwargs['font_size']}, zorder=10)

    # quiver lines legend
    if qname is not None:
        from matplotlib.lines import Line2D
        lines = [Line2D([0], [0], color=qc, linewidth=3, linestyle='-') for qc in qcolor]
        ax.legend(lines, qname, loc='lower right', fontsize=kwargs['font_size'])
    #-----------------------------------------------------------

    return ax