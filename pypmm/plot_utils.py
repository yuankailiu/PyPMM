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
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.legend_handler import HandlerPatch
import matplotlib.transforms as transforms
import matplotlib.ticker as mticker
import matplotlib.legend as mlegend

from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator, LongitudeLocator)

from cartopy import crs as ccrs, feature as cfeature

from pypmm import utils as ut
from pypmm.models import (GSRM_NNR_V2_1_PMM,
                          NNR_MORVEL56_PMM,
                          PLATE_BOUNDARY_FILE,
                          MAS2RAD,
                          MASY2DMY,
                          )

###############  basic general plotting  ###################

def get_legend_pos_figure(fig, legend):
    # Get the figure size in inches
    fig_size_inches = fig.get_size_inches()

    # Calculate the figure dimensions in pixels
    fig_w_pix = fig_size_inches[0] * fig.dpi
    fig_h_pix = fig_size_inches[1] * fig.dpi

    # Check if there is a title and get its bounding box if it exists
    if legend.get_title() is not None:
        bbox = np.array(legend.get_title().get_window_extent())
    elif legend.get_lines() is not None:
        bbox = np.array(legend.get_lines()[0].get_window_extent())
    elif legend.get_patches() is not None:
        bbox = np.array(legend.get_patches()[0].get_window_extent())
    else:
        return None

    # conver to figure portion coord
    item_x = bbox[0,0] / fig_w_pix
    item_y = bbox[0,1] / fig_h_pix

    return item_x, item_y


class HandlerEllipse(HandlerPatch):
    def __init__(self, width=20, height=10, lw=1, **kwargs):
        super().__init__(**kwargs)
        self.width  = width
        self.height = height
        self.lw     = lw

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = (0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent)
        p = mpatches.Ellipse(xy=center, width=self.width, height=self.height, linewidth=self.lw,
                    facecolor=orig_handle.get_facecolor(), edgecolor=orig_handle.get_edgecolor())
        #self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


def create_ellipses_handler_map(ellps, width=20, height=10):
    handler_map = {}
    for i, ell in enumerate(ellps):
        handler_map[ell] = HandlerEllipse(width=width*(0.6**i), height=height*(0.5**i))
    return handler_map


def update_handles(legend, ms=30, ec='k', lw=1, alpha=1.0):
    # forget why i wrote this
    for ha in legend.legend_handles:
        ha.set_sizes([ms])
        ha.set_edgecolor(ec)
        ha.set_linewidth(lw)
        ha.set_alpha(alpha)


def update_axis_projection(ax, new_projection):
    fig = ax.figure  # Get the figure from the ax object
    pos = ax.get_position()  # Get the position of the original axis

    # Remove the original axis from the figure
    fig.delaxes(ax)

    # Create a new axis with the desired projection
    new_ax = fig.add_axes(pos, projection=new_projection)

    return new_ax


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


def tweak_color(color, luminos=1, alpha=1):
    """
    url: https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    Input:
        color   -  str / cmap / (r,g,b) / (r, g, b, a)
        luminos - [0 ~ ??] lighter to darker
        alpha   - transparency [0 ~ 1] invisible to opaque
    Output:
        c       -  str / cmap / (r,g,b) / (r, g, b, a)
    Examples:
        tweak_color('g', 0.3)
        tweak_color('#F034A3', 0.6)
        tweak_color((.3,.55,.1), 0.5, 0.8)
    """
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color

    if float(luminos) != 1.:
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        c = colorsys.hls_to_rgb(c[0], 1 - luminos * (1 - c[1]), c[2])

    if float(alpha) != 1.:
        c = matplotlib.colors.to_rgba(c, alpha=alpha)

    return c


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
def plot_imshow( ax       : matplotlib.axes.Axes,
                 data     : np.ndarray,
                 vlim     : tuple | None = [None,None],
                 cmap     : str   | None = 'RdBu_r',
                 title    : str   | None = None,
                 label    : str   | None = 'mm/yr',
                 intp     : str   | None = 'nearest',
                 cbar     : bool  | None = True,
                 shrink   : float | None = 0.65,
                 aspect   : str   | None = None,
                 axon     : str   | None = 'off',
                 fontsize : float | None = 10,
                 verbose  : bool  | None = False,
                 )     -> tuple :
    vmin, vmax = vlim

    if vmin is None: vmin = np.nanmin(data)
    if vmax is None: vmax = np.nanmax(data)
    vmin = ut.round_precision(vmin, prec=4)
    vmax = ut.round_precision(vmax, prec=4)

    im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, interpolation=intp, aspect=aspect)
    bound = np.argwhere(~np.isnan(data))
    ax.set_xlim(min(bound[:, 1]), max(bound[:, 1]))
    ax.set_ylim(max(bound[:, 0]), min(bound[:, 0]))
    ax.axis(axon)
    if title:
        ax.set_title(title, fontsize=fontsize)
    if cbar:
        cbar = plt.colorbar(im, ax=ax, ticks=[vmin, vmax], orientation='horizontal', shrink=shrink, aspect=shrink*15)
        cbar.ax.tick_params(labelsize=fontsize*shrink)
        cbar.set_label(label=label, size=fontsize*shrink)
    return ax, im, cbar


def tablelegend(ax, handles=None, col_labels=None, row_labels=None, tip_label="", *args, **kwargs):
    """
    Place a table legend on the axes.

    Creates a legend where the labels are not directly placed with the artists,
    but are used as row and column headers, looking like this:

    tip_label   | col_labels[1] | col_labels[2] | col_labels[3]
    -------------------------------------------------------------
    row_labels[1] |
    row_labels[2] |              <artists go there>
    row_labels[3] |


    Parameters
    ----------

    ax : `matplotlib.axes.Axes`
        The artist that contains the legend table, i.e. current axes instant.

    col_labels : list of str, optional
        A list of labels to be used as column headers in the legend table.
        `len(col_labels)` needs to match `ncol`.

    row_labels : list of str, optional
        A list of labels to be used as row headers in the legend table.
        `len(row_labels)` needs to match `len(handles) // ncol`.

    tip_label : str, optional
        Label for the top left corner in the legend table.

    ncol : int
        Number of columns.


    Other Parameters
    ----------------

    Refer to `matplotlib.legend.Legend` for other parameters.

    """
    #################### same as `matplotlib.axes.Axes.legend` #####################
    if handles is None:
        handles, labels, kwargs = mlegend._parse_legend_args([ax], *args, **kwargs)

    if col_labels is None and row_labels is None:
        ax.legend_ = mlegend.Legend(ax, handles, labels, **kwargs)
        ax.legend_._remove_method = ax._remove_legend
        return ax.legend_
    #################### modifications for table legend ############################
    else:
        ncol = kwargs.pop('ncol')
        handletextpad = kwargs.pop('handletextpad', 0 if col_labels is None else -2)
        tip_label = [tip_label]

        # blank rectangle handle
        extra = [Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)]

        # empty label
        empty = [""]

        # number of rows infered from number of handles and desired number of columns
        nrow = len(handles) // ncol

        # organise the list of handles and labels for table construction
        if col_labels is None:
            assert nrow == len(row_labels), "nrow = len(handles) // ncol = %s, but should be equal to len(row_labels) = %s." % (nrow, len(row_labels))
            leg_handles = extra * nrow
            leg_labels  = row_labels
        elif row_labels is None:
            assert ncol == len(col_labels), "ncol = %s, but should be equal to len(col_labels) = %s." % (ncol, len(col_labels))
            leg_handles = []
            leg_labels  = []
        else:
            assert nrow == len(row_labels), "nrow = len(handles) // ncol = %s, but should be equal to len(row_labels) = %s." % (nrow, len(row_labels))
            assert ncol == len(col_labels), "ncol = %s, but should be equal to len(col_labels) = %s." % (ncol, len(col_labels))
            leg_handles = extra + extra * nrow
            leg_labels  = tip_label + row_labels
        for col in range(ncol):
            if col_labels is not None:
                leg_handles += extra
                leg_labels  += [col_labels[col]]
            leg_handles += handles[col*nrow:(col+1)*nrow]
            leg_labels  += empty * nrow

        # Create legend
        ax.legend_ = mlegend.Legend(ax, leg_handles, leg_labels, ncol=ncol+int(row_labels is not None), handletextpad=handletextpad, **kwargs)
        ax.legend_._remove_method = ax._remove_legend
        return ax.legend_


# visualize the error ellipse (pole & covariance)
def draw_confidence_ellipse( ax        : matplotlib.axes.Axes,
                             x         : np.ndarray | None = None ,
                             y         : np.ndarray | None = None ,
                             cov       : np.ndarray | None = None ,
                             n_std     : float      | None = 2.   ,
                             color     : str        | None = 'b'  ,
                             elp_lw    : float      | None = 1    ,
                             elp_alpha : float      | None = 1    ,
                             markersize: float      | None = 24   ,
                             markerec  : float      | None = 'k'  ,
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

    color : str
        color for both edge and face

    elp_lw : float
        linewidth of ellipse edge

    elp_alpha : float
        alpha for facecolor

    markersize : float
        size of the centroid marker

    markerec : str
        edgecolor of the centroid marker

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

    # set color
    ec = color
    fc = tweak_color(color, alpha=elp_alpha)

    # plot the error ellipse
    if cov is not None:
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this 2d dataset.
        pearson = np.clip(pearson, -1.0, 1.0)  # make sure to be bounded for numerical reason
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = mpatches.Ellipse((0, 0), width=ell_radius_x*2, height=ell_radius_y*2, fc=fc, ec=ec, lw=elp_lw, **kwargs)

        # the ellipse bound with the given number of std (n_std=2 for 95% confid. interval)
        scale_x = np.sqrt(cov[0, 0]) * n_std
        scale_y = np.sqrt(cov[1, 1]) * n_std

        # define a affine transf: rotate, scale, then translate
        transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

        # apply the affine transform and then the map coordinate transform
        ellipse = _transf_ellipse(ax, ellipse, transf)
        ax.add_patch(ellipse)

        if print_msg:
            print('--------------------')
            print(f'covariance matrix = \n{cov}')
            print(f'pearson = {pearson}')
            print(f'x_bound = {scale_x}; y_bound = {scale_y}')
            print(ellipse)
            print('--------------------')

        # plot the pole location
        if hasattr(ax, 'projection'):
            ax.scatter(mean_x, mean_y, s=markersize, marker='o', fc=fc, ec=markerec, transform=ccrs.PlateCarree(), **kwargs)
        else:
            ax.scatter(mean_x, mean_y, s=markersize, marker='o', fc=fc, ec=markerec, **kwargs)

    else:
        # plot the pole location, show legend
        if hasattr(ax, 'projection'):
            ax.scatter(mean_x, mean_y, s=markersize, marker='o', fc=fc, transform=ccrs.PlateCarree(), **kwargs)
        else:
            ax.scatter(mean_x, mean_y, s=markersize, marker='o', fc=fc, **kwargs)

    return ellipse


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

def num_to_sigma_string(lst):
    if not lst:
        return "()"
    elif len(lst) == 1:
        return fr"($\pm${lst[0]}$\sigma$)"
    elif len(lst) == 2:
        return fr"($\pm${lst[0]} and $\pm${lst[1]}$\sigma$)"
    else:
        comma = fr", $\pm$"
        return fr"($\pm${comma.join(map(str, lst[:-1]))}, and $\pm${lst[-1]}$\sigma$)"


def plot_pole_covariance( poles     : object,
                          names     : list[str],
                          colors    : list[str],
                          n_std     : float | None = 2,
                          extent    : str   | list | tuple | None = 'auto',
                          radius    : float | None = 20,
                          axes      : list[matplotlib.axes.Axes] | None = None,
                          grids_on  : bool  | None = True,
                          elp_lw    : float | None = 1,
                          elp_alpha : float | None = 1,
                          axLabels  : list  | None = [None,None,'Rate [$^\circ$/Ma]'],
                          leg_ncol  : int   | None = 3,
                          **kwargs,  # for crrs.features() in ax1
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
        if len(axes) == 3:
            ax1, ax2, ax3 = axes
            axleg = ax3
        elif len(axes) == 4:
            ax1, ax2, ax3, axleg = axes
    else:
        ax1 = fig.add_subplot(gspec[0,0], projection=map_proj, aspect='auto')
        ax2 = fig.add_subplot(gspec[0,1], sharey=ax1)
        ax3 = fig.add_subplot(gspec[1,0], sharex=ax1)

    locator = mticker.MaxNLocator(nbins=5, steps=[1,2,4,5,10], prune='both')
    tickw = 1.4

    # axis labels
    lat_label, lon_label, rot_label = axLabels

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
        gl = ax1.gridlines(crs=ccrs.PlateCarree(),
                        color='gray',
                        draw_labels=False,
                        linewidth=0.1,
                        linestyle='--',
                        xlocs=locator,
                        ylocs=locator,
                        xformatter=LongitudeFormatter(),
                        yformatter=LatitudeFormatter(),
                        )
    else:
        gl = ax1.gridlines(crs=ccrs.PlateCarree(),
                        color='gray',
                        draw_labels=True,
                        linewidth=0,
                        xlocs=locator,
                        ylocs=locator,
                        xformatter=LongitudeFormatter(),
                        yformatter=LatitudeFormatter(),
                        )

        # Get the tick locations
        x_ticks = gl.xlocator.tick_values(*ax1.get_xlim())
        y_ticks = gl.ylocator.tick_values(*ax1.get_ylim())
        x_ticklabels = gl.xformatter.format_ticks(x_ticks)
        y_ticklabels = gl.yformatter.format_ticks(y_ticks)

        gl.bottom_labels = False
        gl.right_labels  = False
        ax1.set_xticks(x_ticks, crs=ccrs.PlateCarree())
        ax1.set_yticks(y_ticks, crs=ccrs.PlateCarree())
        ax1.xaxis.set_ticklabels([])
        ax1.yaxis.set_ticklabels([])
        ax1.xaxis.set_ticks_position('both')
        ax1.yaxis.set_ticks_position('both')
    ax1.tick_params(axis="both", direction="in", width=tickw, labeltop=True, labelbottom=False, labelleft=True, labelright=False)
    if lat_label is not None:
        ax1.set_ylabel(lat_label)
    if lon_label is not None:
        ax1.set_xlabel(lon_label)

    Ells = []
    Handler_map = {}
    for pole, color, name in zip(poles, colors, names):
        try:
            cov   = np.flip(pole.sph_cov_deg[:2,:2])  # {lat,lon} flip to {lon,lat}
        except:
            cov   = None

        if not isinstance(n_std, (list,np.ndarray)):
            n_std = [n_std]
        n_std.sort(reverse=True) # descending order

        if isinstance(n_std, (list,np.ndarray)):
            ells = []
            for i in range(len(n_std)):
                ci = tweak_color(color, luminos=(i+1)/len(n_std))
                ell = draw_confidence_ellipse(ax1, x=pole.poleLon, y=pole.poleLat, cov=cov, n_std=n_std[i], color=ci, elp_lw=elp_lw, elp_alpha=elp_alpha, print_msg=False)
                ells.append(ell)
            ells = tuple(ells)
            handler_map = create_ellipses_handler_map(ells)
            Ells.append(ells)
            Handler_map.update(handler_map)
        else:
            ell = draw_confidence_ellipse(ax1, x=pole.poleLon, y=pole.poleLat, cov=cov, n_std=n_std, color=color, elp_lw=elp_lw, elp_alpha=elp_alpha, print_msg=False, label=name)

    # add legend
    legtitle    = 'Pole uncertainties '
    legtitle   += num_to_sigma_string(n_std)

    if len(Ells) != 0:
        ncol = leg_ncol
        leg  = axleg.legend(Ells, names, handler_map=Handler_map,
                          loc='upper left', fontsize=12,
                          bbox_to_anchor=(-0.05, 0.85),
                          frameon=False, ncol=leg_ncol, columnspacing=1,
                          title=legtitle,
                          )
    else:
        handles, labels = ax1.get_legend_handles_labels()
        leg  = axleg.legend(handles, labels,
                          loc='lower left', fontsize=12,
                          title=legtitle,
                          )
    leg._legend_box.align = 'left'


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

        if isinstance(n_std, (list,np.ndarray)):
            for i in range(len(n_std)):
                ci = tweak_color(color, luminos=(i+1)/len(n_std))
                ell = draw_confidence_ellipse(ax2, x=pole.rotRate*MASY2DMY, y=pole.poleLat, cov=cov, n_std=n_std[i],
                                         color=ci, elp_lw=elp_lw, elp_alpha=elp_alpha, print_msg=False)
        else:
            ell = draw_confidence_ellipse(ax2, x=pole.rotRate*MASY2DMY, y=pole.poleLat, cov=cov, n_std=n_std,
                                         color=color, elp_lw=elp_lw, elp_alpha=elp_alpha, print_msg=False)

    ax2.tick_params(axis="both", direction="in", width=tickw, labeltop=True, labelbottom=False, labelleft=False, labelright=True)
    ax2.xaxis.set_ticks_position('both'); ax2.xaxis.set_label_position("top")
    ax2.yaxis.set_ticks_position('both'); ax2.yaxis.set_label_position("right")
    ax2.set_yticks(y_ticks, labels=y_ticklabels)
    ax2.xaxis.set_major_locator(locator)
    if lat_label is not None:
        ax2.set_ylabel(lat_label)
    if rot_label is not None:
        ax2.set_xlabel(rot_label)


    # axis 3 - {Lon, rate}
    for pole, color, name in zip(poles, colors, names):
        try:
            cov = pole.sph_cov_deg[np.ix_([1,2],[1,2])] # {lon,rate}
            cov[0,1] *= 1e6     # deg/yr     -> deg/Ma
            cov[1,0] *= 1e6     # deg/yr     -> deg/Ma
            cov[1,1] *= 1e12    # deg^2/yr^2 -> deg^2/Ma^2
        except:
            cov   = None

        if isinstance(n_std, (list,np.ndarray)):
            for i in range(len(n_std)):
                ci = tweak_color(color, luminos=(i+1)/len(n_std))
                ell = draw_confidence_ellipse(ax3, x=pole.poleLon, y=pole.rotRate*MASY2DMY, cov=cov, n_std=n_std[i],
                                         color=ci, elp_lw=elp_lw, elp_alpha=elp_alpha, print_msg=False)
        else:
            ell = draw_confidence_ellipse(ax3, x=pole.poleLon, y=pole.rotRate*MASY2DMY, cov=cov, n_std=n_std,
                                         color=color, elp_lw=elp_lw, elp_alpha=elp_alpha, print_msg=False)

    ax3.tick_params(axis="both", direction="in", width=tickw, labeltop=False, labelbottom=True, labelleft=True, labelright=False)
    ax3.xaxis.set_ticks_position('both'); ax3.xaxis.set_label_position("bottom")
    ax3.yaxis.set_ticks_position('both'); ax3.yaxis.set_label_position("left")
    ax3.set_xticks(x_ticks, labels=x_ticklabels)
    ax3.yaxis.set_major_locator(locator)
    if lon_label is not None:
        ax3.set_xlabel(lon_label)
    if rot_label is not None:
        ax3.set_ylabel(rot_label)


    # adjust axes
    ax1.tick_params(axis="both", direction="in", width=tickw, labeltop=False, labelbottom=False, labelleft=False, labelright=False)
    plt.subplots_adjust(hspace=.05)
    plt.subplots_adjust(wspace=.05)
    [ll.set_linewidth(tickw) for ll in ax1.spines.values()]
    [ll.set_linewidth(tickw) for ll in ax2.spines.values()]
    [ll.set_linewidth(tickw) for ll in ax3.spines.values()]

    # ccrs feature to axis 1
    kwargs = update_kwargs(kwargs)
    ax1.add_feature(cfeature.COASTLINE, ec='k'   , lw=0.6)
    ax1.add_feature(cfeature.BORDERS  , ec='gray', lw=0.1)
    ax1.add_feature(cfeature.OCEAN    , fc=kwargs['c_ocean'])
    ax1.add_feature(cfeature.LAKES    , fc=kwargs['c_ocean'])
    ax1.add_feature(cfeature.LAND     , fc=kwargs['c_land'])

    return fig, (ax1, ax2, ax3), leg


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
    if plate_name is None:
        return outlines

    else:
        if plate_name in outlines.keys():
            # convert list into shapely polygon object for easy use
            outline = geometry.Polygon(outlines[plate_name])
            return outline

        else:
            if pmm_dict and plate_name in pmm_dict.keys():
                plate_abbrev = pmm_dict[plate_name].Abbrev
            else:
                plate_abbrev = '-'
            print(f'Can NOT find plate "{plate_name}" ({plate_abbrev}) in file: {plate_boundary_file}!')
            return None


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
    kwargs['c_ocean']     = kwargs.get('c_ocean'    , 'w'        )
    kwargs['c_land']      = kwargs.get('c_land'     , 'gainsboro')
    kwargs['c_plate']     = kwargs.get('c_plate'    , 'w'        )
    kwargs['lw_coast']    = kwargs.get('lw_coast'   , 0.3        )
    kwargs['lw_border']   = kwargs.get('lw_border'  , 0.15       )
    kwargs['lw_pbond']    = kwargs.get('lw_pbond'   , 1.4        )
    kwargs['ls_pbond']    = kwargs.get('ls_pbond'   , '--'       )
    kwargs['lc_pbond']    = kwargs.get('lc_pbond'   , 'k'        )
    kwargs['alpha_plate'] = kwargs.get('alpha_plate', 0.4        )
    kwargs['grid_ls']     = kwargs.get('grid_ls'    , '--'       )
    kwargs['grid_lw']     = kwargs.get('grid_lw'    , 0.3        )
    kwargs['grid_lc']     = kwargs.get('grid_lc'    , 'gray'     )
    kwargs['grid_dx']     = kwargs.get('grid_dx'    , 10.        )
    kwargs['grid_dy']     = kwargs.get('grid_dy'    , 10.        )
    kwargs['qnum']        = kwargs.get('qnum'       , 6          )
    kwargs['font_size']   = kwargs.get('font_size'  , 12         )

    # point of interest
    kwargs['pts_lalo']    = kwargs.get('pts_lalo'   , None       )
    kwargs['pts_marker']  = kwargs.get('pts_marker' , '^'        )
    kwargs['pts_ms']      = kwargs.get('pts_ms'     , 20         )
    kwargs['pts_mfc']     = kwargs.get('pts_mfc'    , 'r'        )
    kwargs['pts_mec']     = kwargs.get('pts_mec'    , 'k'        )
    kwargs['pts_mew']     = kwargs.get('pts_mew'    , 1          )
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
                ax               - matplotlib figure and axis
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
        fig, ax = plt.subplots(figsize=figsize, layout="constrained", subplot_kw=dict(projection=map_proj))

    # map extent & grids
    if map_style == 'globe':
        # make the map global rather than have it zoom in to the extents of any plotted data
        ax = update_axis_projection(ax, map_proj)
        ax.set_global()
        if float(kwargs['grid_lw']) > 0:
            dx = float(kwargs['grid_dx'])
            dy = float(kwargs['grid_dy'])
            gl = ax.gridlines(crs=ccrs.PlateCarree(),
                            color=kwargs['grid_lc'],
                            linestyle=kwargs['grid_ls'],
                            linewidth=kwargs['grid_lw'],
                            xlocs=np.arange(-180,180,dx),
                            ylocs=np.arange(-80,81,dy),
                            )
    elif map_style == 'platecarree':
        ax.set_extent(extent, crs=map_proj)
        if float(kwargs['grid_lw']) > 0:
            locator = mticker.MaxNLocator(nbins=5, steps=[1,2,4,5,10], prune='both')
            gl = ax.gridlines(crs=ccrs.PlateCarree(),
                            color=kwargs['grid_lc'],
                            draw_labels=True,
                            linestyle=kwargs['grid_ls'],
                            linewidth=kwargs['grid_lw'],
                            xlocs=locator,
                            ylocs=locator,
                            xformatter=LongitudeFormatter(),
                            yformatter=LatitudeFormatter(),
                            )
            gl.top_labels  = False
            gl.left_labels = False

            # Get the tick locations
            x_ticks = gl.xlocator.tick_values(*ax.get_xlim())
            y_ticks = gl.ylocator.tick_values(*ax.get_ylim())
            ax.set_xticks(x_ticks, crs=ccrs.PlateCarree())
            ax.set_yticks(y_ticks, crs=ccrs.PlateCarree())
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.tick_params(axis="y",direction="in", pad=-32)
            ax.tick_params(axis="x",direction="in")
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.add_feature(cfeature.COASTLINE, color='k', lw=0.6)

    # cartopy features
    ax.add_feature(cfeature.COASTLINE, ec='k'   , lw=kwargs['lw_coast'])
    ax.add_feature(cfeature.BORDERS  , ec='gray', lw=kwargs['lw_border'])
    ax.add_feature(cfeature.OCEAN    , fc=kwargs['c_ocean'], zorder=0.8)
    ax.add_feature(cfeature.LAKES    , fc=kwargs['c_ocean'], zorder=0.8)
    ax.add_feature(cfeature.LAND     , fc=kwargs['c_land'] , zorder=0.8)

    # plot plate boundary polygon
    if plate_boundary:
        poly_lats = np.array(plate_boundary.exterior.coords)[:, 0]
        poly_lons = np.array(plate_boundary.exterior.coords)[:, 1]
        # ccrs.Geodetic()
        ax.plot(poly_lons, poly_lats, color=kwargs['lc_pbond'], transform=ccrs.PlateCarree(), linewidth=kwargs['lw_pbond'], linestyle=kwargs['ls_pbond'])
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
                       center_lalo      : tuple                | None = None,  # deg
                       satellite_height : float                | None = 1e6,   # meter
                       extent           : list  | tuple        | None = None,
                       ax               : matplotlib.axes.Axes | None = None,
                       figsize          : list  | tuple        | None = [5, 5],
                       epole_obj        : object               | None = None,
                       compare_duel     : bool                 | None = None,
                       orb              : bool                 | None = True,
                       helmert          : dict                 | None = False,
                       Ve               : list                 | None = None,   # meter/yr
                       Vn               : list                 | None = None,   # meter/yr
                       Lats             : list                 | None = None,   # deg
                       Lons             : list                 | None = None,   # deg
                       unit             : str                  | None = 'mm',   # mm
                       qunit            : float                | None = 50,     # 50 * 'unit'
                       qscale           : float                | None = 1.0,
                       qwidth           : float                | None = .0075,
                       qcolor           : list[str]   | str    | None = 'coral',
                       qalpha           : list[float] | float  | None = 1.,
                       qname            : list[str]   | str    | None = None,
                       quiverlegend     : bool                 | None = True,
                       **kwargs,
                       ) -> tuple :


    kwargs = update_kwargs(kwargs)

    # multi poles plots:
    N = 0  # default no show any pole/velo
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


    # save input v
    ve_in = np.array(Ve) if Ve is not None else None
    vn_in = np.array(Vn) if Vn is not None else None


    if all(_inp is not None for _inp in (Ve, Vn, Lats, Lons)) and (len(Ve)==N):
        # VECTORS from : input locations and velocities
        print('plot input {lat, lon, ve, vn}')

    elif all(_inp is not None for _inp in (Lats, Lons)) and (epole_obj is not None):
        # VECTORS from : input locations and pole-predicted velocities
        print('plot {ve,vn} est from {lat,lon}')
        Ve   = []
        Vn   = []
        for j, (epole, helm) in enumerate(zip(epole_obj, helmert)):
            # calculate plate motion on sample points
            _ve, _vn = epole.get_velocity_enu(lat=Lats[j], lon=Lons[j], orb=orb, helmert=helm)[:2]
            Ve.append(_ve)
            Vn.append(_vn)

    elif epole_obj is not None:
        # VECTORS from : arbitrary gridded locations and pole-predicted velocities
        print('plot {ve,vn} est from some regular grids, qnum=', kwargs['qnum'])
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


    # check if there is a 3rd guy (i assume that is your GPS) to plot
    # if yes, keep the 1st and the 3rd guys for comparison
    if compare_duel is not None and ve_in is not None:
        if (len(compare_duel)==3):
            print(f'plot the first pole_pred {compare_duel[0]} and your input GPS {compare_duel[2]}')
            Ve    = [Ve[0]]    + [ve_in]
            Vn    = [Vn[0]]    + [vn_in]
            qname = [qname[0]] + [compare_duel[2]]
            Lats  = [Lats[0]] * 2
            Lons  = [Lons[0]] * 2


    if unit == 'mm':
        factor = 1e3
    elif unit == 'cm':
        factor = 1e2
    else:
        unit = 'meter'
        factor = 1.
    qscale *= factor


    # compare the first two pole
    if compare_duel is not None and len(epole_obj)>=2:
        rmse = ut.calc_wrms(Ve[0]-Ve[1])
        rmsn = ut.calc_wrms(Vn[0]-Vn[1])
        show_str = fr'$RMS_e=${rmse*1e3:.3f} mm/yr' + '\n' + fr'$RMS_n=${rmsn*1e3:.3f} mm/yr'
        ax.annotate(show_str, xy=(1,.97), xycoords='axes fraction', fontsize=12, annotation_clip=False, ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='black', boxstyle='round,pad=0.2'), zorder=20)


    # arrows
    Q, Qk = [], []
    if all(_inp is not None for _inp in (Ve, Vn, Lats, Lons)):

        # Zip the lists together
        zipped = list(zip(Ve, Vn, Lats, Lons, qcolor, qalpha, qname))

        # iterate over the reversed enumeration (plot arrows InSAR at front)
        for j, (ve, vn, lats, lons, qc, qa, qn) in enumerate(reversed(zipped)):
            # scale the vector unit
            ve = np.array(ve) * factor
            vn = np.array(vn) * factor

            if True:
                # correcting for "East" further toward polar region; re-normalize ve, vn
                norm = np.sqrt(ve**2 + vn**2)
                ve /= np.cos(np.deg2rad(lats))
                renorm = np.sqrt(ve**2 + vn**2)/norm
                ve /= renorm
                vn /= renorm

            # ---------- plot inplate vectors --------------
            if 'qec' in kwargs:
                qec = kwargs.get('qec', 'k')
                qlw = kwargs.get('qlw', 1)
                q = ax.quiver(lons, lats, ve, vn,
                            scale=qscale, width=qwidth, facecolor=qc,
                            edgecolor=qec, lw=qlw,
                            angles="xy", alpha=qa, zorder=3,
                            transform=ccrs.PlateCarree(),
                            )
            else:
                q = ax.quiver(lons, lats, ve, vn,
                            scale=qscale, width=qwidth, color=qc,
                            angles="xy", alpha=qa, zorder=3,
                            transform=ccrs.PlateCarree(),
                            )

            if 'qkX' not in kwargs:
                X, Y, qkstep = None, None, None
            else:
                X, Y, qkstep = kwargs['qkX'], kwargs['qkY'], kwargs['qkstep']

            if any(var is None for var in [X, Y, qkstep]):
                X, Y, qkstep = 0.4, -0.158, -0.076

            if quiverlegend:
                qk = ax.quiverkey(q, X=X, Y=Y+qkstep*(j+1), U=-qunit,
                                label=qn, coordinates='axes',
                                labelpos='E', labelsep=0.5,
                                fontproperties={'size':kwargs['font_size']},
                                )
                Qk.append(qk)
            Q.append(q)

        if quiverlegend:
            ax.text(X, Y, f'Plate motion ({qunit} {unit}/yr)',
                    clip_on=False, transform=ax.transAxes)
    #-----------------------------------------------------------

    return ax, Q, Qk