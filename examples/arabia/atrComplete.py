#!/usr/bin/env python
import h5py
import argparse
import numpy as np

from mintpy.utils import utils as ut
from mintpy.utils import readfile, writefile


def cmd_parser():
    parser = argparse.ArgumentParser(description='Geometry file parser')
    parser.add_argument('-i', '--input', required=True, help='Input file path')
    parser.add_argument('-o', '--output', default=None, help='Output file path')
    parser.add_argument('-f', '--force', action='store_true', help='Force to overwrite datasets!')
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input
    return args


def latlon_from_meta(meta):
    length, width = int(meta['LENGTH']), int(meta['WIDTH'])
    y0 = float(meta['Y_FIRST'])
    x0 = float(meta['X_FIRST'])
    dy = float(meta['Y_STEP'])
    dx = float(meta['X_STEP'])

    lats = np.linspace(y0, y0 + (length - 1) * dy, length)
    lons = np.linspace(x0, x0 +  (width - 1) * dx,  width)
    Lons, Lats = np.meshgrid(lons, lats)

    return length, width, Lats, Lons


def main(args):
    infile = args.input
    outfile = args.output

    # Read attributes from the input file
    atr = readfile.read_attribute(infile)

    # compute slantRangeDistance
    inc_angle = readfile.read(infile, datasetName='incidenceAngle')[0]
    range_dist = ut.incidence_angle2slant_range_distance(atr, inc_angle)

    # Generate latitude and longitude arrays
    length, width, Lats, Lons = latlon_from_meta(atr)
    print(np.sum(np.isnan(inc_angle)), np.sum(inc_angle==0))
    Lats[np.isnan(inc_angle)] = np.nan
    Lons[np.isnan(inc_angle)] = np.nan

    # Allocate save dict
    dataDict = {}


    # Read existing datasets from the input file
    with h5py.File(infile, 'r') as f:
        dsNames = [i for i in f.keys() if isinstance(f[i], h5py.Dataset) and f[i].shape[-2:] == (length, width)]
        print("Existing datasets:", dsNames)

        for ds in dsNames:
            dataDict[ds] = np.array(f[ds])  # Convert to numpy array

        if not args.force:
            if 'latitude' not in dsNames:
                dataDict['latitude'] = Lats

            if 'longitude' not in dsNames:
                dataDict['longitude'] = Lons

            if np.nanmin(dataDict['slantRangeDistance']) == np.nanmax(dataDict['slantRangeDistance']):
                dataDict['slantRangeDistance'] = range_dist

        else:
                dataDict['latitude'] = Lats
                dataDict['longitude'] = Lons
                dataDict['slantRangeDistance'] = range_dist

    # Write to the output file
    writefile.write(dataDict, outfile, ref_file=infile)

    print("Datasets written to", outfile)


if __name__ == '__main__':
    args = cmd_parser()
    main(args)
