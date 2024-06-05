import pandas as pd
from satpy import Scene
import numpy as np
import cartopy.crs as ccrs
import cartopy
import rioxarray
import xarray as xr 
import satpy
from satpy import MultiScene
from satpy.multiscene import timeseries
from glob import glob
from tqdm import tqdm
from utils.etc import benchmark
import xarray
import gc
import logging
from tqdm.contrib.logging import logging_redirect_tqdm
import os
import time
from datetime import datetime, timedelta
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
from functools import partial
import zarr
import traceback



lon = np.arange(-7.775,28.975+0.05,0.05)
lat = np.arange(28.975,61.825+0.05,0.05)
grid_x, grid_y = np.mgrid[-7.775:28.975+0.05:0.05, 28.975:61.825+0.05:0.05]
output_chunks = {"data": -1, "time": 1, "x": -1, "y": -1}
grid_dataset = xr.Dataset(coords=dict(
        x=("x", grid_x[:,0]),
        y=("y", grid_y[0,:]))).rio.write_crs('wgs84')
 
NON_HRV_BANDS = [
    "IR_016",
    "IR_039",
    "IR_087",
    "IR_097",
    "IR_108",
    "IR_120",
    "IR_134",
    "VIS006",
    "VIS008",
    "WV_062",
    "WV_073",
]


valid_types = (str, np.ndarray, np.number, list, tuple)


def write_zarr(ds, store, first=False):
    if first:
        ds.to_zarr(store, mode='w',)
    else:
        ds.to_zarr(store, append_dim='time')

def get_dataarray(filename):

    retry = 3
    for i in range(3):
        try:
            with satpy.config.set(cache_dir='/capstor/scratch/cscs/kschuurm/cache', tmp_dir='/capstor/scratch/cscs/kschuurm/tmp', data_dir='/capstor/scratch/cscs/kschuurm/data'):
                scn = Scene(reader="seviri_l1b_native", filenames=[filename])
                
                scn.load(NON_HRV_BANDS)

                for channel in scn.wishlist:
                    scn[channel] = scn[channel].drop_vars("acq_time", errors="ignore")
                scn_dataset = scn.to_xarray_dataset()
                scn_dataset = scn_dataset[NON_HRV_BANDS].rio.write_crs(scn_dataset.crs.item())

                for variable in scn_dataset.variables.values():
                    for k, v in variable.attrs.items():
                        if not isinstance(v, valid_types) or isinstance(v, bool):
                            variable.attrs[k] = str(v)

                scn_dataset = scn_dataset.to_dataarray()
                scn_dataset['time'] = scn.start_time
                
                reprojected = scn_dataset.rio.reproject_match(grid_dataset, nodata=np.nan)
                reprojected['crs'] = str(reprojected.crs.item())
                reprojected.attrs = {}
                reprojected = reprojected.expand_dims(dim='time', axis=1).rename({'variable':'channel'}).to_dataset(name='channel_data').astype(np.float16)
                
                reprojected.time.encoding['units'] = 'minutes since 2000-01-01'
                reprojected.time.encoding['dtype'] = np.int64
                reprojected.time.attrs = {'long_name': 'nominal start time scan'}
                reprojected['channel_data'].encoding.clear()

                if reprojected.channel_data.isnull().sum().item() > 0:
                    print('Scn:', scn.start_time, ', contains nans')
                    with open('failed_files.txt', 'a') as f:
                        f.write(f'{filename}\n')
                    return None
                
                min_q = reprojected.channel_data.quantile(0.001, dim=['x', 'y', 'time'])
                max_q = reprojected.channel_data.quantile(0.999, dim=['x', 'y', 'time'])


                start_time = datetime.strftime(scn.start_time, '%Y%m%d%H%M%S')
                with open('quantiles_min.txt', 'a') as f:
                    s = ';'.join([str(round(x,4)) for x in min_q.values]) 
                    f.write(f'{start_time};{s}\n')
                with open('quantiles_max.txt', 'a') as f:
                    s = ';'.join([str(round(x,4)) for x in max_q.values]) 
                    f.write(f'{start_time};{ s }\n')

                reprojected['time'] = reprojected.time.chunk({'time':-1})
                reprojected['channel_data'] = reprojected.channel_data.chunk({'time':1, 'channel':-1, 'y':-1, 'x':-1})

                return reprojected
        except Exception as e:
            if i < retry-1:
                print(filename, 'retry', i, e)
            else:
                print(filename, 'failed', e)
                with open('failed_files.txt', 'a') as f:
                    f.write(f'{filename}\n')

                print(traceback.format_exc())
                return None
        else:
            break


from argparse import ArgumentParser

if __name__ == '__main__':
    
    argparse = ArgumentParser()
    argparse.add_argument('-y', '--year', type=str)

    args = argparse.parse_args()


    YEAR = args.year
    zarr_fn = f'/capstor/scratch/cscs/kschuurm/ZARR/SEVIRI_FULLDISK_{YEAR}.zarr'

    print(YEAR, zarr_fn)


    fns = glob(f'/capstor/scratch/cscs/kschuurm/DATA/EO:EUM:DAT:MSG:HRSEVIRI/HRSEVIRI{YEAR}/*.nat')
    def sortfunc(nm):
        return nm.replace('-NA.nat','').split('-')[-1]
    fns.sort(key = sortfunc)


    fns_times = [sortfunc(x).split('.')[0] for x in fns]
    fns_times = [datetime.strptime(x, '%Y%m%d%H%M%S') - timedelta(minutes=12) for x in fns_times]
    fns_times = pd.DatetimeIndex(fns_times).round('15min')
    fns_t = [(t,fn) for t, fn in zip(fns_times, fns)]

    if os.path.exists(zarr_fn):
        existing_store = xarray.open_zarr(zarr_fn)
        existing_store_times = existing_store.time.values
        existing_store.close()
        print('files present', len(fns_t))
        fns_t = [(t,fn) for t, fn in fns_t if t not in existing_store_times]
        print('files not yet processed', len(fns_t))
    else:
        existing_store_times = None


    with zarr.DirectoryStore(zarr_fn) as store:

        if existing_store_times is None:
            t, fn = fns_t.pop(0)
            ds = get_dataarray(fn)
            if ds is not None:
                write_zarr(ds, store, first=True)
            else:
                print('first file failed')
                ds = get_dataarray(fns_t.pop(0)[1])
                write_zarr(ds, store, first=True)


        with ThreadPool(10) as p:
            output = p.imap(get_dataarray, [x[1] for x in fns_t], chunksize=1)

            for o in tqdm(output, total=len(fns_t), smoothing=0):
                if o is not None:
                    write_zarr(o, store)