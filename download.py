from functools import partial
import eumdac
import shutil
import datetime
import numpy as np
import zipfile
import os
from multiprocessing import Pool
import multiprocessing
import time
import sys
import gc
import pandas as pd
from tqdm import tqdm
from download import initialize_download, check_downloaded_files, download_product
import calendar
from datetime import datetime

def initialize_download():
    credentials = ('', '')
    token = eumdac.AccessToken(credentials)
    datastore = eumdac.DataStore(token)
    print('token created')
    selected_collection = datastore.get_collection('EO:EUM:DAT:MSG:HRSEVIRI')
    return selected_collection
 
def check_downloaded_files(MY_PATH, DATASET_ID, SAVE_PATH):
    if DATASET_ID not in os.listdir(MY_PATH):
        os.mkdir(SAVE_PATH)
        print('folder created')
    else:
        print('folder already present')
    downloaded_products = set(os.listdir(SAVE_PATH))
    if not downloaded_products:
        print('folder is empty')
    else:
        print('with {} files'.format(len(downloaded_products)))
    return downloaded_products
 
def download_product(product, SAVE_PATH):
    retry = 3
    for i in range(retry):
        try:
            with product.open() as fsrc, open(SAVE_PATH+fsrc.name, mode='wb') as fdst:
                data_file = fsrc.name
                shutil.copyfileobj(fsrc, fdst)
                
            with zipfile.ZipFile(SAVE_PATH+data_file, 'r') as zobj:
                zobj.extractall(SAVE_PATH)
        except Exception as e:

            if isinstance(e, eumdac.product.ProductError):
                data_file='None'
                
            if i < retry-1:
                print('retry', i, data_file, e)
            else:
                print('failed', data_file, e)
                with open('failed_downloads.txt', 'a') as f:
                    f.write(str(data_file) + f': {e}' '\n')
        else:
            break
        finally:
            if os.path.exists(SAVE_PATH+data_file):
                os.remove(SAVE_PATH+data_file)


def main():
    selected_collection = initialize_download()
    
    years = [2017, 2018, 2019, 2020, 2021, 2022]  

    dt_ls = [[datetime(year, month, 1), datetime(year, month, calendar.monthrange(year, month)[1], 23,59)] for year in years for month in range(1, 13)]

    for start, end in tqdm(dt_ls, desc='months'):
        
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)

        print('download:', start, end)
        YEAR = start.year
        DATASET_ID = 'HRSEVIRI{}'.format(YEAR)
        MY_PATH = '/capstor/scratch/cscs/kschuurm/DATA/EO:EUM:DAT:MSG:HRSEVIRI/'
        SAVE_PATH = MY_PATH + '{}/'.format(DATASET_ID)
        downloaded_products = check_downloaded_files(MY_PATH, DATASET_ID, SAVE_PATH)
        products = selected_collection.search(dtstart=start, dtend=end)
        
        # find products to download
        products_to_download = []
        for i, product in enumerate(products):
            if str(product)+'.nat' not in downloaded_products:
                products_to_download.append(product)
        print('{} files to download'.format(len(products_to_download)))
        
        # download products
        if products_to_download:
            for j in range(0, len(products_to_download)+100, 100):
                part = partial(download_product, SAVE_PATH=SAVE_PATH)
                with Pool(10) as p:   # EUMETSAT only allows 10 concurrent connections
                    p.map(part, products_to_download[j:j+100])
                gc.collect()



if __name__ == '__main__':
    main()