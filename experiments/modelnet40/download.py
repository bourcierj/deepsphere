import sys
import os
import glob
from tqdm import tqdm
# from load_MN40 import ModelNet40DatasetCache, compute_mean_std

import argparse
# data_path = '../../data/shrec17/'

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='./data/')
    return parser.parse_args()

def _check_exists(path):
    files = glob.glob(os.path.join(path, "**/*.off"), recursive=True)
    return len(files) > 0


def _gen_bar_updater():
        pbar = tqdm(total=None)

        def bar_update(count, block_size, total_size):
            if pbar.total is None and total_size:
                pbar.total = total_size
            progress_bytes = count * block_size
            pbar.update(progress_bytes - pbar.n)

        return bar_update

def _download(url, path):
    from six.moves import urllib
    filename = os.path.basename(os.path.normpath(url))
    file_path = os.path.join(path, filename)
    if os.path.exists(file_path):
        return file_path

    print('Downloading ' + url + ' to ' + path)
    urllib.request.urlretrieve(
        url, file_path,
        reporthook=_gen_bar_updater()
    )
    print("Downloaded.")
    return file_path

def _unzip(file_path, path=None, delete=False):
    import zipfile
    if not path:
        path, _ = os.path.splitext(file_path)

    print('Unzipping', file_path)
    zip_ref = zipfile.ZipFile(file_path, 'r')
    zip_ref.extractall(path)
    zip_ref.close()
    if delete:
        os.unlink(file_path)
    print('Done.', file_path)


def download_modelnet40(path):
    if _check_exists(path):
        return
    elif not os.path.exists(path):
        os.makedirs(path)

    url = 'http://modelnet.cs.princeton.edu/ModelNet40.zip'
    file_path = _download(url, path)
    _unzip(file_path)

if __name__ == '__main__':

    args = parse_arguments()

    download_modelnet40(args.data_path)

    # if args.compute_stats:
    #     # Construct the datasets
    #     # get preprocessing and experiments parameters
    #     train_kwargs = dict()  #@TODO
    #     test_kwargs = dict()  #@TODO

    #     train_set = ModelNet40Dataset(args.data_path, 'train', **train_kwargs)
    #     val_set = ModelNet40Dataset(args.data_path, 'val', **train_kwargs)
    #     test_set = ModelNet40Dataset(args.data_path, 'test', **test_kwargs)

    #     # Compute statistics on training set
    #     mean, std = compute_mean_std(train_set, 'train', data_path, **train_kwargs)
    #     print(mean, std)
