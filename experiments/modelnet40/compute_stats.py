from glob import glob
import os
from tqdm import tqdm
import numpy as np


class StatsRecorder:
    def __init__(self, data=None):
        """Recorder for mean and standard deviation statistics
        Args:
            data: (ndarray, shape (num_observations, num_dimensions)):
                Initial sample data
        """
        if data is not None:
            data = np.atleast_2d(data)
            self.mean = data.mean(axis=0)
            self.std  = data.std(axis=0)
            self.nobservations = data.shape[0]
            self.ndimensions = data.shape[1]
        else:
            self.nobservations = 0

    def update(self, data):
        """Update current stats with data
        Args:
            data: (ndarray, shape (num_observations, num_dimensions)):
                New sample data
        """
        if self.nobservations == 0:
            self.__init__(data)
        else:
            data = np.atleast_2d(data)
            if data.shape[1] != self.ndimensions:
                raise ValueError("Data dims don't match prev observations.")

            newmean = data.mean(axis=0)
            newstd = data.std(axis=0)

            m = self.nobservations * 1.0
            n = data.shape[0]
            tmp = self.mean

            self.mean = m/(m+n)*tmp + n/(m+n)*newmean
            self.std  = m/(m+n)*self.std**2 + n/(m+n)*newstd**2 +\
                        m*n/(m+n)**2 * (tmp - newmean)**2
            self.std  = np.sqrt(self.std)

            self.nobservations += n

def compute_stats(data_dir):
    files = sorted(glob(os.path.join(data_dir, "*.npy")))
    sr = StatsRecorder()
    for f in tqdm(files):
        data = np.load(f)
        sr.update(data)
    print(sr.mean)
    print(sr.std)
    return sr.mean, sr.std

def compute_stats_dataset(dataset):
    processed_dir = dataset.processed_dir
    mean, std = compute_stats(processed_dir)
    return mean, std

def normalize_data(mean, std, data_dir):
    files = sorted(glob(os.path.join(data_dir, "*.npy")))
    for f in tqdm(files):
        data = np.load(f)
        data_normalized = (data-mean)/std
        new_f = f.replace("b64", "sp5")
        np.save(new_f, data_normalized)

if __name__ =='__main__':

    from dataset import ModelNet40Dataset

    root = './data/ModelNet40'
    experiment = 'deepsphere__healpix_nside_32'

    # Compute stats for training set
    train_set = ModelNet40Dataset(root, 'train', nside=32, nfeat=6,
                                  experiment=experiment, cache=True, verbose=False)
    # mean, std = compute_stats()
    print("Train:")
    mean, std = compute_stats_dataset(train_set)
    print("Mean: {}\nStd: {}".format(mean, std))

    savepath = './experiments/modelnet40/stats'
    if not os.path.exists(savepath):
        os.makedirs(savepath, exist_ok=True)
    with open(savepath + experiment +'_train_mean.txt', 'w') as mean_fp:
        mean_fp.write(str(mean.tolist()))
    with open(savepath + experiment+'_train_std.txt', 'w') as std_fp:
        std_fp.write(str(std.tolist()))

    # Compute stats for testing set
    test_set = ModelNet40Dataset(root, 'test', nside=32, nfeat=6,
                                 experiment=experiment, cache=True, verbose=False)
    print("Test:")
    mean, std = compute_stats_dataset(test_set)
    print("Mean: {}\nStd: {}".format(mean, std))
    with open(savepath + experiment +'_test_mean.txt', 'w') as mean_fp:
        mean_fp.write(str(mean.tolist()))
    with open(savepath + experiment +'_test_std.txt', 'w') as std_fp:
        std_fp.write(str(std.tolist()))
