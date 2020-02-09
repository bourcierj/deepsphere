import torch
from dataset import ModelNet40Dataset
from torch.utils.data import DataLoader

from tqdm import tqdm
from time import time

if __name__ == '__main__':

    root = './data/ModelNet40'
    # Process training set
    train_set = ModelNet40Dataset(root, 'train', nside=32, nfeat=6,
                                  cache=True, verbose=False)

    def my_collate(batch):
        data, target = tuple(map(list, zip(*batch)))
        return (data, target)
    train_loader = DataLoader(train_set, batch_size=4, shuffle=False,
                              collate_fn=my_collate,
                              num_workers=torch.multiprocessing.cpu_count())

    # for idx in tqdm(range(len(train_set))):
    #     img, target = train_set[idx]
    #     print('Img data: shape:', img.shape)
    #     print('Target: {} ({})'.format(target, train_set.classes[0]))
    #     if idx > 10:
    #         break
    # assert(all(train_set.is_cached[idx] for idx in range(12)))
    # assert(all(not train_set.is_cached[idx] for idx in range(12, len(train_set))))
    print("#--------- Training Set ---------#")
    n_batches_processed = 0
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        n_batches_processed += 1

    print('Done. Number of batches processed: {}'
          .format(n_batches_processed))

    # Assert all files have been processed
    is_cached = train_set._init_cached_flags()
    assert(all(is_cached))

    # Pass over the training set to see that it loads fast now
    print('Pass over dataset after processing:')
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        pass

    # Process testing set
    test_set = ModelNet40Dataset(root, 'test', nside=32, nfeat=6,
                                 cache=True, verbose=False)

    test_loader = DataLoader(test_set, batch_size=4, shuffle=False,
                             collate_fn=my_collate,
                             num_workers=torch.multiprocessing.cpu_count())

    print("#--------- Testing Set ---------#")
    n_batches_processed = 0
    for batch_idx, (data, target) in tqdm(enumerate(test_loader)):
        n_batches_processed += 1

    print('Done. Number of batches processed: {}'
          .format(n_batches_processed))

    # Assert all files have been processed
    is_cached = test_set._init_cached_flags()
    assert(all(is_cached))

    # Pass over the training set to see that it loads fast now
    print('Pass over dataset after processing:')
    for batch_idx, (data, target) in tqdm(enumerate(test_loader)):
        pass
