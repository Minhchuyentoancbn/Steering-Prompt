import os
import os.path
import hashlib
import errno
from torchvision import transforms


dataset_stats = {
    'CIFAR100': {
        'mean': (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
        'std' : (0.2673342858792409, 0.25643846291708816, 0.2761504713256834),
        'size' : 32
    }
}

# transformations
def get_transform(dataset='CIFAR100', phase='test'):
    transform_list = []

    # get mean and std
    dset_mean = (0.0, 0.0, 0.0) # dataset_stats[dataset]['mean']
    dset_std = (1.0, 1.0, 1.0) # dataset_stats[dataset]['std'] 


    if phase == 'train':
        transform_list.extend([
            transforms.RandomResizedCrop(224),  # Random resized crop to 224
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),  # Random color jitter
            transforms.RandomGrayscale(p=0.2),  # Random grayscale
            transforms.RandomApply([
                transforms.GaussianBlur(224 // 20 * 2 + 1 , sigma=(0.1, 2.0))
            ], p=0.1),  # Random gaussian blur
            transforms.RandomSolarize(128, 0.2),  # Random solarization
            transforms.ToTensor(), # Transform to tensor
            transforms.Normalize(dset_mean, dset_std), # Normalize
        ])
    else:
        transform_list.extend([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(dset_mean, dset_std),
        ])

    return transforms.Compose(transform_list)


def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    import urllib
    # from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)