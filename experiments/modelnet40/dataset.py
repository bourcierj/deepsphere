import torch
from torch.utils.data import Dataset
# from torch_geometric.data import Dataset as GraphDataset
# from torch_geometric.data import DataLoader

"""Load dataset ModelNet40 and project it a sphere with Healpix sampling
"""
import glob
import os
import re
from tqdm import tqdm
import pickle as pkl
from itertools import cycle
from itertools import zip_longest
import numpy as np
import trimesh
import healpy as hp


def rotmat(a, b, c, hom_coord=False):
    """
    Create a rotation matrix with an optional fourth homogeneous coordinate.
    Apply to mesh using mesh.apply_transform(rotmat(a,b,c, True))
    Args:
        a, b, c (floats): ZYZ-Euler angles
    """
    def z(a):
        return np.array([[np.cos(a), np.sin(a), 0, 0],
                         [-np.sin(a), np.cos(a), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    def y(a):
        return np.array([[np.cos(a), 0, np.sin(a), 0],
                         [0, 1, 0, 0],
                         [-np.sin(a), 0, np.cos(a), 0],
                         [0, 0, 0, 1]])

    r = z(a).dot(y(b)).dot(z(c))
    if hom_coord:
        return r
    else:
        return r[:3, :3]


def make_sgrid(sampling='healpix', *args, **kwargs):
    """
    Return a Healpix, Equiangular, or Icosahedral sampling coordinates.
    Args:
        type (str, one of 'healpix', 'equiangular', 'icosahedral'): the type of sampling
    """
    if sampling != 'healpix': #TOIMPLEMENT
        raise NotImplementedError()
    else:
        nside = kwargs['nside']
        npix = 12 * nside**2
        x, y, z = hp.pix2vec(nside, range(npix), nest=True)
        coords = np.vstack([x, y, z]).transpose()
        coords = np.asarray(coords, dtype=np.float32)
    # R = rotmat(alpha, beta, gamma, hom_coord=False)
    # grid = np.einsum('ij,nj->ni', R, coords)
    sgrid = coords
    return sgrid

def render_model(mesh, sgrid, outside=False, multiple=False):
    """
    Render mesh on the sampling grid.
    """
    # Cast rays
    # triangle_indices = mesh.ray.intersects_first(ray_origins=sgrid, ray_directions=-sgrid)
    if outside:
        index_tri, index_ray, loc = mesh.ray.intersects_id(
            ray_origins=(sgrid-sgrid), ray_directions=sgrid, multiple_hits=multiple, return_locations=True)
    else:
        index_tri, index_ray, loc = mesh.ray.intersects_id(
            ray_origins=sgrid, ray_directions=-sgrid, multiple_hits=multiple, return_locations=True)
    loc = loc.reshape((-1, 3))  # fix bug if loc is empty

    if multiple:
        grid_hits = sgrid[index_ray]
        if outside:
            dist = np.linalg.norm(loc, axis=-1)
        else:
            dist = np.linalg.norm(grid_hits - loc, axis=-1)
        dist_im = np.ones((sgrid.shape[0],3))*-1
        for index in range(np.max(index_ray)+1):
            for i, ind in enumerate(np.where(index_ray==index)[0]):
                if dist[ind] > 1:
                    continue
                try:
                    dist_im[index, i] = dist[ind]
                except:
                    pass
        return dist_im

        # max_index = np.argsort(index_ray)[1]
        # s=np.sort(index_ray)
        # print(s[:-1][s[1:] == s[:-1]])
        # index_tri_mult, index_mult, loc_mult = index_tri[max_index:], index_ray[max_index:], loc[max_index:]
        # index_tri, index_ray, loc = index_tri[:max_index], index_ray[:max_index], loc[:max_index]

    # Each ray is in 1-to-1 correspondence with a grid point. Find the position of these points
    grid_hits = sgrid[index_ray]
    grid_hits_normalized = grid_hits / np.linalg.norm(grid_hits, axis=1, keepdims=True)

    # Compute the distance from the grid points to the intersection pionts
    if outside:
        dist = np.linalg.norm(loc, axis=-1)
    else:
        dist = np.linalg.norm(grid_hits - loc, axis=-1)

    # For each intersection, look up the normal of the triangle that was hit
    normals = mesh.face_normals[index_tri]
    normalized_normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    # Construct spherical images
    dist_im = np.ones(sgrid.shape[0])
    dist_im[index_ray] = dist
    # dist_im = dist_im.reshape(theta.shape)

    n_dot_ray_im = np.zeros(sgrid.shape[0])
    n_dot_ray_im[index_ray] = np.einsum("ij,ij->i", normalized_normals,
                                        grid_hits_normalized)

    nx, ny, nz = normalized_normals[:, 0], normalized_normals[:, 1], \
        normalized_normals[:, 2]
    gx, gy, gz = grid_hits_normalized[:, 0], grid_hits_normalized[:, 1], \
        grid_hits_normalized[:, 2]
    wedge_norm = np.sqrt((nx * gy - ny * gx) ** 2 + (nx * gz - nz * gx) ** 2
                         + (ny * gz - nz * gy) ** 2)
    n_wedge_ray_im = np.zeros(sgrid.shape[0])
    n_wedge_ray_im[index_ray] = wedge_norm

    # Combine channels to construct final image
    im = np.stack((dist_im, n_dot_ray_im, n_wedge_ray_im), axis=0)

    return im


def random_rotmat(a=None, z=None, c=None):
    """Random rotation matrix"""
    if a is None:
        a = np.random.rand() * 2 * np.pi
    if z is None:
        z = np.arccos(np.random.rand() * 2 - 1)
    if c is None:
        c = np.random.rand() * 2 * np.pi
    rot = rotmat(a, z, c, True)
    return rot

def ToMesh(path, rot=False, tr=0.):
    """
    Load a mesh file to mesh object with Trimesh
    Args:
        path (str): Path to the mesh file
        rot (bool): Random rotation
        tr (float): Random translation, amound of translation max vector
    """
    mesh = trimesh.load_mesh(path)
    mesh.remove_degenerate_faces()
    mesh.fix_normals()
    mesh.fill_holes()
    mesh.remove_duplicate_faces()
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()

    mesh.apply_translation(-mesh.centroid)
    # Normalize mesh
    r = np.max(np.linalg.norm(mesh.vertices, axis=-1))
    mesh.apply_scale(1 / r)

    if tr > 0:
        tr = np.random.rand() * tr
        rot_R = random_rotmat()
        mesh.apply_transform(rot_R)
        mesh.apply_translation([tr, 0, 0])

        if not rot:
            mesh.apply_transform(rot_R.T)

    if rot:
        mesh.apply_transform(random_rotmat())
    # Normalize mesh
    r = np.max(np.linalg.norm(mesh.vertices, axis=-1))
    mesh.apply_scale(0.99 / r)
    # mesh.remove_degenerate_faces()
    mesh.fix_normals()
    # mesh.fill_holes()
    # mesh.remove_duplicate_faces()
    # mesh.remove_infinite_values()
    # mesh.remove_unreferenced_vertices()

    return mesh

def ProjectOnSphere(mesh, nside, outside=False, multiple=False):
    """Project a mesh object onto the sphere
    Args:
        nside (int): nside parameter for Healpix, power of 2
        mesh: mesh object
        outside (either None, 'equator', 'pole' or 'both'): orient the mesh on sphere.
            None means keep the default orientation
        multiple: #@WHAT?
    """
    if outside == 'equator':
        rot = rotmat(0,np.arccos(1-np.random.rand()*0.3)-np.pi/8,0, True)
        mesh.apply_translation([1.5, 0, 0])
        mesh.apply_transform(rot)
    if outside == 'pole':
        rot = rotmat(np.random.rand()*np.pi/4-np.pi/8,np.pi/2,0, True)
        mesh.apply_translation([1.5, 0, 0])
        mesh.apply_transform(rot.T)
    if outside == 'both':
        rot = rotmat(0,-np.random.rand()*np.pi/2,0, True)
        mesh.apply_translation([1.5, 0, 0])
        mesh.apply_transform(rot)

    sgrid = make_sgrid(nside=nside, outside=outside, multiple=multiple)
    im = render_model(mesh, sgrid, outside=outside, multiple=multiple)
    if multiple:
        return im.astype(np.float32)
    npix = sgrid.shape[0]
    im = im.reshape(3, npix)

    from scipy.spatial.qhull import QhullError
    try:
        convex_hull = mesh.convex_hull
    except QhullError:
        convex_hull = mesh

    hull_im = render_model(convex_hull, sgrid, outside=outside, multiple=multiple)
    hull_im = hull_im.reshape(3, npix)
    # Concatenate convex hull to shape features
    im = np.concatenate((im, hull_im), axis=0)
    assert len(im) == 6  # must have 6 features
    im = im.astype(np.float32).T

    return im  # must be (npix x 6)

def Transform(path, nside=32, rot=False, verbose=True):
    if verbose:
        print("Transform {}...".format(path))
    try:
        mesh = ToMesh(path, rot=rot, tr=0.)
        data = ProjectOnSphere(mesh, nside)
        return data
    except:
        print("Exception during transform of {}".format(path))
        raise


class ModelNet40Dataset(Dataset):
    """
    The ModelNet40 Dataset: https://modelnet.cs.princeton.edu/
    Allows preprocessing the OFF files and caching processed data in Numpy arrays
    for faster loading.
    This class is intented to be wrapped by a Torch Geometric Dataset.
    """
    def __init__(self, root='./data/ModelNet40', role='train', nside=32, nfeat=6,
                 nfile=None, experiment='deepsphere__healpix_nside_32',
                 fix=False, cache=True, verbose=True):
        self.experiment = experiment
        self.nside = nside
        self.nfeat = nfeat
        self.root = os.path.expanduser(root)
        self.role = role
        file = root+"/stats.pkl"
        try:
            stats = pkl.load(open(file, 'rb'))
            self.mean = stats[nside][role]['mean'][:nfeat]
            self.std = stats[nside][role]['std'][:nfeat]
            self.stats_loaded = True
        except:
            self.mean = 0.
            self.std = 1.
            self.stats_loaded = False
            if verbose:
                print('ModelNet40: No statistics currently available.')

        classes = sorted(glob.glob(os.path.join(self.root, '*/')))
        self.classes = [os.path.split(_class[:-1])[-1] for _class in classes]
        self.nclasses = len(self.classes)
        self.dir = os.path.join(self.root, "{}", role)

        if role not in ("train", "test"):
            raise ValueError("Invalid role: {}".format(role))

        if not self._check_exists():
            raise RuntimeError("Dataset not found. Please download it.")

        self.files = []
        self.labels = []  # might be utterly useless
        for i, _class in enumerate(self.classes):
            files = sorted(glob.glob(os.path.join(self.dir.format(_class), '*.off')))
            self.files += files
            self.labels += [i]*len(files)

        self.processed_dir = os.path.join(self.root, self.experiment, role)
        os.makedirs(self.processed_dir, exist_ok=True)
        if nfile is not None:
            self.files = self.files[:nfile]
            self.labels = self.labels[:nfile]

        if fix:
            self._fix()

        self.cache = cache
        # boolean list of flags indicating if files have been cached
        self.is_cached = self._init_cached_flags()
        self.verbose = verbose

    def __getitem__(self, idx):
        file_path = self.files[idx]
        target = self.labels[idx]

        if not self.is_cached[idx]:
            img = Transform(file_path, self.nside, verbose=self.verbose)
            if self.cache:
                self._cache_numpy(img, idx)
        else:
            img = self._load_numpy(idx)

        return img, target

    def _cache_numpy(self, img, idx):
        if self.is_cached[idx]:
            return
        numpy_path = self._get_cached_path(idx)

        np.save(numpy_path, img)
        self.is_cached[idx] = True

    def _load_numpy(self, idx):
        numpy_path = self._get_cached_path(idx)
        # Assumes file already exists
        img = np.load(numpy_path)
        return img

    def _get_cached_path(self, idx):
        file_path = self.files[idx]
        suffix = os.path.splitext(os.path.split(file_path)[-1])[0]
        suffix += '.npy'
        numpy_path = os.path.join(self.processed_dir, suffix)
        return numpy_path

    def _check_exists(self):
        files = glob.glob(os.path.join(self.dir.format(self.classes[0]), "*.off"))
        return len(files) > 0

    def _init_cached_flags(self):
        cached = []
        for idx in range(len(self)):
            numpy_path = self._get_cached_path(idx)
            cached.append(os.path.exists(numpy_path))

        return cached

    def _fix(self):
        """
        Fix OFF files with wrong headers: headers like OFF18515 26870 0 with the counts
        attached to 'OFF'; put the counts on a separate line
        """
        verbose = self.verbose
        if verbose:
            print('Fixing OFF files')
        regexp = re.compile(r'OFF[\n]?(-?\d+) (-?\d+) (-?\d+)')
        count = 0
        for idx, file in enumerate(self.files):
            with open(file, 'rt') as fp:
                text = fp.read()
                text = text.lstrip()
                pattern = regexp.sub(r"OFF\n\1 \2 \3", text)
                if text != pattern:
                    if verbose:
                        print('{}: Wrong header: {}  :'.format(file), end=' ')
                    count += 1
                    with open(file, "wt") as fp:
                        fp.write(pattern)
                        if self.verbose:
                            print('corrected')
            if verbose:
                print("{}/{}({:%})  {} files fixed"
                      .format(idx+1, len(self), (idx+1)/ len(self), count), end="\r")
        if verbose:
            print('Done. {} files fixed'.format(count))

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':

    dataset = ModelNet40Dataset('./data/ModelNet40', 'train', nside=32, nfeat=6,
                                cache=False, verbose=True)
