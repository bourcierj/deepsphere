import numpy as np
import torch
import healpy as hp


class SphereHealpix():
    """
    Sphere tessellated with Healpix sampling.
    The graph is a "skeleton" that does not actually contain any data. Inteneed to be used
    with PyTorch Geometric.
    Args:
        nside (int):
            The healpix nside parameter, must be a power of 2, less than 2**30.
        indexes (list of int, optional)
            List of indexes to use. This allows to build the graph from a part of the
            sphere only. If None, the default, the whole sphere is used.
        similarity (str, optional):
            Weighting scheme for edges
        dtype (data-type, optional):
            The desired data type of the weight matrix.
    Attrs:
        edge_index (torch.LongTensor):
            Matrix of graph edges in COO format, compliant with PyTorch Geometric
        edge_weight (torch.FloatTensor):
            Matrix of edge weights, each row contains values (edge_id, weight), compliant
            with PyTorch Geometric
    """
    import healpy as hp
    def __init__(self, nside=16, indexes=None, similarity='perraudin', dtype=np.float32, **kwargs):

        assert(similarity in {'perraudin', 'renata'})
        if indexes is None:
            indexes = range(nside**2 * 12)
        self.nside = nside
        npix = 12 * nside**2  # number of pixels
        self.npix = npix
        if npix >= max(indexes) + 1:
            # If the user input is not consecutive nodes, we need to use a slower
            # method.
            usefast = True
            indexes = range(npix)
        else:
            usefast = False
            indexes = list(indexes)
        # Get the coordinates
        x, y, z = hp.pix2vec(nside, indexes)
        coords = np.vstack([x, y, z]).transpose()
        coords = np.asarray(coords, dtype=dtype)
        # Get the 7-8 neighbors
        neighbors = hp.pixelfunc.get_all_neighbours(nside, indexes)
        # Indices of non-zero values in the adjacency matrix
        col_index = neighbors.T.reshape((npix * 8))
        row_index = np.repeat(indexes, 8)
        # Remove pixels that are out of our indexes of interest (part of sphere)
        if usefast:
            keep = (col_index < npix)
            # Remove fake neighbors (some pixels have less than 8)
            keep &= (col_index >= 0)
            col_index = col_index[keep]
            row_index = row_index[keep]
        else:
            col_index_set = set(indexes)
            keep = [c in col_index_set for c in col_index]
            inverse_map = [np.nan] * (nside**2 * 12)
            for i, index in enumerate(indexes):
                inverse_map[index] = i
            col_index = [inverse_map[el] for el in col_index[keep]]
            row_index = [inverse_map[el] for el in row_index[keep]]

        # Tensor edge index matrix in COO format
        self.edge_index = torch.LongTensor(np.vstack((col_index, row_index)))

        # Set the edge weights
        # compute Euclidean distances between neighbors
        distances = np.sum((coords[row_index] - coords[col_index])**2, axis=1)
        if similarity == 'perraudin':
            # Similarity proposed by Perraudin, 2019
            kernel_width = np.mean(distances)
            weights = np.exp(-distances / (2 * kernel_width))
        else:
            # Similarity proposed by Renata & Pascal, ICCV 2017
            weights = 1 / distances

        # Tensor edge weight matrix in vector format
        self.edge_weight = torch.FloatTensor(weights.reshape(-1, 1))


class SphereEquiangular():
    """
    Sphere tessellated with Equiangular sampling.
    The graph is a "skeleton" that does not actually contain any data. Inteneed to be used
    with PyTorch Geometric.
    Args:
        bw (int):
            bandwidth, size of grid  (default = 64)
        indexes (list of int, optional)
            List of indexes to use. This allows to build the graph from a part of the
            sphere only. If None, the default, the whole sphere is used.
        similarity (str, optional):
            Weighting scheme for edges
        n_neighbors (int, optional):
            Number of neighbors for building the graph (4 or 8)
        dtype (data-type, optional):
            The desired data type of the weight matrix.
    Attrs:
        edge_index (torch.LongTensor):
            Matrix of graph edges in COO format, compliant with PyTorch Geometric
        edge_weight (torch.FloatTensor):
            Matrix of edge weights, each row contains values (edge_id, weight), compliant
            with PyTorch Geometric
    """
    def __init__(self, bw=64, indexes=None, similarity='renata', n_neighbors=8, dtype=np.float32, **kwargs):
        assert(similarity in {'perraudin', 'renata'})
        assert(n_neighbors in {4, 8})
        if indexes is None:
            indexes = range((2*bw)**2)
        self.bw = bw
        # Find a mean to take only indexes from grid
        # Sampling: SOFT (SO(3) Fourier Transform optimal)
        beta = np.pi * (2 * np.arange(2 * bw) + 1) / (4. * bw) #SOFT
        alpha = np.arange(2 * bw) * np.pi / bw
        theta, phi = np.meshgrid(*(beta, alpha),indexing='ij')
        # Get the coordinates
        ct = np.cos(theta)
        st = np.sin(theta)
        cp = np.cos(phi)
        sp = np.sin(phi)
        x = st * cp
        y = st * sp
        z = ct
        coords = np.vstack([x.flatten(), y.flatten(), z.flatten()]).transpose()
        coords = np.asarray(coords, dtype=dtype)
        npix = len(coords)  # number of pixels
        self.npix = npix

        # Find the 4 or 8 neighbors of all vertices
        col_index = []
        for ind in indexes:
            neighbor = self.get_neighbor(ind, n_neighbors)
            col_index += list(neighbor)

        col_index = np.asarray(col_index)
        row_index = np.hstack([np.repeat(indexes, n_neighbors)])
        # Tensor edge index matrix in COO format
        self.edge_index = torch.LongTensor(np.vstack((col_index, row_index)))

        # Set the edge weights
        # compute Euclidean distances between neighbors
        distances = np.sum((coords[row_index] - coords[col_index])**2, axis=1)
        if similarity == 'perraudin':
            # Similarity proposed by Perraudin, 2019
            kernel_width = np.mean(distances)
            weights = np.exp(-distances / (2 * kernel_width))
        else:
            # Similarity proposed by Renata & Pascal, ICCV 2017
            weights = 1 / distances
        # Tensor edge weight matrix in vector format
        self.edge_weight = torch.FloatTensor(weights.reshape(-1, 1))

    def get_neighbor(self, ind, n_neighbors=8):
        """Find equiangular neighbors of vertex ind"""
        npix = self.npix
        bw = self.bw
        def south(x, bw):
            if x >= npix - 2*bw:
                return (x + bw)%(2*bw) + npix - 2*bw
            else:
                return x + 2*bw
        def north(x, bw):
            if x < 2*bw:
                return (x + bw)%(2*bw)
            else:
                return x - 2*bw
        def west(x, bw):
            if x%(2*bw)==0:
                x += 2*bw
            return x -1
        def east(x, bw):
            if x%(2*bw)==2*bw-1:
                x -= 2*bw
            return x + 1

        if n_neighbors == 8:
            neighbor = [south(west(ind,bw),bw), west(ind,bw), north(west(ind,bw), bw), north(ind,bw),
                        north(east(ind,bw),bw), east(ind,bw), south(east(ind,bw),bw), south(ind,bw)]
        else:
            neighbor = [west(ind,bw), north(ind,bw), east(ind,bw), south(ind,bw)]
        return neighbor


class SphereIcosahedral():
    """
    Sphere tessellated with Icosahedral sampling
    The graph is a "skeleton" that does not actually contain any data. Inteneed to be used
    with PyTorch Geometric.
    Note: The sampled points are positioned on the vertices.
    Args:
        level (int):
            The level of icosaedral division. 0 is the unit icosahedron, and each progressive
            mesh resolution is one level above the previous.
        indexes (list of int, optional)
            List of indexes to use. This allows to build the graph from a part of the
            sphere only. If None, the default, the whole sphere is used.
            #@NOTIMPLEMENTED
        similarity (str, optional):
            Weighting scheme for edges
        dtype (data-type, optional):
            The desired data type of the weight matrix.
    Attrs:
        edge_index (torch.LongTensor):
            Matrix of graph edges in COO format, compliant with PyTorch Geometric
        edge_weight (torch.FloatTensor):
            Matrix of edge weights, each row contains values (edge_id, weight), compliant
            with PyTorch Geometric
    """
    def __init__(self, level, indexes=None, similarity='perraudin', dtype=np.float32, **kwargs):
        if indexes is not None:
            raise NotImplementedError()
        self.intp = None
        # get coordinates of vertices, and faces
        self.coords, self.faces = self.icosahedron(upward=True)
        self.level = level
        # subdivide and normalize up to the level

        for i in range(level):
            self.subdivide()
            self.normalize()
        self.lat, self.long = self.xyz2latlon()

        npix = self.coords.shape[0]
        self.npix = npix
        self.nf = 20 * 4**self.level  # number of faces
        self.ne = 30 * 4**self.level  # number of edges
        self.nv = self.ne - self.nf + 2  # number of vertices (i.e number of pixels)
        self.nv_prev = int((self.ne / 4) - (self.nf / 4) + 2)
        self.nv_next = int((self.ne * 4) - (self.nf * 4) + 2)

        # Find the neighbors of all vertices
        col_index = []
        for ind in range(npix):
            # Get the 5 or 6 neighbors. If there's only 5 neighbors add a fake one
            neighbor = self.get_neighbor(ind)
            if len(neighbor) == 5:
                neighbor.append(-1)
            else:
                assert(len(neighbor) == 6)
            col_index += list(neighbor)

        col_index = np.asarray(col_index)
        row_index = np.repeat(range(npix), 6)
        # Remove fake neighbors (some pixels have 5)
        keep = (col_index < npix)
        keep &= (col_index >= 0)
        col_index = col_index[keep]
        row_index = row_index[keep]
        # Tensor edge index matrix in COO format
        self.edge_index = torch.LongTensor(np.vstack((col_index, row_index)))

        # Set the edge weights
        distances = np.sum((self.coords[row_index] - self.coords[col_index])**2, axis=1)
        if similarity == 'perraudin':
            kernel_width = np.mean(distances)
            weights = np.exp(-distances / (2 * kernel_width))
        else:
            weights = 1 / distances
        # Tensor edge weight matrix in vector format
        self.edge_weight = torch.FloatTensor(weights.reshape(-1, 1))


    def subdivide(self):
        """Subdivide a mesh into smaller triangles."""
        from icosahedral_utils import _unique_rows

        faces = self.faces
        vertices = self.coords
        face_index = np.arange(len(faces))
        # the (c,3) int set of vertex indices
        faces = faces[face_index]
        # the (c, 3, 3) float set of points in the triangles
        triangles = vertices[faces]
        # the 3 midpoints of each triangle edge vstacked to a (3*c, 3) float
        src_idx = np.vstack([faces[:, g] for g in [[0, 1], [1, 2], [2, 0]]])
        mid = np.vstack([triangles[:, g, :].mean(axis=1) for g in [[0, 1],
                                                                   [1, 2],
                                                                   [2, 0]]])
        mid_idx = (np.arange(len(face_index) * 3)).reshape((3, -1)).T
        # for adjacent faces we are going to be generating the same midpoint
        # twice, so we handle it here by finding the unique vertices
        unique, inverse = _unique_rows(mid)

        mid = mid[unique]
        src_idx = src_idx[unique]
        mid_idx = inverse[mid_idx] + len(vertices)
        # the new faces, with correct winding
        f = np.column_stack([faces[:, 0], mid_idx[:, 0], mid_idx[:, 2],
                             mid_idx[:, 0], faces[:, 1], mid_idx[:, 1],
                             mid_idx[:, 2], mid_idx[:, 1], faces[:, 2],
                             mid_idx[:, 0], mid_idx[:, 1], mid_idx[:, 2], ]).reshape((-1, 3))
        # add the 3 new faces per old face
        new_faces = np.vstack((faces, f[len(face_index):]))
        # replace the old face with a smaller face
        new_faces[face_index] = f[:len(face_index)]

        new_vertices = np.vstack((vertices, mid))
        # source ids
        nv = vertices.shape[0]
        identity_map = np.stack((np.arange(nv), np.arange(nv)), axis=1)
        src_id = np.concatenate((identity_map, src_idx), axis=0)

        self.coords = new_vertices
        self.faces = new_faces
        self.intp = src_id

    def normalize(self, radius=1):
        '''Reproject to spherical surface'''
        vectors = self.coords
        scalar = (vectors ** 2).sum(axis=1)**.5
        unit = vectors / scalar.reshape((-1, 1))
        offset = radius - scalar
        self.coords += unit * offset.reshape((-1, 1))

    def icosahedron(self, upward=False):
        """Create an icosahedron, a 20 faced polyhedron."""
        phi = (1 + 5**0.5) / 2
        radius = (phi**2 + 1)**0.5
        vertices = [0, 1, phi, 0, -1, phi, 0, 1, -phi, 0, -1, -phi, phi, 0, 1,
                    phi, 0, -1, -phi, 0, 1, -phi, 0, -1, 1, phi, 0, -1, phi, 0,
                    1, -phi, 0, -1, -phi, 0]
        vertices = np.reshape(vertices, (-1, 3)) / radius
        faces = [0, 1, 6, 0, 6, 9, 0, 9, 8, 0, 8, 4, 0, 4, 1, 1, 6, 11, 11, 6, 7,
                 6, 7, 9, 7, 9, 2, 9, 2, 8, 2, 8, 5, 8, 5, 4, 5, 4, 10, 4, 10,
                 1, 10, 1, 11, 3, 10, 11, 3, 11, 7, 3, 7, 2, 3, 2, 5, 3, 5, 10]
        faces = np.reshape(faces, (-1,3))
        if upward:
            vertices = self._upward(vertices, faces)
        return vertices, faces

    def xyz2latlon(self):
        x, y, z = self.coords[:, 0], self.coords[:, 1], self.coords[:, 2]
        lon = np.arctan2(y, x)
        xy2 = x**2 + y**2
        lat = np.arctan2(z, np.sqrt(xy2))
        return lat, lon

    def _upward(self, V_ico, F_ico, ind=11):
        V0 = V_ico[ind]
        Z0 = np.array([0, 0, 1])
        k = np.cross(V0, Z0)
        ct = np.dot(V0, Z0)
        st = -np.linalg.norm(k)
        R = self._rot_matrix(k, ct, st)
        V_ico = V_ico.dot(R)
        # rotate a neighbor to align with (+y)
        ni = self.get_neighbor(ind, F_ico)[0]
        vec = V_ico[ni].copy()
        vec[2] = 0
        vec = vec/np.linalg.norm(vec)
        y_ = np.eye(3)[1]

        k = np.eye(3)[2]
        crs = np.cross(vec, y_)
        ct = -np.dot(vec, y_)
        st = -np.sign(crs[-1])*np.linalg.norm(crs)
        R2 = self._rot_matrix(k, ct, st)
        V_ico = V_ico.dot(R2)
        return V_ico

    def _rot_matrix(self, rot_axis, cos_t, sin_t):
        k = rot_axis / np.linalg.norm(rot_axis)
        I = np.eye(3)

        R = []
        for i in range(3):
            v = I[i]
            vr = v*cos_t+np.cross(k, v)*sin_t+k*(k.dot(v))*(1-cos_t)
            R.append(vr)
        R = np.stack(R, axis=-1)
        return R

    def get_neighbor(self, ind, faces=None):
        """Find icosahedron neighbors of vertex ind"""

        if faces is None:
            faces = self.faces
        rows, _ = np.isin(faces, ind).nonzero()
        adj_faces = faces[rows]
        neighbor = np.unique(adj_faces)
        neighbor = neighbor[np.where(neighbor != ind)]
        return neighbor.tolist()
