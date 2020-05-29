import numpy as np
from scipy import spatial


class PointCloud:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

        self.nx = None
        self.ny = None
        self.nz = None
        self.planarity = None

        self.no_points = len(x)
        self.sel = None

    def select_n_points(self, n):

        if self.no_points > n:
            self.sel = np.round(np.linspace(0, self.no_points-1, n)).astype(int)
        else:
            self.sel = np.arange(0, self.no_points, 1)

    def estimate_normals(self, neighbors):

        self.nx = np.full(self.no_points, np.nan)
        self.ny = np.full(self.no_points, np.nan)
        self.nz = np.full(self.no_points, np.nan)
        self.planarity = np.full(self.no_points, np.nan)

        kdtree = spatial.cKDTree(np.column_stack((self.x, self.y, self.z)))
        query_points = np.column_stack((self.x[self.sel], self.y[self.sel], self.z[self.sel]))
        _, idxNN_all_qp = kdtree.query(query_points, k=neighbors, p=2, n_jobs=-1)

        for (i, idxNN) in enumerate(idxNN_all_qp):
            selected_points = np.column_stack((self.x[idxNN], self.y[idxNN], self.z[idxNN]))
            C = np.cov(selected_points.T, bias=False)
            eig_vals, eig_vecs = np.linalg.eig(C)
            idx_sort = eig_vals.argsort()[::-1] # sort from large to small
            eig_vals = eig_vals[idx_sort]
            eig_vecs = eig_vecs[:,idx_sort]
            self.nx[self.sel[i]] = eig_vecs[0,2]
            self.ny[self.sel[i]] = eig_vecs[1,2]
            self.nz[self.sel[i]] = eig_vecs[2,2]
            self.planarity[self.sel[i]] = (eig_vals[1]-eig_vals[2])/eig_vals[0]

    def transform(self, H):

        XInE = np.column_stack((self.x, self.y, self.z))
        XInH = PointCloud.euler_coord_to_homogeneous_coord(XInE)
        XOutH = np.transpose(H @ XInH.T)
        XOut = PointCloud.homogeneous_coord_to_euler_coord(XOutH)

        self.x = XOut[:,0]
        self.y = XOut[:,1]
        self.z = XOut[:,2]

    @staticmethod
    def euler_coord_to_homogeneous_coord(XE):

        no_points = np.shape(XE)[0]
        XH = np.column_stack((XE, np.ones(no_points)))

        return XH

    @staticmethod
    def homogeneous_coord_to_euler_coord(XH):

        XE = np.column_stack((XH[:,0]/XH[:,3], XH[:,1]/XH[:,3], XH[:,2]/XH[:,3]))

        return XE
