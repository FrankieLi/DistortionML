import torch

import numpy as np
import pinhole_camera_module as phutil
from scipy.spatial.transform import Rotation as R

def test_ray_plane():

    N_sample = 100
    for i in range(N_sample):

        rays = 2*(np.reshape(np.random.rand(9), (3, 3)) - 0.5)
        rmat = R.random().as_matrix()

        max_translation = 5
        tvec = max_translation * (np.reshape(np.random.rand(3), (1, 3)) - 0.5)
        tvec = np.array([0, 0, 5], dtype=float)
        rmat = torch.from_numpy(rmat)
        rmat.requires_grad_(True)
        rays = torch.from_numpy(np.atleast_2d(rays))  # shape (npts, 3)
        rays.requires_grad_(True)
        tvec = torch.from_numpy(tvec)
        tvec.requires_grad_(True)


        rp_torch = phutil.ray_plane_torch(rays, rmat, tvec)

        rp_original = phutil.ray_plane(rays.detach().numpy(),
            rmat.detach().numpy(), tvec.detach().numpy())

        rp_orig = torch.tensor(rp_original)
        if not torch.all(torch.isclose(rp_torch, rp_orig, equal_nan=True)):
            print(f'orig:  {rp_orig}')
            print(f'torch: {rp_torch}')
            raise('Error: Numpy and Pytorch solution m mismatch')

