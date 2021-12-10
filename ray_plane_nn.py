import torch
import pinhole_camera_module as phutil

import torch
import torch.nn as nn



class RayPlaneNet(nn.Module):

    """
    Simple network to approximate the ray-plane intersection function.


    Ray Plane intersection takes as an input;

    1 x 3: rays
    3 x 3: rotation matrix
    1 x 3: translation vector


    output:

    1 x 2: pixel location


    Question of the day:
       1. What's a good way to model this input? n x 3 rays? Or leave as single
          ray?
       2. What kind of network would be needed for this?


    For simplicity, we can start with a fully connected network of 4 layers. We
    will have the network take in 1 ray at a time.
    """

    def __init__(self):
        super(RayPlaneNet, self).__init__()

        model = nn.Sequential(
          nn.Linear(5, 3),
          nn.ReLU(),
          nn.Linear(64),
          nn.ReLU(),
          nn.Linear(64),
          nn.ReLU(),
          nn.Linear(64),
          nn.ReLU(),
          nn.Linear(2),
        )
