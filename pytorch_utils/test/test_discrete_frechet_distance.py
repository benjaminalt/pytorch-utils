import unittest

from pytorch_utils.discrete_frechet_distance import frdist
from scipy.stats import norm
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def _c(ca, i, j, p, q):

    if ca[i, j] > -1:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = np.linalg.norm(p[i]-q[j])
    elif i > 0 and j == 0:
        ca[i, j] = max(_c(ca, i-1, 0, p, q), np.linalg.norm(p[i]-q[j]))
    elif i == 0 and j > 0:
        ca[i, j] = max(_c(ca, 0, j-1, p, q), np.linalg.norm(p[i]-q[j]))
    elif i > 0 and j > 0:
        ca[i, j] = max(
            min(
                _c(ca, i-1, j, p, q),
                _c(ca, i-1, j-1, p, q),
                _c(ca, i, j-1, p, q)
            ),
            np.linalg.norm(p[i]-q[j])
            )
    else:
        ca[i, j] = float('inf')

    return ca[i, j]


def frdist_reference(p, q):
    p = np.array(p, np.float64)
    q = np.array(q, np.float64)

    len_p = len(p)
    len_q = len(q)

    if len_p == 0 or len_q == 0:
        raise ValueError('Input curves are empty.')

    # if len_p != len_q or len(p[0]) != len(q[0]):
    #     raise ValueError('Input curves do not have the same dimensions.')

    ca = (np.ones((len_p, len_q), dtype=np.float64) * -1)

    dist = _c(ca, len_p-1, len_q-1, p, q)
    return dist


def brownian_motion(n: int):
    # Process parameters
    delta = 0.25
    dt = 0.1

    # Initial condition.
    x, y, z = 0.0, 0.0, 0.0

    curve = []
    # Iterate to compute the steps of the Brownian motion.
    for k in range(n):
        x = x + norm.rvs(scale=delta ** 2 * dt)
        y = y + norm.rvs(scale=delta ** 2 * dt)
        z = z + norm.rvs(scale=delta ** 2 * dt)
        curve.append([x, y, z])
    return torch.tensor(curve)


class TestDiscreteFrechetDistance(unittest.TestCase):
    def test_frdist_pytorch(self):
        curve_1 = brownian_motion(200)
        # curve_1.requires_grad = True
        curve_2 = brownian_motion(200)
        # curve_2.requires_grad = True

        dist_orig = frdist_reference(curve_1.detach().numpy(), curve_2.detach().numpy())
        dist_torch = frdist(curve_1, curve_2)
        self.assertTrue(torch.allclose(torch.tensor(dist_orig, dtype=torch.double), dist_torch))
        # loss = torch.nn.MSELoss()(dist_torch, torch.zeros_like(dist_torch))
        # loss.backward()
        # self.assertIsNotNone(curve_1.grad)
        # self.assertIsNotNone(curve_2.grad)

    @unittest.skip
    def test_frdist_pytorch_optimize(self):
        np.random.seed(42)

        curve_1 = brownian_motion(200)
        curve_1.requires_grad = True
        curve_2 = brownian_motion(200)
        curve_2.requires_grad = True

        optimizer = torch.optim.Adam([curve_2])

        fig, axes = plt.subplots(2, sharex="col")
        axes[0].plot(range(len(curve_1)), curve_1[:, 0].tolist())
        axes[0].plot(range(len(curve_2)), curve_2[:, 0].tolist())
        num_iterations = 100
        for i in tqdm(range(num_iterations)):
            optimizer.zero_grad()
            dist_torch = frdist(curve_1, curve_2)
            loss = torch.nn.MSELoss()(dist_torch, torch.zeros_like(dist_torch))
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                axes[1].plot(range(len(curve_2)), curve_2[:, 0].tolist(), color="gray", alpha=i/num_iterations)

        axes[1].plot(range(len(curve_1)), curve_1[:, 0].tolist())
        axes[1].plot(range(len(curve_2)), curve_2[:, 0].tolist())
        plt.show()
