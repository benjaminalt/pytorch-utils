import unittest

import torch
from pytorch_utils import utils


class TestUtils(unittest.TestCase):
    def test_differentiable_where(self):
        t = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], requires_grad=True)
        where = utils.differentiable_where(t, torch.tensor(0.5), torch.tensor(0.0), torch.tensor(1.0))
        loss = torch.nn.MSELoss()(where, torch.ones_like(t))
        loss.backward()
        self.assertIsNotNone(t.grad)

    def test_differentiable_len(self):
        t = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).unsqueeze(1)
        t.requires_grad = True
        length = utils.differentiable_len(t)
        loss = torch.nn.MSELoss()(length, torch.tensor(6.0))
        loss.backward()
        self.assertIsNotNone(t.grad)

    def test_differentiable_len_batch(self):
        t = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                          [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]]).unsqueeze(-1)
        t.requires_grad = True
        length = utils.differentiable_len(t)
        loss = torch.nn.MSELoss()(length, torch.tensor(6.0))
        loss.backward()
        self.assertIsNotNone(t.grad)


if __name__ == '__main__':
    unittest.main()
