import torch


def _c(ca: torch.Tensor, i: int, j: int, p: torch.Tensor, q: torch.Tensor):
    # setting_mask = torch.zeros_like(ca)
    # setting_mask[i, j] = 1.0
    # clearing_mask = torch.ones_like(ca)
    # clearing_mask[i, j] = 0.0
    ca = ca.clone()
    if ca[i, j] > -1:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = torch.norm(p[i]-q[j])
        # ca = ca * clearing_mask + torch.norm(p[i]-q[j]) * setting_mask
    elif i > 0 and j == 0:
        ca[i, j] = torch.max(_c(ca, i - 1, 0, p, q), torch.norm(p[i] - q[j]))
        # ca = ca * clearing_mask + setting_mask * torch.max(_c(ca, i - 1, 0, p, q), torch.norm(p[i] - q[j]))
    elif i == 0 and j > 0:
        ca[i, j] = torch.max(_c(ca, 0, j - 1, p, q), torch.norm(p[i] - q[j]))
        # ca = ca * clearing_mask + setting_mask * torch.max(_c(ca, 0, j - 1, p, q), torch.norm(p[i] - q[j]))
    elif i > 0 and j > 0:
        ca[i, j] = torch.max(
            torch.min(torch.stack((_c(ca, i - 1, j, p, q), _c(ca, i - 1, j - 1, p, q), _c(ca, i, j - 1, p, q)))), torch.norm(p[i] - q[j]))
        # ca = ca * clearing_mask + setting_mask * torch.max(torch.min(torch.stack((_c(ca, i - 1, j, p, q), _c(ca, i - 1, j - 1, p, q), _c(ca, i, j - 1, p, q)))), torch.norm(p[i] - q[j]))
    else:
        ca[i, j] = torch.tensor(float('inf'))
        # ca = ca * clearing_mask + setting_mask * torch.tensor(float('inf'))

    return ca[i, j]


def frdist(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Computes the discrete Fréchet distance between
    two curves. The Fréchet distance between two curves in a
    metric space is a measure of the similarity between the curves.
    The discrete Fréchet distance may be used for approximately computing
    the Fréchet distance between two arbitrary curves,
    as an alternative to using the exact Fréchet distance between a polygonal
    approximation of the curves or an approximation of this value.
    This is a Python 3.* implementation of the algorithm produced
    in Eiter, T. and Mannila, H., 1994. Computing discrete Fréchet distance.
    Tech. Report CD-TR 94/64, Information Systems Department, Technical
    University of Vienna.
    http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf
    Function dF(P, Q): real;
        input: polygonal curves P = (u1, . . . , up) and Q = (v1, . . . , vq).
        return: δdF (P, Q)
        ca : array [1..p, 1..q] of real;
        function c(i, j): real;
            begin
                if ca(i, j) > −1 then return ca(i, j)
                elsif i = 1 and j = 1 then ca(i, j) := d(u1, v1)
                elsif i > 1 and j = 1 then ca(i, j) := max{ c(i − 1, 1), d(ui, v1) }
                elsif i = 1 and j > 1 then ca(i, j) := max{ c(1, j − 1), d(u1, vj) }
                elsif i > 1 and j > 1 then ca(i, j) :=
                max{ min(c(i − 1, j), c(i − 1, j − 1), c(i, j − 1)), d(ui, vj ) }
                else ca(i, j) = ∞
                return ca(i, j);
            end; /* function c */
        begin
            for i = 1 to p do for j = 1 to q do ca(i, j) := −1.0;
            return c(p, q);
        end.
    Parameters
    ----------
    P : Input curve - two dimensional array of points
    Q : Input curve - two dimensional array of points
    Returns
    -------
    dist: float64
        The discrete Fréchet distance between curves `P` and `Q`.
    """
    p = p.double()
    q = q.double()

    len_p = p.size(0)
    len_q = q.size(0)

    ca = (torch.ones((len_p, len_q), dtype=torch.float64) * -1)

    dist = _c(ca, len_p - 1, len_q - 1, p, q)
    return dist
