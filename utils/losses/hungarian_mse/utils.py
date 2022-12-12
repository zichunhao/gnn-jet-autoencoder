import math
import torch


EPS = 1e-16


def get_p_polar(p):
    """
    (E, px, py, pz) or (px, py, pz)-> (eta, phi, pt)
    """
    if p.shape[-1] == 4:
        _, px, py, pz = p.unbind(-1)
    elif p.shape[-1] == 3:
        px, py, pz = p.unbind(-1)
    else:
        raise ValueError(
            f"Wrong last dimension of p. Should be 3 or 4 but found: {p.shape[-1]}."
        )

    pt = torch.sqrt(px**2 + py**2 + EPS)
    try:
        eta = torch.asinh(pz / (pt + EPS))
    except AttributeError:
        eta = arcsinh(pz / (pt + EPS))
    phi = torch.atan2(py + EPS, px + EPS)

    return torch.stack((pt, eta, phi), dim=-1)


def arcsinh(z):
    return torch.log(z + torch.sqrt(1 + torch.pow(z, 2)))


def get_polar_rel(p, jet, input_cartesian=True):
    if input_cartesian:
        p = get_p_polar(p)  # Convert to polar first
        jet = get_p_polar(jet)

    pt, eta, phi = p.unbind(-1)

    num_particles = p.shape[-2]
    pt /= (jet[..., 0] + EPS).unsqueeze(dim=-1).repeat(1, num_particles)
    eta -= jet[..., 1].unsqueeze(dim=-1).repeat(1, num_particles)
    phi -= jet[..., 2].unsqueeze(dim=-1).repeat(1, num_particles)
    phi = (phi + math.pi) % (2 * math.pi) - math.pi

    return torch.stack((pt, eta, phi), dim=-1)


def get_p_cartesian(p, return_p0=False):
    if p.shape[-1] == 4:
        _, pt, eta, phi = p.unbind(-1)
    elif p.shape[-1] == 3:
        pt, eta, phi = p.unbind(-1)
    else:
        raise ValueError(
            f"Wrong last dimension of p. Should be 3 or 4 but found: {p.shape[-1]}."
        )

    px = pt * torch.cos(phi)
    py = pt * torch.cos(phi)
    pz = pt * torch.sinh(eta)

    if not return_p0:
        return torch.stack((px, py, pz), dim=-1)
    else:
        E = pt * torch.cosh(eta)
        return torch.stack((E, px, py, pz), dim=-1)


def check_p_dim(p):
    """Check whether p is a 3- or 4-vector.

    Raise
    -----
    ValueError if p is not a 3- or 4-vector (i.e. p.shape[-1] is not 3 or 4).
    """
    if p.shape[-1] not in [3, 4]:
        raise ValueError(
            f"Wrong last dimension of p. Should be 3 or 4 but found: {p.shape[-1]}."
        )
