"""Device handling. Only the code that CREATES a model (fit_probe, fit_abmil, run_nn_probes) needs a device;
everything that receives a model (scorers, encoders, plots) uses the model's own device."""
import torch

_DEVICE = None


def _auto_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device():
    """The default device: set_device(...) if called, otherwise cuda > mps > cpu."""
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = _auto_device()
    return _DEVICE


def set_device(device):
    global _DEVICE
    _DEVICE = torch.device(device)


def resolve_device(device=None):
    """None -> default device; str / torch.device -> torch.device."""
    return get_device() if device is None else torch.device(device)


def model_device(model):
    """Device of a model's parameters (cpu if it has none)."""
    return next(model.parameters(), torch.empty(0)).device