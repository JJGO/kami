from .base import KerasModel, retrieve_output
from .. import networks
from ..losses.vae import KL_unit_spherical_gaussian_mean
from ..losses.vae import KL_unit_spherical_gaussian_logvar


class VAE(KerasModel):

    def __init__(self, model, beta_vae=1, **model_kwargs):

        self.compiled = False

        if isinstance(model, str):
            model = getattr(networks, model)(**model_kwargs)
            self.params = model.params
            self.params['model'] = model
        self.model = model

        self.hooks = {
            "input": model.input,
            "output": model.output,
            "z_mean": retrieve_output(model, "*enc_mean"),
            "z_logvar": retrieve_output(model, "*enc_logvar")
        }

        self.outputs = ['output']

        self.loss = {
            'z_mean': [(KL_unit_spherical_gaussian_mean, beta_vae)],
            'z_logvar': [(KL_unit_spherical_gaussian_logvar, beta_vae)],
        }
