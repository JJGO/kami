from {datafile} import {Dataset}

from kirin import models as kirin_models
import models


def Data(dataset, **data_params):
    params = dict(locals())
    del params['data_params']

    data = {Dataset}(**data_params)

    params.update(data.params)
    return data, params


def Model(model, cls='KerasModel', **model_params):

    params = dict(locals())
    del params['model_params']

    if hasattr(models, cls):
        cls = getattr(models, cls)
    elif hasattr(kirin_models, cls):
        cls = getattr(kirin_models, cls)
    else:
        raise ValueError(f"Unknown cls {cls}")

    model = cls(model, **model_params)

    params.update(model.params)
    return model, params
