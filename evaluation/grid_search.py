import hashlib
import json
import tensorflow as tf


def grid_hparams(specification):
    """
    Grids over specification and yields tf.contrib.training.HParams for each configuration
    :param specification: Dictionary of parameters to grid on. Keys are parameter names,
                          and values are list of grid points
    :return: Yields a tf.contrib.training.HParams object for each gridded configuration
    """
    yield from _grid_hparams_req({}, specification)


def _grid_hparams_req(current_params, specification):
    """
    Recursively complete parameter configuration, and yield tf.contrib.training.HParams object
    :param current_params: Dictionary of currently populated parameters
    :param specification: The rest of dict specification to grid on
    :return: Yields a new tf.contrib.training.HParams object
    """
    if not specification:
        current_params['id'] = hashlib.md5(json.dumps(current_params, sort_keys=True).encode('utf-8')).hexdigest()
        yield tf.contrib.training.HParams(**current_params)
    else:
        specification = specification.copy()
        key = next(iter(specification))
        values = specification[key]
        del specification[key]
        if isinstance(values, list):
            for value in values:
                current_params[key] = value
                yield from _grid_hparams_req(current_params, specification)
        else:
            # If parameter is not a list of values to grid on
            current_params[key] = values
            yield from _grid_hparams_req(current_params, specification)
