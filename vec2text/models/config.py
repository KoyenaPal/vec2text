import json

import transformers

# update: I am adding this to the config to allow for specific handling of
# which gradients we use for inversion (e.g. the gradients of the layer 1, etc.)
# also adding another attribute to know which version of gradient reduction we use (i.e., SVD or JL, i.e.,  Johnson-Lindenstrauss)
NEW_ATTRIBUTES = {
    "embedder_torch_dtype": "float32",
    "reduction_version_SVD": "SVD",
    "reduction_version_JL": "JL",
    "embed_in_gradient": "embed_in",
    "embed_out_gradient": "embed_out",
    "layer_0_gradient": "layers.0",
    "layer_1_gradient": "layers.1",
    "layer_2_gradient": "layers.2",
    "layer_3_gradient": "layers.3",
    "layer_4_gradient": "layers.4",
    "layer_5_gradient": "layers.5",
}


class InversionConfig(transformers.configuration_utils.PretrainedConfig):
    """We create a dummy configuration class that will just set properties
    based on whatever kwargs we pass in.

    When this class is initialized (see experiments.py) we pass in the
    union of all data, model, and training args, all of which should
    get saved to the config json.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            try:
                json.dumps(value)
                setattr(self, key, value)
            except TypeError:
                # value was not JSON-serializable, skip
                continue
        super().__init__()

    def __getattribute__(self, key):
        try:
            return super().__getattribute__(key)
        except AttributeError as e:
            if key in NEW_ATTRIBUTES:
                return NEW_ATTRIBUTES[key]
            else:
                raise e
