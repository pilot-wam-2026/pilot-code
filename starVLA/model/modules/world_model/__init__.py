def get_world_model(config):
    from .CosmoPredict25 import _CosmoPredict25_Interface
    return _CosmoPredict25_Interface(config)
