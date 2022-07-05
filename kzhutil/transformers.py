import shutil


def load_pretrained(model_class, model_name, model_path, **kwargs):
    """
    load pretrained transformers. If a saved model provided, then load directly. Else load from hub and save.

    :param model_class: e.g. transformers.AutoModel, transformers.AutoTokenizer
    :param model_name: the transformer model name
    :param model_path: try to load a model in this path
    """
    cache_path = model_path + "/cache"
    try:
        model = model_class.from_pretrained(model_path, cache_dir=cache_path, **kwargs)
    except Exception:
        model = model_class.from_pretrained(model_name, cache_dir=cache_path, **kwargs)
        model.save_pretrained(model_path)
        shutil.rmtree(cache_path)
    return model
