
def set_default_val(config, key, value):
    if key not in config:
        config[key] = value