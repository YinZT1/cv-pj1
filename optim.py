def sgd(w, dw, config=None):
    '''
    -config: hyper params
    -w:weight
    -dw:grad
    '''    
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config