import pickle


def save_dict(state_dict: dict, fname: str):
    with open(fname, 'wb') as f:
        pickle.dump(state_dict, f)


def load_dict(fname: str):
    with open(fname, 'rb') as f:
        state_dict = pickle.load(f)
    
    return state_dict
    

def load_checkpoint(fname, model, optim=None):
    states = load_dict(fname)
    model.load_state_dict(states['model'])
    if optim is not None:
        optim.load_state_dict(states['optim'])
    step = states['step']

    return model, optim, step