import pickle


def save_dict(state_dict: dict, fname: str):
    with open(fname, 'wb') as f:
        pickle.dump(state_dict, f)


def load_dict(fname: str):
    with open(fname, 'rb') as f:
        state_dict = pickle.load(fname)
    
    return state_dict
    
