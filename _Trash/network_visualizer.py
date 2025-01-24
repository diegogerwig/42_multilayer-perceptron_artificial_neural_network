import numpy as np
import json

def prepare_network_data(model):
    return {
        'W': [w.tolist() for w in model['W']]
    }

def save_network_data(model, filename='network_data.json'):
    data = prepare_network_data(model)
    with open(filename, 'w') as f:
        json.dump(data, f)
    return data