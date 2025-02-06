import numpy as np

'''
Selection guide:

Ramdom initialization is not recommended for deep networks
Use He initialization for ReLU networks
Use Glorot/Xavier for sigmoid/tanh networks
Normal distributions often work better for larger networks
Uniform distributions can be more stable for smaller networks
'''

def random(input_size, output_size):
    """
    Initialize weights using random initialization.
    Scaled between min_value and max_value.
    Not recommended for deep networks.
    """
    min_value = np.min([input_size, output_size])
    max_value = np.max([input_size, output_size])
    weight = np.random.rand(input_size, output_size) * (max_value - min_value) + min_value
    return weight

def he_normal(input_size, output_size):
    """
    Initialize weights using He normal initialization.
    """
    weight = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
    return weight

def he_uniform(input_size, output_size):
    """
    Initialize weights using He uniform initialization.
    """
    limit = np.sqrt(6 / input_size)
    weight = np.random.uniform(-limit, limit, (input_size, output_size))
    return weight

def glorot_uniform(input_size, output_size):
    """
    Initialize weights using Glorot/Xavier uniform initialization.
    """
    limit = np.sqrt(6 / (input_size + output_size))
    weight = np.random.uniform(-limit, limit, (input_size, output_size))
    return weight

def glorot_normal(input_size, output_size):
    """
    Initialize weights using Glorot/Xavier normal initialization.
    """
    stddev = np.sqrt(2 / (input_size + output_size))
    weight = np.random.randn(input_size, output_size) * stddev
    return weight