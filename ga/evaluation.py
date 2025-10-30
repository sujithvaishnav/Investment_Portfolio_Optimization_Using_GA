import numpy as np
from ga.fitness import fitness_function

def evaluate_portfolio_weights(individual, returns_df, **kwargs):
    # Ensure individual is a flat list of floats
    weights = [float(w) for w in individual]
    
    # Call the actual fitness function
    fitness = fitness_function(weights, returns_df, **kwargs)
    
    # Flatten the tuple in case it's nested
    if isinstance(fitness, tuple):
        # Extract the first float if nested
        fitness = tuple(float(f) for f in np.ravel(fitness))
    else:
        # If single float, wrap in a tuple
        fitness = (float(fitness),)
    
    return fitness

