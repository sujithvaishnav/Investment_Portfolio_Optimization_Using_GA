import random
import numpy as np
from deap import base, creator, tools

def setup_deap(num_assets):
    """
    Create DEAP creators and toolbox for given number of assets.
    Returns toolbox.
    """
    # create creators only once
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=num_assets)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # registration of common operators is done in run_ga
    return toolbox

def genotype_to_phenotype(genotype, max_weight=0.5):
    """
    genotype: array-like of positive floats
    returns: normalized numpy array of weights that sum to 1, clipped to max_weight then renormalized
    """
    arr = np.array(genotype, dtype=float)
    # avoid divide by zero
    if arr.sum() == 0:
        arr = np.ones_like(arr)
    w = arr / arr.sum()
    w = np.clip(w, 0.0, max_weight)
    # if clipping led to zero-sum, replace with equal weights clipped then renormalize
    if w.sum() == 0:
        w = np.clip(np.ones_like(w) / len(w), 0.0, max_weight)
    w = w / w.sum()
    w = np.real(w).astype(float)
    return w
