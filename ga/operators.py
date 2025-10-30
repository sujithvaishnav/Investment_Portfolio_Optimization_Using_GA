import numpy as np
from deap import tools

def tournament_selection(population, fitnesses, tourn_size=3, num_parents=150):
    """
    Perform tournament selection.
    - population: list of individuals
    - fitnesses: list of fitness values
    - tourn_size: number of individuals per tournament
    - num_parents: number of parents to select
    Returns: list of selected parents
    """
    selected = []
    pop_size = len(population)
    
    for _ in range(num_parents):
        indices = np.random.choice(pop_size, tourn_size, replace=False)
        tournament = [(population[i], fitnesses[i]) for i in indices]
        winner = min(tournament, key=lambda x: x[1])[0]  # lower fitness wins
        selected.append(winner)
    return selected

def blx_alpha_crossover(parent1, parent2, alpha=0.5):
    """
    Blend crossover (BLX-alpha) between two parents
    Returns two offspring
    """
    child1, child2 = np.zeros_like(parent1), np.zeros_like(parent2)
    for i in range(len(parent1)):
        c_min = min(parent1[i], parent2[i])
        c_max = max(parent1[i], parent2[i])
        I = c_max - c_min
        lower = c_min - alpha * I
        upper = c_max + alpha * I
        child1[i] = np.random.uniform(lower, upper)
        child2[i] = np.random.uniform(lower, upper)
    return child1, child2

def uniform_mutation(individual, indpb=0.05, lower=0.0, upper=1.0):
    """
    Uniform mutation for an individual.
    Each gene has probability indpb to be replaced by a random value.
    """
    mutant = individual.copy()
    for i in range(len(mutant)):
        if np.random.rand() < indpb:
            mutant[i] = np.random.uniform(lower, upper)
    return mutant

def generational_replacement(population, offspring):
    """
    Replace entire population with offspring
    """
    return offspring
