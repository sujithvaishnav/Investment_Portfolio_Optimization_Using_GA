import random
import numpy as np
from deap import base, creator, tools
from deap.tools import HallOfFame
from ga.population import setup_deap, genotype_to_phenotype
from ga.fitness import fitness_function

import random

def safe_mutation(individual, low=0.0, up=1.0, indpb=0.1):
    """
    Mutate each gene with probability indpb.
    Ensures all values are floats and within [low, up].
    """
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = float(random.uniform(low, up))
    return individual,


def run_ga(returns_df,
           num_generations=50,
           pop_size=660,
           num_parents=150,
           max_weight=0.5,
           alpha=1.0, beta=1.0, gamma=0.0, delta=0.0,
           crossover_prob=0.5, mutation_prob=0.4, indpb=0.05,
           esg_tickers=None,
           seed=None):
    """
    returns: best_weights (np.array), fitness_history (list of best fitness per generation)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # dynamic number of assets
    num_assets = returns_df.shape[1]
    toolbox = setup_deap(num_assets)

    # register operators
    toolbox.register("evaluate", lambda ind: 
        tuple(float(f) for f in np.ravel(fitness_function(
            genotype_to_phenotype(np.array(ind), max_weight=max_weight),
            returns_df,
            alpha=alpha, beta=beta, gamma=gamma, delta=delta,
            max_weight=max_weight, esg_tickers=esg_tickers
        )))
    )

    toolbox.register("select", tools.selTournament, tournsize=3)
    # We will call cxBlend and mutUniformFloat directly from tools

    # create initial population
    population = toolbox.population(n=pop_size)
    hof = HallOfFame(maxsize=5)

    # evaluate initial population
    invalid = [ind for ind in population if not ind.fitness.valid]
    for ind in invalid:
        ind.fitness.values = toolbox.evaluate(ind)

    hof.update(population)
    fitness_history = [min(ind.fitness.values[0] for ind in population)]

    # main loop
    for gen in range(1, num_generations + 1):
        # select parents
        num_parents = min(max(2, int(num_parents)), len(population))
        parents = toolbox.select(population, num_parents)

        # generate offspring until population size
        offspring = []
        while len(offspring) < pop_size:
            p1, p2 = random.sample(parents, 2)
            c1 = toolbox.clone(p1)
            c2 = toolbox.clone(p2)

            # crossover
            if random.random() < crossover_prob:
                tools.cxBlend(c1, c2, alpha=0.5)
                if hasattr(c1, "fitness"):
                    del c1.fitness.values
                if hasattr(c2, "fitness"):
                    del c2.fitness.values

            # mutation (per-individual)
            if random.random() < mutation_prob:
                c1, = safe_mutation(c1, low=0.0, up=1.0, indpb=indpb)
                if hasattr(c1, "fitness"):
                    del c1.fitness.values
            if random.random() < mutation_prob:
                c2, = safe_mutation(c2, low=0.0, up=1.0, indpb=indpb)
                if hasattr(c2, "fitness"):
                    del c2.fitness.values

            offspring.append(c1)
            if len(offspring) < pop_size:
                offspring.append(c2)

        # evaluate invalid offspring
        invalid_off = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_off:
            ind.fitness.values = toolbox.evaluate(ind)

        population[:] = offspring
        hof.update(population)

        gen_best = min(ind.fitness.values[0] for ind in population)
        fitness_history.append(gen_best)
        print(f"Generation {gen}: Best fitness = {gen_best:.6f}")

    # best individual
    best_ind = hof[0] if len(hof) > 0 else tools.selBest(population, 1)[0]
    best_arr = np.array(best_ind, dtype=float)
    best_weights = genotype_to_phenotype(best_arr, max_weight=max_weight)

    return best_weights, fitness_history
