import array
import random
import numpy
import math
import time

from deap import algorithms, base, benchmarks, creator, tools

IND_SIZE = 30
XMIN = -30
XMAX = 30
SMIN = 0.5
SMAX = 3

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMin, strategy=None)
creator.create("Strategy", array.array, typecode="d")

# Individual generator
def generateES(icls, scls, size, xmin, xmax, smin, smax):

    ind = icls(random.uniform(xmin, xmax) for _ in range (size))
          
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
    return ind
   
def checkStrategy(minstrategy):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children
        return wrappper
    return decorator


toolbox = base.Toolbox()
toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
                 IND_SIZE, XMIN, XMAX, SMIN, SMAX)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxESBlend, alpha=0.1)
toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=10)
toolbox.register("evaluate", benchmarks.ackley)

toolbox.decorate("mate", checkStrategy(SMIN))
toolbox.decorate("mutate", checkStrategy(SMIN))



def eaMuCommaLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                    stats=None, halloffame=None, verbose=__debug__):

    assert lambda_ >= mu, "lambda must be greater or equal to mu."

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    
    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print (logbook.stream)

    numEvals = mu
    # Begin the generational process
    for gen in range(1, ngen + 1):
        if (numEvals <= 200000):
            # Vary the population
            offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)
            
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            
            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

                # Select the next generation population
                population[:] = toolbox.select(offspring, mu)
                
                # Update the statistics with the new population
                record = stats.compile(population) if stats is not None else {}
                logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print (logbook.stream)
            
            numEvals += logbook.select("nevals")[gen]
        else:
            print("--------------------------------------------------------")
            print("Total Evals: %d" %numEvals)
            return population, logbook


def main():
    seed = time.time_ns()
    random.seed()

    MU, LAMBDA = 100, 1000
    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    pop, logbook = eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA, 
                                              cxpb=0.6, mutpb=0.3, ngen=500, stats=stats, halloffame=hof)

   
    print("--------------------------------------------------------")
    print ("SEED: %d" %seed)
    print("--------------------------------------------------------")
    print("BEST INDIVIDUAL FITNESS: %f" %hof[0].fitness.values)
    print("--------------------------------------------------------")
    print("BEST INDIVIDUAL GENOTYPE:")
    print(hof[0])

    return pop, logbook, hof

if __name__ == "__main__":
    main()
