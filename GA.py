import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass
import ioh

from ga_functions import initialize_population, crossover, mutation, mating_seletion

budget = 5000
dimension = 50

def studentnumber1_studentnumber2_GA(problem):

    if isinstance(problem, ioh.iohcpp.problem.LABS):
        population_size = 4
        num_elitism = 1
        mutation_rate = 0.06980201831374439
        crossover_rate = 0.95

    elif isinstance(problem, ioh.iohcpp.problem.IsingRing):
        population_size = 2
        num_elitism = 1
        mutation_rate = 0.049524893651331885
        crossover_rate = 0.98

    # Initialize the population
    initial_pop = initialize_population((population_size), dimension)
    X = [x for x in initial_pop]
    #print(f"Inital population example: {X[0]}")

    # Evaluate the population
    f = [problem(x) for x in X]
    #print(f"Initial population fitness examples: {f[:3]}") 
    #print(f"Problem evaluations examples: {problem.state.evaluations}")

    while problem.state.evaluations < budget:
        
        # Elitism
        elite_idxs = np.argsort(f)[-num_elitism:]
        elites_X = [X[i] for i in elite_idxs]
        elites_f = [f[i] for i in elite_idxs]

        # Mating Selection
        parents = mating_seletion(X,f)

        # Crossover
        offspring = parents.copy()

        for i in range(0, population_size - (population_size%2), 2):
            child1, child2 = crossover(parents[i], parents[i+1], crossover_rate)
            offspring[i] = child1
            offspring[i+1] = child2
                
                

        # Mutation
        offspring = [mutation(x, mutation_rate, problem.state.evaluations) for x in offspring]

        # Evaluate the offspring
        X = offspring

        f = []
        for x in X:
            if problem.state.evaluations >= budget:
                break
            f.append(problem(x))
            

        #f = [problem(x) for x in X]

        # Elitism
        survivors_idx = np.argsort(f)[-(population_size - num_elitism):]
        survivors_X = [X[i] for i in survivors_idx]
        survivors_f = [f[i] for i in survivors_idx]
        X = elites_X + survivors_X
        f = elites_f + survivors_f
    print(f"Best final fitness: {max(f)}")



def create_problem(fid: int):
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="genetic_algorithm",  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    np.random.seed(0)
    F18, _logger = create_problem(18)
    for run in range(20): 
        studentnumber1_studentnumber2_GA(F18)
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    np.random.seed(0)
    F19, _logger = create_problem(19)
    for run in range(20): 
        studentnumber1_studentnumber2_GA(F19)
        F19.reset()
    _logger.close()