import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass

#Bayesian
from skopt import BayesSearchCV

from ga_functions import initialize_population, crossover, mutation, mating_seletion

"""

population_size = 50
num_elitism = 10
mutation_rate = 0.015
crossover_rate = 0.75
"""

budget = 5000
dimension = 50

# Make the experiments reproducible


def studentnumber1_studentnumber2_GA(problem, population_size, num_elitism, mutation_rate, crossover_rate):
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

        #for i in range(0, population_size - (population_size%2), 2):
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
    #print(f"Best final fitness: {max(f)}")
    return max(f)





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


"""if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    F18, _logger = create_problem(18)
    for run in range(20): 
        studentnumber1_studentnumber2_GA(F18)
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    F19, _logger = create_problem(19)
    for run in range(20): 
        studentnumber1_studentnumber2_GA(F19)
        F19.reset()
    _logger.close()"""


"""def optimize(params):
    F18, _logger = create_problem(18)
    values = []
    np.random.seed(params[4])
    for run in range(20): 
        #value = studentnumber1_studentnumber2_GA(F18, params[0], params[1], params[2], params[3])
        value = studentnumber1_studentnumber2_GA(F18, params[0], params[1], params[2], 0)
        values.append(value)
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    result = np.mean(np.array(values))
    
    print(np.round(result,3), np.round(max(values), 3), params[0], params[1], params[2], params[3])
    
    #print(np.round(result,3), np.round(max(values), 3), params[0], params[1], params[2],0)
    with open('ga_1p.csv', 'a') as file:
        file.write(f'{np.round(result,3)}, {np.round(max(values), 3)},{ params[0]}, {params[1]}, {params[2]}, {params[3]}, {params[4]}; \n')

    
    
    return 1/result
    #return 1/max(values)"""

def optimize(params):
    F19, _logger = create_problem(19)
    values = []
    np.random.seed(0)
    for run in range(20): 
        value = studentnumber1_studentnumber2_GA(F19, params[0], params[1], params[2], params[3])

        values.append(value)
        F19.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    result = np.mean(np.array(values))
    
    print(np.round(result,3), np.round(max(values), 3), params[0], params[1], params[2], params[3])
    
    if result >= 47.8:
        with open('ga_new_19p.csv', 'a') as file:
            file.write(f'{np.round(result,3)}, {np.round(max(values), 3)}, {params[0]}, {params[1]}, {params[2]}, {params[3]}; \n')

    
    
    return 1/result
    #return 1/max(values)

    

from skopt.space import Real, Integer


param_space = [
    Integer(2, 3, name='population_size'),
    Integer(1, 2, name='num_elitism'),
    Real(0.001, 0.1, name='mutation_rate'),
    Real(0.5, 0.98, name='crossover_rate'),
]

from skopt import gp_minimize

result = gp_minimize(
    optimize,  # Your objective function
    param_space,
    n_calls=300,  # Number of optimization iterations
    n_random_starts=5,
    random_state=0,  # Set a random seed for reproducibility
    n_jobs=-1
)

