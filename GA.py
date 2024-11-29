import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass
import ioh


# Initialize population with random binary strings
def initialize_population(n, dimension):
    population = [np.random.randint(0,2,dimension) for i in range(n)]
    return population

# One point crossover
def crossover(p1, p2, crossover_rate):
    # Check if crossover should be performed
    if crossover_rate <= np.random.uniform(0,1):
        return p1, p2
    
    size = len(p1)
    point1 = np.random.randint(0,size)

    off1 = p1.copy()
    off2 = p2.copy()

    off1[point1:] = p2[point1:]
    off2[point1:] = p1[point1:]

    return off1, off2

"""# Uniform Crossover
def crossover(p1, p2, crossover_rate):
   #if(np.random.uniform(0,1) < crossover_rate):
    for i in range(len(p1)) :
        #if np.random.uniform(0,1) < 0.5:
        if np.random.uniform(0,1) < crossover_rate:
            t = p1[i].copy()
            p1[i] = p2[i]
            p2[i] = t

    return p1, p2 """

# Standard bit mutation using mutation rate p
"""def mutation(p, mutation_rate):
    for i in range(len(p)) :
        if np.random.uniform(0,1) < mutation_rate:
            p[i] = 1 - p[i]
    return p"""

#Non-Uniform Mutation
def mutation(p, mutation_rate, generation):
    current_mutation_rate = mutation_rate * (1 - generation / 10000)

    for i in range(len(p)) :
        if np.random.uniform(0,1) < current_mutation_rate:
            p[i] = 1 - p[i]
    return p


# Roulette wheel selection
def mating_seletion(parent, parent_f):

    # Normalize fitness values
    f_normalized = [f - min(parent_f) + 0.001 for f in parent_f]
    f_sum = sum(f_normalized) 
    f_normalized = [f / f_sum for f in f_normalized]

    # Cumulative roulette wheel values
    rw = [f_normalized[0]]
    for i in range(1,len(parent_f)):
        rw.append(rw[i-1] + f_normalized[i])
    
    # Sort parents using roulette wheel values
    select_parent = []
    for i in range(len(parent)) :
        r = np.random.uniform(0,1)
        index = 0
        while(r > rw[index]) :
            index = index + 1
        
        select_parent.append(parent[index].copy())
    return select_parent

# budget = 5000
# dimension = 50

def studentnumber1_studentnumber2_GA(problem, dimension=50, budget=5000, population_size=100, mutation_rate=0.049524893651331885, crossover_rate=0.98, num_elitism=1):

    # population_size = 100
    # num_elitism = 1
    # mutation_rate = 0.049524893651331885
    # crossover_rate = 0.98

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

    return max(f)



def create_problem(fid: int, dimension: int):
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    # l = logger.Analyzer(
    #     root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
    #     folder_name="run",  # the folder name to which the raw performance data will be stored
    #     algorithm_name="genetic_algorithm",  # name of your algorithm
    #     algorithm_info="Practical assignment of the EA course",
    # )
    # # attach the logger to the problem
    # problem.attach_logger(l)
    # return problem, l
    return problem


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    np.random.seed(0)
    F18, _logger = create_problem(18, 50)
    for run in range(20): 
        print(f"Run {run}")
        studentnumber1_studentnumber2_GA(F18, 50)
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    np.random.seed(0)
    F23, _logger = create_problem(23, 49)
    for run in range(20): 
        print(f"Run {run}")
        studentnumber1_studentnumber2_GA(F23, 49)
        F23.reset()
    _logger.close()