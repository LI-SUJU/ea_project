import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass

from ga_functions import initialize_population

budget = 5000
dimension = 50

sigma_init = 0.5

mu_ = 15
lambda_ = 100
tau =  1.0 / np.sqrt(dimension)

# Make the experiments reproducible
np.random.seed(0)

def evaluate_problem(problem, x):
    x_binary = x.round().astype(int)
    return problem(x_binary)

# TODO can use fitness to choose
def recombination(parent, parent_sigma):
    p1, p2 = np.random.choice(len(parent), 2, replace=False)
    offspring = (parent[p1] + parent[p2])/2
    sigma = (parent_sigma[p1] + parent_sigma[p2])/2
    return offspring, sigma
            
def studentnumber1_studentnumber2_ES(problem):
    
    # Initialize the population
    initial_pop = np.random.rand(mu_, dimension)
    X = [x for x in initial_pop]
    #print(f"Inital population example: {X[0]}")
    sigma = np.random.rand(mu_, dimension) * sigma_init

    # Evaluate the population
    f = [evaluate_problem(problem, x) for x in X]
    #print(f"Initial population fitness examples: {f[:3]}") 
    #print(f"Problem evaluations examples: {problem.state.evaluations}")
    while problem.state.evaluations < budget:
        
        # Recombination
        # Add [lambda_] new offspring recombining old ones
        offspring = []
        offspring_sigma = []
        for i in range(lambda_):
            new_offspring, new_sigma = recombination(X, sigma)
            offspring.append(new_offspring)
            offspring_sigma.append(new_sigma)

        # Mutation
        offspring_sigma *= np.exp(np.random.normal(0, tau, size=(lambda_, dimension)))
        for i in range(lambda_):
            for j in range(dimension):
                offspring[i][j] += np.random.normal(0, offspring_sigma[i][j])
        offspring = np.clip(offspring, -1, 1)

        # Evaluate the offspring
        X = offspring
        f = [evaluate_problem(problem, x) for x in offspring]

        # Selection
        # Only chooses the best mu_ individuals
        # Last elements becasue we want to maximize
        survivors_idx = np.argsort(f)[-mu_:]
        
        X = [X[i] for i in survivors_idx]
        f = [f[i] for i in survivors_idx]
        sigma = [offspring_sigma[i] for i in survivors_idx]
        print(f"Problem evaluations examples: {problem.state.evaluations}")
        print(f"Best fitness: {max(f)}")
        #print(f"Best solution: {X[np.argmax(f)]}")
        #print(f"Best sigma: {sigma[np.argmax(f)]}")


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
    F18, _logger = create_problem(18)
    for run in range(20): 
        studentnumber1_studentnumber2_ES(F18)
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    F19, _logger = create_problem(19)
    for run in range(20): 
        studentnumber1_studentnumber2_ES(F19)
        F19.reset()
    _logger.close()


