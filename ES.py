import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass
import ioh

budget = 5000
dimension = 50

params = {
    "F18": [0.0898819276640423, 12, 46],
    "F19": [0.15769530578033306, 29, 66],
}

# Make the experiments reproducible
np.random.seed(0)

def evaluate_problem(problem, x):
    x = np.clip(x, 0, 1)
    x_binary = x.round().astype(int)
    return problem(x_binary)

# TODO can use fitness to choose
def recombination_mean(parent, parent_sigma):
    p1, p2 = np.random.choice(len(parent), 2, replace=False)
    offspring = (parent[p1] + parent[p2])/2
    sigma = (parent_sigma[p1] + parent_sigma[p2])/2
    return offspring, sigma

def recombination_discrete(parent, parent_sigma):
    p1, p2 = np.random.choice(len(parent), 2, replace=False)
    offspring = np.zeros(dimension)
    sigma = np.zeros(dimension)
    for i in range(dimension):
        r = np.random.rand()
        offspring[i] = parent[p1][i] if r < 0.5 else parent[p2][i]
        sigma[i] = parent_sigma[p1][i] if r < 0.5 else parent_sigma[p2][i]
    return offspring, sigma

def mutation(offspring, offspring_sigma, global_perturbation, tau_local, tau_global):
    new_offspring_sigma = np.zeros_like(offspring_sigma)
    for i, s in enumerate(offspring_sigma):
        new_offspring_sigma[i] = s * np.exp(tau_local * np.random.normal(0, 1) + 
                                            tau_global * global_perturbation) # This was global permutation before
                                            #tau_global * global_perturbation) # This was global permutation before
    
    new_offspring = np.zeros_like(offspring)
    for i, x in enumerate(offspring):
        new_offspring[i] = x + new_offspring_sigma[i] * np.random.normal(0, 1)

    return new_offspring, new_offspring_sigma
            
def s3442209_s4115597_ES(problem):
    
    if isinstance(problem, ioh.iohcpp.problem.LABS):
        sigma_init, mu_, lambda_ = params['F18']
    elif isinstance(problem, ioh.iohcpp.problem.IsingRing):
        sigma_init, mu_, lambda_ = params['F19']
    else:
        sigma_init, mu_, lambda_ = params['F18']

    lambda_ = lambda_ + mu_
    tau_local = 1.0 / np.sqrt(2*np.sqrt(dimension))
    tau_global = 1.0 / np.sqrt(2*dimension)

    # Initialize the population
    initial_pop = np.random.rand(mu_, dimension)
    X = [x for x in initial_pop]
    #print(f"Inital population example: {X[0]}")
    #sigma = np.random.rand(mu_, dimension) * sigma_init
    sigma = np.ones((mu_, dimension)) * sigma_init

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
            new_offspring, new_sigma = recombination_discrete(X, sigma)
            offspring.append(new_offspring)
            offspring_sigma.append(new_sigma)
        #offspring = [np.clip(o, 0, 1) for o in offspring]

        # Mutation
        for i in range(lambda_):
            global_perturbation = np.random.normal(0, 1)
            o, os = mutation(offspring[i], offspring_sigma[i], global_perturbation, tau_local, tau_global)
            offspring[i] = o
            offspring_sigma[i] = os
        #offspring = [np.clip(o, 0, 1) for o in offspring]

        # Evaluate the offspring
        # Ensure that budget is not exceeded
        offspring_f = []
        for x in offspring:
            if problem.state.evaluations >= budget:
                break
            offspring_f.append(evaluate_problem(problem, x))
        offspring = [offspring[i] for i in range(len(offspring_f))]
        offspring_sigma = [offspring_sigma[i] for i in range(len(offspring_f))]

        X += [o for o in offspring]
        sigma = [s for s in sigma] + [s for s in offspring_sigma]
        f += offspring_f

        # Selection
        # Only chooses the best mu_ individuals
        # Last elements becasue we want to maximize
        survivors_idx = np.argsort(f)[-mu_:]
        #f_norm = np.array(f) - np.min(f)
        #survivors_idx = np.random.choice(list(range(len(f_norm))), p=f_norm/np.sum(f_norm), size=mu_, replace=False)

        X = [X[i] for i in survivors_idx]
        f = [f[i] for i in survivors_idx]
        sigma = [sigma[i] for i in survivors_idx]
        #print(f"Problem evaluations examples: {problem.state.evaluations}")
        #print(f"Best fitness: {max(f)}")
        #print(f"Best solution: {X[np.argmax(f)]}")
        #print(f"Best sigma: {sigma[np.argmax(f)]}")
    print(f"Best final fitness: {max(f)}")
    #return max(f)

def create_problem(fid: int):
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="evolutionary_strategies",  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    F18, _logger = create_problem(18)
    for run in range(20): 
        s3442209_s4115597_ES(F18)
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    F19, _logger = create_problem(19)
    for run in range(20): 
        s3442209_s4115597_ES(F19)
        F19.reset()
    _logger.close()