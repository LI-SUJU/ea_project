import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass

from ga_functions import initialize_population

budget = 5000
dimension = 50

sigma_init = 0.1

mu_ = 15
lambda_ = 100
tau =  1.0 / np.sqrt(dimension)
tau_local = 1.0 / np.sqrt(2*np.sqrt(dimension))
tau_global = 1.0 / np.sqrt(2*dimension)

mu_ = 2
lambda_ = 70

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

def mutation(offspring, offspring_sigma, global_perturbation):
    new_offspring_sigma = np.zeros_like(offspring_sigma)
    for i, s in enumerate(offspring_sigma):
        new_offspring_sigma[i] = s * np.exp(tau_local * np.random.normal(0, 1) + 
                                            tau_global * global_perturbation)
    
    new_offspring = np.zeros_like(offspring)
    for i, x in enumerate(offspring):
        new_offspring[i] = x + np.random.normal(0, new_offspring_sigma[i])

    return new_offspring, new_offspring_sigma
            
def studentnumber1_studentnumber2_ES(problem, mu_=mu_, lambda_=lambda_, sigma_init=sigma_init, tau=tau):
    
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
            new_offspring, new_sigma = recombination_mean(X, sigma)
            offspring.append(new_offspring)
            offspring_sigma.append(new_sigma)

        # Mutation
        if False:
            offspring_sigma *= np.exp(np.random.normal(0, tau, size=(lambda_, dimension)))
            for i in range(lambda_):
                for j in range(dimension):
                    offspring[i][j] += np.random.normal(0, offspring_sigma[i][j])
            offspring = np.clip(offspring, -1, 1)
        else:
            for i in range(lambda_):
                global_perturbation = np.random.normal(0, 1)
                o, os = mutation(offspring[i], offspring_sigma[i], global_perturbation)
                offspring[i] = o
                offspring_sigma[i] = os

        # Evaluate the offspring
        offspring_f = [evaluate_problem(problem, x) for x in offspring]
        X += [o for o in offspring]
        sigma = [s for s in sigma] + [s for s in offspring_sigma]
        f += offspring_f

        # Selection
        # Only chooses the best mu_ individuals
        # Last elements becasue we want to maximize
        survivors_idx = np.argsort(f)[-mu_:]
        
        X = [X[i] for i in survivors_idx]
        f = [f[i] for i in survivors_idx]
        sigma = [sigma[i] for i in survivors_idx]
        #print(f"Problem evaluations examples: {problem.state.evaluations}")
        #print(f"Best fitness: {max(f)}")
        #print(f"Best solution: {X[np.argmax(f)]}")
        #print(f"Best sigma: {sigma[np.argmax(f)]}")
    print(f"Best final fitness: {max(f)}")
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


if __name__ == "__main__":
    
    vmax = 0
    params = {"sigma_init": 0, "mu_": 0, "lambda_": 0, "tau": 0}
    for i in range(100):

        sigma_init = np.random.uniform(0.01, 0.3)
        mu_ = np.random.randint(2, 20)
        lambda_ = mu_ + np.random.randint(0, 100)
        tau = 1.0 / np.sqrt(dimension)

        print(f"sigma_init: {sigma_init}")
        print(f"mu_: {mu_}")
        print(f"lambda_: {lambda_}")
        print(f"tau: {tau}")

        F18, _logger = create_problem(18)
        values = []
        for run in range(5):
            v = studentnumber1_studentnumber2_ES(F18, mu_=mu_, lambda_=lambda_, sigma_init=sigma_init, tau=tau)
            values.append(v)
            F18.reset()
        _logger.close()
        
        print(np.mean(values), np.std(values))
        if np.mean(values) > vmax:
            vmax = np.mean(values)
            params["sigma_init"] = sigma_init
            params["mu_"] = mu_
            params["lambda_"] = lambda_
            params["tau"] = tau
    
    print("Best params:")
    print(params)
    print(f"Best fitness: {vmax}")




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


