import numpy as np
from ioh import get_problem, logger, ProblemClass
from ES import s3442209_s4115597_ES, create_problem

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


def optimize(params):
    sigma_init = params[0]
    mu_ = params[1]
    lambda_ = params[2]
    tau_factor = 1#params[3]

    # Int error fix
    mu_ = int(mu_)
    lambda_ = int(lambda_)
    lambda_ = lambda_ + mu_
    tau_local = tau_factor / np.sqrt(2*np.sqrt(dimension))
    tau_global = tau_factor / np.sqrt(2*dimension)


    F18, _logger = create_problem(19)
    values = []
    np.random.seed(0)
    for run in range(20):
        v = s3442209_s4115597_ES(F18, mu_=mu_, lambda_=lambda_, sigma_init=sigma_init, tau_local=tau_local, tau_global=tau_global)
        values.append(v)
        F18.reset()
    _logger.close()
    
    result = np.mean(np.array(values))
    print(np.round(result,3), np.round(max(values), 3), params[0], int(params[1]), int(params[2]))

    if result >= 3.5:
        with open('es_19.csv', 'a') as file:
            file.write(f'{np.round(result,3)}, {np.round(max(values), 3)}, {params[0]}, {params[1]}, {params[2]}; \n')

    return 1/result #+ max(values)
    #return 1/max(values)
    
    


if __name__ == "__main__":
    
    vmax = 0
    
    from skopt.space import Real, Integer

    param_space = [
        Real(0, 0.5, name='sigma_init'),
        Real(2, 50, name='mu'),
        Real(1, 100, name='lambda'),
        #Real(0, 1, name='tau_factor')
    ]

    from skopt import dummy_minimize, gp_minimize

    result = gp_minimize(
        optimize,  # Your objective function
        param_space,
        n_calls=1000,  # Number of optimization iterations
        n_random_starts=5,
        random_state=0,  # Set a random seed for reproducibility
        n_jobs=-1
    )




