import numpy as np
import optuna
from ioh import logger, get_problem
from GA_hybrid import hybrid_ga, create_problem

def tune_hyperparameters():
    """Tune hyperparameters using Optuna."""
    max_total_evaluations = 1000000  # Limit for total evaluations across all trials
    evaluation_counter = [0]  # Use a mutable object to track evaluations across trials

    def objective(trial):
        if evaluation_counter[0] >= max_total_evaluations:
            raise optuna.exceptions.TrialPruned()  # Stop trial if evaluation limit is reached

        # Suggest hyperparameters
        pop_size = trial.suggest_int("population_size", 5, 100, step=5)
        mutation_rate = trial.suggest_float("mutation_rate", 0.01, 0.1, step=0.005)
        crossover_rate = trial.suggest_float("crossover_rate", 0.5, 0.9, step=0.05)

        # Print suggested hyperparameters for this trial
        print(f"Trial {trial.number}: Testing with population_size={pop_size}, mutation_rate={mutation_rate:.2f}, crossover_rate={crossover_rate:.2f}")

        # Create problems
        F18 = create_problem(fid=18, dimension=50)  # LABS problem
        F23 = create_problem(fid=23, dimension=49)  # N-Queens problem

        # Reset problems
        F18.reset()
        F23.reset()

        # Evaluate GA on both problems
        max_evaluations_per_problem = 5000
        fitness_F18 = hybrid_ga(F18, 50, 5000, pop_size, mutation_rate, crossover_rate)
        fitness_F23 = hybrid_ga(F23, 49, 5000, pop_size, mutation_rate, crossover_rate)

        # Update the global evaluation counter
        evaluation_counter[0] += max_evaluations_per_problem * 2  # Two problems per trial

        if evaluation_counter[0] >= max_total_evaluations:  # Stop if evaluation limit is reached
            raise optuna.exceptions.TrialPruned()

        # Combined fitness (minimizing negative of fitness as Optuna maximizes objective)
        # print(f"  Fitness for LABS problem (F18): {fitness_F18}")
        # print(f"  Fitness for N-Queens problem (F23): {fitness_F23}")
        combined_fitness = fitness_F18 + fitness_F23
        print(f"  Combined fitness: {combined_fitness}")
        return combined_fitness

    print("Starting hyperparameter tuning...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=150)  # Number of trials can be increased if evaluation limit allows

    print("Hyperparameter tuning completed.")
    print(f"Best hyperparameters found: {study.best_params}")
    # save best hyperparameters
    with open("best_hyperparameters_hybrid_ga.txt", "w") as f:
        f.write(str(study.best_params))

    return study.best_params

if __name__ == "__main__":
    # Tune hyperparameters
    best_params = tune_hyperparameters()
    print("\nBest Hyperparameters:")
    print(best_params)

    # Solve LABS problem with tuned hyperparameters
    # print("\nSolving LABS problem with tuned hyperparameters...")
    # np.random.seed(0)
    # F18, _logger = create_problem(18, 50)
    # for run in range(20): 
    #     print(f"Run {run}")
    #     hybrid_ga(F18, 50)
    #     F18.reset() # it is necessary to reset the problem after each independent run
    # _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    # np.random.seed(0)
    # F23, _logger = create_problem(23, 49)
    # for run in range(20): 
    #     print(f"Run {run}")
    #     hybrid_ga(F23, 49)
    #     F23.reset()
    # _logger.close()

    algorithm_name = "hybrid_ga"

    F18 = create_problem(18, 50)
    _logger_F18 = logger.Analyzer(store_positions=True, algorithm_name=algorithm_name)
    F18.attach_logger(_logger_F18)
    for i in range(20):
        print(f"  Run {i + 1}/20 for LABS problem...")
        best_fitness = hybrid_ga(
            F18, 50, 5000, best_params["population_size"], best_params["mutation_rate"], best_params["crossover_rate"]
        )
        print(f"    Best fitness achieved in this run: {best_fitness}")
        F18.reset()
    _logger_F18.close()
    print("Finished solving LABS problem.")

    # Solve N-Queens problem with tuned hyperparameters
    print("\nSolving N-Queens problem with tuned hyperparameters...")
    F23 = create_problem(23, 49)
    _logger_F23 = logger.Analyzer(store_positions=True, algorithm_name=algorithm_name)
    F23.attach_logger(_logger_F23)
    for i in range(20):
        print(f"  Run {i + 1}/20 for N-Queens problem...")
        best_fitness = hybrid_ga(
            F23, 49, 5000, best_params["population_size"], best_params["mutation_rate"], best_params["crossover_rate"]
        )
        print(f"    Best fitness achieved in this run: {best_fitness}")
        F23.reset()
    _logger_F23.close()
    print("Finished solving N-Queens problem.")
