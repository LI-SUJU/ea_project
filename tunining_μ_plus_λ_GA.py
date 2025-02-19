import numpy as np
import random
import optuna
from ioh import logger, get_problem
from GA_μ_plus_λ import μ_plus_λ_GA, create_problem

# Set global random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def tune_hyperparameters():
    """Tune hyperparameters using Optuna."""
    max_total_evaluations = 1000000  # Limit for total evaluations across all trials
    evaluation_counter = [0]  # Use a mutable object to track evaluations across trials

    def objective(trial):
        if evaluation_counter[0] >= max_total_evaluations:
            raise optuna.exceptions.TrialPruned()  # Stop trial if evaluation limit is reached

        # Suggest hyperparameters
        pop_size = trial.suggest_int("population_size", 3, 100, step=1)
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
        fitness_F18 = μ_plus_λ_GA(F18, 50, 5000, pop_size, mutation_rate, crossover_rate, num_elitism=2, seed=RANDOM_SEED)
        fitness_F23 = μ_plus_λ_GA(F23, 49, 5000, pop_size, mutation_rate, crossover_rate, num_elitism=2, seed=RANDOM_SEED)

        # Update the global evaluation counter
        evaluation_counter[0] += max_evaluations_per_problem * 2  # Two problems per trial

        if evaluation_counter[0] >= max_total_evaluations:  # Stop if evaluation limit is reached
            raise optuna.exceptions.TrialPruned()

        # Combined fitness (minimizing negative of fitness as Optuna maximizes objective)
        combined_fitness = fitness_F18 + fitness_F23
        print(f"  Combined fitness: {combined_fitness}")
        return combined_fitness

    print("Starting hyperparameter tuning...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=150)  # Number of trials can be increased if evaluation limit allows

    print("Hyperparameter tuning completed.")
    print(f"Best hyperparameters found: {study.best_params}")
    # save best hyperparameters
    with open("best_hyperparameters_μ_plus_λ_GA.txt", "w") as f:
        f.write(str(study.best_params))

    return study.best_params

if __name__ == "__main__":
    # Tune hyperparameters
    best_params = tune_hyperparameters()
    print("\nBest Hyperparameters:")
    print(best_params)

    # Solve LABS problem with tuned hyperparameters
    algorithm_name = "(μ+λ)GA"

    F18 = create_problem(18, 50)
    _logger_F18 = logger.Analyzer(store_positions=True, algorithm_name=algorithm_name)
    F18.attach_logger(_logger_F18)
    for i in range(20):
        print(f"  Run {i + 1}/20 for LABS problem...")
        np.random.seed(RANDOM_SEED + i)  # Use different seeds for each run
        random.seed(RANDOM_SEED + i)
        best_fitness = μ_plus_λ_GA(
            F18, 50, 5000, best_params["population_size"], best_params["mutation_rate"], best_params["crossover_rate"], num_elitism=2
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
        np.random.seed(RANDOM_SEED + i)  # Use different seeds for each run
        random.seed(RANDOM_SEED + i)
        best_fitness = μ_plus_λ_GA(
            F23, 49, 5000, best_params["population_size"], best_params["mutation_rate"], best_params["crossover_rate"], num_elitism=2
        )
        print(f"    Best fitness achieved in this run: {best_fitness}")
        F23.reset()
    _logger_F23.close()
    print("Finished solving N-Queens problem.")
