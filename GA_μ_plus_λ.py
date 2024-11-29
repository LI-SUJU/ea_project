import numpy as np
from ioh import get_problem, ProblemClass

def initialize_population(n, dimension):
    """Initialize the population with random binary strings."""
    return [np.random.randint(0, 2, dimension) for _ in range(n)]

def tournament_selection(population, fitness, k=3):
    """Tournament selection for parent selection."""
    selected = []
    for _ in range(len(population)):
        participants = np.random.choice(range(len(population)), k, replace=False)
        best = max(participants, key=lambda x: fitness[x])
        selected.append(population[best].copy())
    return selected

def crossover(parent1, parent2, crossover_rate):
    """Uniform crossover with crossover rate."""
    if np.random.rand() < crossover_rate:
        child1, child2 = parent1.copy(), parent2.copy()
        for i in range(len(parent1)):
            if np.random.rand() < 0.5:
                child1[i], child2[i] = parent2[i], parent1[i]
        return child1, child2
    return parent1.copy(), parent2.copy()

def adaptive_mutation(individual, base_mutation_rate, generation, max_generations):
    """Adaptive mutation with decreasing rate over generations."""
    mutation_rate = base_mutation_rate * (1 - generation / max_generations)
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual

def μ_plus_λ_GA(problem, dimension=50, budget=5000, population_size=100, mutation_rate=0.05, crossover_rate=0.8, num_elitism=2):
    """Improved Genetic Algorithm (μ+λ-GA) with strict budget enforcement."""
    # Initialize population and evaluate fitness
    population = initialize_population(population_size, dimension)
    fitness = [problem(ind) for ind in population]

    generation = 0
    max_generations = budget // population_size

    while problem.state.evaluations < budget:
        generation += 1

        # Elitism: Select the top individuals to preserve
        elites = sorted(zip(population, fitness), key=lambda x: -x[1])[:num_elitism]
        elite_population = [e[0] for e in elites]

        # Selection: Perform tournament selection
        selected = tournament_selection(population, fitness)

        # Crossover and Mutation
        offspring = []
        for i in range(0, len(selected) - 1, 2):
            # Perform crossover
            child1, child2 = crossover(selected[i], selected[i + 1], crossover_rate)

            # Check budget before mutation and evaluation
            if problem.state.evaluations + 2 > budget:
                break  # Stop if adding two evaluations would exceed the budget

            # Apply mutation
            child1 = adaptive_mutation(child1, mutation_rate, generation, max_generations)
            child2 = adaptive_mutation(child2, mutation_rate, generation, max_generations)

            # Add offspring to the list
            offspring.append(child1)
            offspring.append(child2)

        # Evaluate offspring fitness
        offspring_fitness = []
        for ind in offspring:
            if problem.state.evaluations >= budget:
                break  # Stop if budget is reached
            offspring_fitness.append(problem(ind))

        # Combine population and offspring (μ+λ)
        combined_population = elite_population + offspring
        combined_fitness = [problem(ind) for ind in combined_population if problem.state.evaluations < budget]

        # Survival selection: Select the best individuals for the next generation
        sorted_combined = sorted(zip(combined_population, combined_fitness), key=lambda x: -x[1])
        population = [ind[0] for ind in sorted_combined[:population_size]]
        fitness = [ind[1] for ind in sorted_combined[:population_size]]

    print(f"Best fitness achieved: {max(fitness)}")
    return max(fitness)

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
    problem = get_problem(18, dimension=50, problem_class=ProblemClass.PBO)
    μ_plus_λ_GA(problem, dimension=50)
