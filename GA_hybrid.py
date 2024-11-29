import numpy as np
from ioh import get_problem, ProblemClass


def initialize_population(n, dimension):
    """Initialize population with random binary strings."""
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
    """Uniform crossover."""
    if np.random.rand() < crossover_rate:
        child1, child2 = parent1.copy(), parent2.copy()
        for i in range(len(parent1)):
            if np.random.rand() < 0.5:
                child1[i], child2[i] = parent2[i], parent1[i]
        return child1, child2
    return parent1.copy(), parent2.copy()


def dynamic_mutation(individual, base_rate, progress):
    """Dynamic mutation rate based on progress."""
    mutation_rate = base_rate * (1 - progress)
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual


def local_search(individual, problem, max_steps=10):
    """Simple local search (hill climbing)."""
    best = individual.copy()
    best_fitness = problem(best)

    for _ in range(max_steps):
        neighbor = best.copy()
        idx = np.random.randint(len(neighbor))
        neighbor[idx] = 1 - neighbor[idx]  # Flip a random bit
        neighbor_fitness = problem(neighbor)

        if neighbor_fitness > best_fitness:
            best, best_fitness = neighbor, neighbor_fitness

    return best


def hybrid_ga(problem, dimension=50, budget=5000, population_size=100, mutation_rate=0.05, crossover_rate=0.8):
    """Hybrid GA with local search."""
    population = initialize_population(population_size, dimension)
    fitness = [problem(ind) for ind in population]

    best_so_far = max(fitness)
    while problem.state.evaluations < budget:
        progress = problem.state.evaluations / budget

        # Selection
        selected = tournament_selection(population, fitness)

        # Crossover and Mutation
        offspring = []
        for i in range(0, len(selected) - 1, 2):
            child1, child2 = crossover(selected[i], selected[i + 1], crossover_rate)
            child1 = dynamic_mutation(child1, mutation_rate, progress)
            child2 = dynamic_mutation(child2, mutation_rate, progress)
            offspring.append(child1)
            offspring.append(child2)

        # Local Search
        for i in range(len(offspring)):
            if np.random.rand() < 0.3:  # Apply local search to 30% of offspring
                offspring[i] = local_search(offspring[i], problem)

        # Evaluate offspring
        offspring_fitness = [problem(ind) for ind in offspring]

        # Combine populations (elitism)
        combined_population = population + offspring
        combined_fitness = fitness + offspring_fitness

        # Survival selection
        sorted_combined = sorted(zip(combined_population, combined_fitness), key=lambda x: -x[1])
        population = [ind[0] for ind in sorted_combined[:population_size]]
        fitness = [ind[1] for ind in sorted_combined[:population_size]]

        # Track the best solution
        best_so_far = max(best_so_far, max(fitness))

    print(f"Best fitness achieved: {best_so_far}")
    return best_so_far

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
    hybrid_ga(problem, dimension=50)
