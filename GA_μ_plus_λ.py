import numpy as np
from ioh import get_problem, ProblemClass


def initialize_population(n, dimension):
    """Initialize the population with random binary strings."""
    return [np.random.randint(0, 2, dimension) for _ in range(n)]


def tournament_selection(population, fitness, k=3):
    """Tournament selection for parent selection."""
    selected = []
    n = len(population)
    k = min(k, n)  # Ensure k does not exceed the population size
    for _ in range(n):
        participants = np.random.choice(range(n), k, replace=False)
        best = max(participants, key=lambda x: fitness[x])
        selected.append(population[best].copy())
    return selected


def two_point_crossover(parent1, parent2):
    """Two-point crossover."""
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have the same length.")
    point1, point2 = sorted(np.random.choice(range(len(parent1)), 2, replace=False))
    child1 = np.concatenate([parent1[:point1], parent2[point1:point2], parent1[point2:]])
    child2 = np.concatenate([parent2[:point1], parent1[point1:point2], parent2[point2:]])
    return child1, child2


def adaptive_mutation(individual, base_mutation_rate, generation, max_generations, stagnation):
    """Adaptive mutation with dynamic adjustment based on stagnation."""
    if stagnation:
        # Increase mutation rate if stagnation is detected
        mutation_rate = min(0.2, base_mutation_rate * 2)
    else:
        # Decrease mutation rate as generations progress
        mutation_rate = base_mutation_rate * (1 - generation / max_generations)
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual


def hamming_distance(ind1, ind2):
    """Calculate Hamming distance between two binary individuals."""
    return sum(c1 != c2 for c1, c2 in zip(ind1, ind2))


def fitness_sharing(fitness, population, sharing_radius=5):
    """Apply fitness sharing to maintain diversity."""
    shared_fitness = np.array(fitness, dtype=float)
    for i in range(len(population)):
        for j in range(len(population)):
            if i != j and hamming_distance(population[i], population[j]) < sharing_radius:
                shared_fitness[i] *= 0.9  # Penalize similar individuals
    return shared_fitness


def select_elites(population, fitness, num_elitism):
    """Select diverse elites."""
    elites = sorted(zip(population, fitness), key=lambda x: -x[1])
    selected_elites = [elites[0][0]]  # Always include the best individual
    for ind, fit in elites[1:]:
        if all(hamming_distance(ind, e) > 5 for e in selected_elites):  # Ensure diversity
            selected_elites.append(ind)
        if len(selected_elites) >= num_elitism:
            break
    return selected_elites


def local_search(individual, problem, max_steps=10, budget=5000):
    """Bit-flip hill climbing to improve an individual."""
    best = individual.copy()
    best_fitness = problem(best)
    for _ in range(max_steps):
        if problem.state.evaluations >= budget:  # Stop if budget is exceeded
            break
        candidate = best.copy()
        bit = np.random.randint(len(candidate))
        candidate[bit] = 1 - candidate[bit]  # Flip a random bit
        candidate_fitness = problem(candidate)
        if candidate_fitness > best_fitness:
            best = candidate
            best_fitness = candidate_fitness
    return best


def dynamic_population_size(initial_size, final_size, generation, max_generations, min_size=3):
    """Adjust population size dynamically, with a minimum size."""
    size = int(initial_size - (initial_size - final_size) * (generation / max_generations))
    return max(size, min_size)  # Ensure population size does not go below min_size


def μ_plus_λ_GA(problem, dimension=50, budget=5000, initial_population_size=100, final_population_size=50,
                mutation_rate=0.05, crossover_rate=0.8, num_elitism=2, seed=None):
    """Improved Genetic Algorithm (μ+λ-GA) with enhancements for F18."""
    # Initialize population and evaluate fitness
    if seed is not None:
        np.random.seed(seed)

    population = initialize_population(initial_population_size, dimension)
    fitness = [problem(ind) for ind in population]

    generation = 0
    max_generations = budget // initial_population_size
    best_fitness = -np.inf
    stagnation_counter = 0

    while problem.state.evaluations < budget:
        generation += 1

        # Check for stagnation
        current_best_fitness = max(fitness)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        stagnation = stagnation_counter > 5

        # Elitism: Select the top individuals to preserve
        elites = select_elites(population, fitness, num_elitism)

        # Selection: Perform tournament selection with fitness sharing
        shared_fitness = fitness_sharing(fitness, population)
        selected = tournament_selection(population, shared_fitness)

        # Crossover and Mutation
        offspring = []
        for i in range(0, len(selected) - 1, 2):
            if problem.state.evaluations >= budget:  # Stop if budget is exceeded
                break
            # Perform crossover
            child1, child2 = two_point_crossover(selected[i], selected[i + 1])

            # Apply mutation
            child1 = adaptive_mutation(child1, mutation_rate, generation, max_generations, stagnation)
            child2 = adaptive_mutation(child2, mutation_rate, generation, max_generations, stagnation)

            # Add offspring to the list
            offspring.append(child1)
            offspring.append(child2)

        # Evaluate offspring fitness (respecting budget)
        offspring_fitness = []
        for child in offspring:
            if problem.state.evaluations < budget:
                offspring_fitness.append(problem(child))
            else:
                break

        # Combine population and offspring (μ+λ)
        combined_population = elites + offspring[:len(offspring_fitness)]
        combined_fitness = [problem(ind) for ind in combined_population]

        # Survival selection: Select the best individuals for the next generation
        sorted_combined = sorted(zip(combined_population, combined_fitness), key=lambda x: -x[1])
        population_size = dynamic_population_size(initial_population_size, final_population_size, generation, max_generations)
        population = [ind[0] for ind in sorted_combined[:population_size]]
        fitness = [ind[1] for ind in sorted_combined[:population_size]]

        # Apply local search to the best individual
        if problem.state.evaluations < budget:
            population[0] = local_search(population[0], problem, budget=budget)

    print(f"Best fitness achieved: {max(fitness)}")
    return max(fitness)


def create_problem(fid: int, dimension: int):
    """Create the problem instance for F18."""
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)
    return problem


if __name__ == "__main__":
    # Create the F18 problem (LABS problem)
    problem = get_problem(18, dimension=50, problem_class=ProblemClass.PBO)
    # Run the improved GA
    μ_plus_λ_GA(problem, dimension=50, budget=5000)
