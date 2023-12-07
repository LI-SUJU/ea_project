import numpy as np

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

#Boundary Mutation
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

