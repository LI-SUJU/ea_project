# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 13:06:13 2024

@author: admin
"""

import numpy as np
from ioh import get_problem, logger, ProblemClass
from time import time
from time import perf_counter


np.random.seed(2024)
dimension = 10


def evaluate_in_batches(problem, population, batch_size):
    fitness = []
    for i in range(0, len(population), batch_size):
        batch = population[i:i + batch_size]  # 取出当前批次
        batch_fitness = [problem(ind) for ind in batch]  # 评估当前批次
        fitness.extend(batch_fitness)  # 将结果添加到 fitness 列表中
    return np.array(fitness)


def ES(problem, mu, lambda_, sigma, max_evaluations, batch_size=10):
    t0 = time()
    dim = 10
    fitness_history = []
    population = np.random.uniform(-5, 5, (mu, dim))

    fitness = evaluate_in_batches(problem, population, batch_size)

    while problem.state.evaluations < max_evaluations:

        fitness_history.append(fitness[0])

        offspring = []
        for _ in range(lambda_):
            parents = population[np.random.choice(range(mu), size=2, replace=False)]
            child = np.mean(parents, axis=0) + np.random.normal(0, sigma, size=dim)
            offspring.append(child)
        offspring = np.array(offspring)
        offspring_fitness = evaluate_in_batches(problem, offspring, batch_size)

        combined_population = np.vstack((population, offspring))
        combined_fitness = np.hstack((fitness, offspring_fitness))
        indices = np.argsort(combined_fitness)
        population = combined_population[indices[:mu]]
        fitness = combined_fitness[indices[:mu]]
    t1 = time()
    print(f"Running time for (mu+lambda):{t1-t0}")
    return np.min(fitness), fitness_history


def ES2(problem, mu, lambda_, sigma, max_evaluations, batch_size=100):
    t2 = perf_counter()
    dim = 10
    fitness_history = []
    population = np.random.uniform(-5, 5, (mu, dim))

    while problem.state.evaluations < max_evaluations:
        #print(f"Current evaluations: {problem.state.evaluations}")
        # 评估父代适应度
        fitness = np.array([problem(x) for x in population])
        # 记录当前种群的最佳适应度值
        fitness_history.append(fitness[0])

        offspring = []
        for _ in range(lambda_):
            parents = population[np.random.choice(range(mu), size=2, replace=False)]
            child = np.mean(parents, axis=0) + np.random.normal(0, sigma, size=dim)
            offspring.append(child)
        offspring = np.array(offspring)

        offspring_fitness = np.array([problem(x) for x in offspring])

        indices = np.argsort(offspring_fitness)
        population = offspring[indices[:mu]]
    t3 = perf_counter()
    print(f"Running time for (mu,lambda):{t3-t2}")
    return np.min(fitness), fitness_history

def create_problem(fid: int):
    # Declaration of problems to be tested.
    #print(f"Function ID: {fid}, Dimension: {dimension}, Instance: 1, Problem Class: {ProblemClass.BBOB}")
    problem = get_problem(fid, 1, 10, ProblemClass.BBOB)  # dimension 和 instance 为位置参数
    log_folder = "ioh_logs"
    l = logger.Analyzer(
        root=log_folder,
        folder_name="run",  # 日志文件夹名
        algorithm_name="Evolution Strategy",
        algorithm_info="Results logged with IOHProfiler"
    )
    problem.attach_logger(l)
    '''
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="evolution strategy",  # name of your algorithm
        algorithm_info="Practical assignment part2 of the EA course",
    )
    # Attach the logger to the problem
    problem.attach_logger(l)'''
    return problem, l



hyperparameter_space = {
    "sigmas": [ 0.01, 0.1, 1],
    "mu_values": [16, 64, 4],
    "lambda_values": [128, 64, 256]}
if __name__ == "__main__":
    fid = 23  # BBOB function ID
    dimension = 10
    budget = 50000
    results = []

    for sigma in hyperparameter_space['sigmas']:
        for mu in hyperparameter_space['mu_values']:
            for lambda_ in hyperparameter_space['lambda_values']:
                print(f"Testing with Sigma={sigma}, Mu={mu}, Lambda={lambda_}")

                avg_fitness_ES = 0
                avg_fitness_ES2 = 0

                for run in range(20):
                    print(f"Run {run + 1} for (Sigma={sigma}, Mu={mu}, Lambda={lambda_})")

                    problem, logger_ = create_problem(fid)
                    best_fitness_ES,_ = ES(problem, mu, lambda_, sigma, budget)
                    avg_fitness_ES += best_fitness_ES
                    logger_.close()

                    problem, logger_ = create_problem(fid)
                    best_fitness_ES2,_ = ES2(problem, mu, lambda_, sigma, budget)
                    avg_fitness_ES2 += best_fitness_ES2
                    logger_.close()

                # 计算平均值
                avg_fitness_ES /= 20
                avg_fitness_ES2 /= 20

                # 存储结果
                results.append({
                    "Sigma": sigma,
                    "Mu": mu,
                    "Lambda": lambda_,
                    "Avg_Fitness_ES": avg_fitness_ES,
                    "Avg_Fitness_ES2": avg_fitness_ES2
                })

    # 输出结果
    print("\nFinal Results:")
    for result in results:
        print(result)
