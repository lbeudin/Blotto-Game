# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 14:07:59 2021

@author: Lbeudin
"""
import numpy as np
from scipy.stats import skewnorm
import time


def competition(allocation1,   allocation2):
    """
    Calculates the score between 2 allocations
    Parameters
    ----------
    allocation1 : TYPE int[]
        DESCRIPTION.
    allocation2 : TYPE int[]
        DESCRIPTION.

    Returns 2 double that represent the score of the allocation 1 and 2
    -------
    None.
    """
    score1 = 0
    score2 = 0
    castles = len(allocation1)
    castles_rules = False
    castleWin = 1
    for i in range(0,   castles):
        previous = castles_rules
        if allocation1[i] > allocation2[i]:
            score1 += i+1
            three_castles_rules = True
        else:
            if(allocation1[i] < allocation2[i]):
                score2 += i+1
                three_castles_rules = False
        if previous == castles_rules and allocation1[i] != allocation2[i]:
            castleWin += 1
        else:
            castleWin = 1
        if(castleWin == 3):
            if three_castles_rules:
                score1 += sum(range(i+2, castles+1))
            else:
                score2 += sum(range(i+2,  castles+1))
            break
    return([score1,  score2])


def BestOfGame(simulation,  castles):
    """
    Find the best allocation with the highest score in average in the
    vectSimulation

    Parameters
    ----------
    simulation : TYPE int[,  ]
        DESCRIPTION.
        all allocation possible
    castles : TYPE int
        DESCRIPTION.
        number of castles to be allocated
    Returns the best allocation and the linked probabilty
    -------
    None.

    """
    size = len(simulation)
    vect = np.zeros((size),  dtype=int)
    maxi = 0
    probMax = 0
    allocationMax = np.array((castles),  dtype=int)
    for i in range(0,  size):
        prob = 0
        if i % 1000 == 0:
            print("Indicated step :", i)
        for j in range(0, size):
            winner_loser = competition(simulation[i, :], simulation[j, :])
            vect[j] = winner_loser[0]
            if winner_loser[0] > winner_loser[1]:
                prob += 1
        if maxi < np.mean(vect):
            maxi = np.mean(vect)
            allocationMax = simulation[i, :]
            probMax = prob/len(vect)

    return([allocationMax, probMax])


def probaBestOfGameSummary(simulation_vect, best_allocation, castles):
    """
    Calculates for a given allocation the probability of winning
    the average score of the allocation versus competitor
    Parameters
    ----------
    vectSimulation : TYPE int[,  ]
        DESCRIPTION.
        all allocation possible
    vectallocation : TYPE
        DESCRIPTION.
        best allocation found
    castles : TYPE
        DESCRIPTION.
        numer of castle to allocate
    Returns print different metrics to evaluate the power of a given allocation
    -------
    None.

    """
    size = len(simulation_vect)
    prob = 0
    meanWin = 0
    meanLose = 0
    for i in range(0, size):
        winner_loser = competition(best_allocation, simulation_vect[i, :])
        if winner_loser[0] > winner_loser[1]:
            prob = prob+1
            meanWin = winner_loser[0] + meanWin
            meanLose = winner_loser[1] + meanLose
    print("Allocation :", best_allocation)
    print("Average result allocation", meanWin/size)
    print("Average results of competitor", meanLose/size)
    print("Probability of winning ", prob/size)


castles = 10
soldiers = 100
np.random.seed(123)

values_random_law = 1000
random_skew_back = skewnorm.rvs(a=-8, loc=castles, size=values_random_law)
random_skew_front = skewnorm.rvs(a=8, loc=castles, size=values_random_law)
random_norm = skewnorm.rvs(a=0,  loc=castles,   size=values_random_law)

# here ze standardize to get results included between 0 and castles
random_skew_back = (random_skew_back - min(random_skew_back))
random_skew_back = random_skew_back / max(random_skew_back) * castles
weight_skew_back = np.histogram(random_skew_back)[0]
weight_skew_back = weight_skew_back / sum(np.histogram(random_skew_back)[0])


random_skew_front = (random_skew_front - min(random_skew_front))
random_skew_front = random_skew_front / max(random_skew_front) * castles
weight_skew_front = np.histogram(random_skew_front)[0]
weight_skew_front = weight_skew_front / sum(np.histogram(random_skew_front)[0])

random_norm = (random_norm - min(random_norm))
random_norm = random_norm / max(random_norm) * castles
weight_norm = np.histogram(random_norm)[0] / sum(np.histogram(random_norm)[0])

uniform_proba = np.array([1/castles for i in range(0, castles)])


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Part 1 - Try to Find the best allocation among multiple distribution
Results : we conclude on the best distribution
Time to run : 1 hour
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
simulation = 5000

list_prob = [weight_skew_back, weight_skew_front, weight_norm, uniform_proba]

combination_sim_vect = np.zeros((1, 10), dtype=int)
for i in range(0,  len(list_prob)):
    randomvect = np.random.multinomial(soldiers, list_prob[i], size=simulation)
    combination_sim_vect = np.concatenate((combination_sim_vect, randomvect))

tic = time.perf_counter()
game_matrix = BestOfGame(combination_sim_vect,   castles)
toc = time.perf_counter()
print("Time :",  str(toc-tic))
print(game_matrix)


# Time : 131.44247440000026
# [array([13,  29, 28,   14,    9,    3,    0,    4,    0,    0]),
# 0.9217695576105973]

# 3335.9828703000003
# [array([16,   33,   26,   12,    2,    6,    2,    1,    1,    1]),
# 0.944152792360382]


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Part 2 - Find the best allocation among the best distribution
Results : we get the best allocation
Time to run : 1 hour
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
simulation = 20000

# Choices made - for the simulatin I train my best choice
# as if we were using all ressources of soldiers
# tab that containes weighted vector
list_prob = [weight_skew_front]

combination_sim_vect = np.zeros((1, 10), dtype=int)
for i in range(0, len(list_prob)):
    randomvect = np.random.multinomial(soldiers, list_prob[i], size=simulation)
    combination_sim_vect = np.concatenate((combination_sim_vect, randomvect))

tic = time.perf_counter()
# Combination_sim_vect is a large tab (1e6)*4 of simulated allocation
game_matrix = BestOfGame(combination_sim_vect, castles)
toc = time.perf_counter()
print("Time :",  str(toc-tic))
print(game_matrix)

# RESULTS :
# Time : 3819.1758566
# [array([13, 31, 26, 15,  8,  4,  2,  0,  1,  0]), 0.9142042897855107]

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Part 3 - test the best allocation & calculate the proba
Results : give the probability of success of the best allocation found
Time to run : 30 sec
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

simulation = 1000000
list_prob = [weight_skew_back, weight_skew_front, weight_norm, uniform_proba]

combination_sim_vect = np.zeros((1,  10),  dtype=int)
for i in range(0,  len(list_prob)):
    randomvect = np.random.multinomial(soldiers, list_prob[i], size=simulation)
    combination_sim_vect = np.concatenate((combination_sim_vect, randomvect))

tic = time.perf_counter()
probaBestOfGameSummary(
    combination_sim_vect, [13, 31, 26, 15,  8,  4,  2,  0,  1,  0], castles)
toc = time.perf_counter()
print("Time :",  str(toc-tic))


# RESULTS
# Allocation : [13, 31, 26, 15, 8, 4, 2, 0, 1, 0]
# Average result allocation 53.60424859893785
# Average results of competitor 0.0527962368009408
# Probability of winning  0.9762440059389985
# Time : 23.001896900000247
