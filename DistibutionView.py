# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 08:22:21 2021

@author: LBeudin
"""

from scipy.stats import skewnorm
import matplotlib.pyplot as plt
import numpy as np

numValues = 1000
maxValue = 10
skewness = 0
# Negative - left skewed, positive - right skewed.

random = skewnorm.rvs(a=skewness, loc=maxValue, size=numValues)

# minimum value is equal to 0.Standadize all the vlues between 0 and 1.
random = (random - min(random))
random = random / max(random) * maxValue

print(np.histogram(random)[0] / sum(np.histogram(random)[0]))
plt.hist(random, 30, density=True, color='red', alpha=0.1)
plt.show()


skewness = +8
random_right = skewnorm.rvs(a=skewness, loc=maxValue, size=numValues)
random_right = (random_right - min(random_right))
random_right = random_right / max(random_right) * maxValue
plt.hist(random_right, 30, density=True, color='red', alpha=0.1)
plt.show()


skewness = -8
random_left = skewnorm.rvs(a=skewness, loc=maxValue, size=numValues)
random_left = (random_left - min(random_left))
random_left = random_left / max(random_left) * maxValue
plt.hist(random_left, 30, density=True, color='red', alpha=0.1)
plt.show()
