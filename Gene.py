# -*- coding: utf-8 -*- 
import numpy as np
import random
np.random.seed(1)
random.seed(1)

class Gene():
#------------------------------------------------
    def __init__(self, dim, minval, maxval):
        self.Dim        = dim                                    # Dimension
        self.Fitness    = 0.                                     # Fitness value
        self.Gene       = np.random.uniform(minval, maxval, dim)    # Gene
        self.minval     = minval
        self.maxval     = maxval
    
    #--------------------------------------------
    def gene(self):
        return self.Gene

    def geneAllZero(self):
        self.Gene = np.zeros(self.Dim)

    def set_gene(self, g):
        self.Gene = g
    
    #--------------------------------------------
    def fitness(self):
        return self.Fitness
    
    def fitAllzero(self):
        self.Fitness = 0.0

    def set_fitness(self, f):
        self.Fitness = f