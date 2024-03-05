# -*- coding: utf-8 -*- 
import numpy as np
import math
import random
import copy
import matplotlib.pyplot as plt
from Gene import Gene

np.set_printoptions(precision=8, floatmode='maxprec')
NUM_ELITE = 0
random.seed(1)
np.random.seed(1)

class GeneticAlgorithm():
#----------------------------------------------
    ##  1. Constructor
    def __init__(self, dim: int, stepsize: float, GenNumber: int, PopNumber: int, ParentNumber: int, ChildrenNumber: int, functype: str, minval: float, maxval: float):
        
        self.dim            = dim                                         # dimension
        self.stepsize       = stepsize                                    # stepsize for REXStar
        self.GenNumber      = GenNumber                                   # Number of generations
        self.PopNumber      = PopNumber                                   # Number of population
        self.ParentNumber   = ParentNumber                                # Number of parents
        self.ChildrenNumber = ChildrenNumber                              # Number of children
        self.functype       = functype                                    # Objective function
        
        self.minval     = minval   
        self.maxval     = maxval
        self.idx_tmp    = np.random.randint(0, PopNumber, ParentNumber)

        self.Gpar   = np.zeros(dim)
        self.Gbest  = np.zeros(dim)
        
        self.population     = ['pop']*PopNumber
        self.parent         = ['parent']*ParentNumber
        self.mirror         = ['reflection']*ParentNumber
        self.bestparents    = ['bestparent']*ParentNumber
        self.children       = ['children']*ChildrenNumber
        self.best           = [Gene(dim, minval, maxval)]

        for i in range(PopNumber):
            self.population[i] = Gene(dim, minval, maxval)
        
        for i in range(ParentNumber):
            self.parent[i]      = Gene(dim, minval, maxval)
            self.mirror[i]      = Gene(dim, minval, maxval)
            self.bestparents[i] = Gene(dim, minval, maxval)
        
        for i in range(ChildrenNumber):
            self.children[i] = Gene(dim, minval, maxval)

#------------------------------------------------
    ##  2. Objective Function
    def Cal_fitness(self, X, functype) -> float:
        fitness = 0.

        #Spher Function
        if(functype == 'Sphere'):
            for i in range(self.dim):
                fitness += (X[i] - 1.0)**2
        
        #Rosenbrock_star Function
        if(functype == 'Rosenbrock_star'):
            for i in range(1, self.dim):
                fitness += 100*(X[0] - X[i]**2)**2 + (1.0 - X[i])**2
        
        #Rosenbrock_chain Function
        if(functype == 'Rosenbrock_chain'):
            for i in range(self.dim-1):
                fitness += 100*(X[i+1] - X[i]**2)**2 + (1.0 - X[i])**2
        
        #Rastring Function
        if(functype == 'Rastrigin'):
            sumval = 0.
            for i in range(self.dim):
                sumval += (X[i] - 1.0)**2 - 10*math.cos(2*math.pi*(X[i] - 1.0))
            fitness = 10*self.dim + sumval

        #Bohachevsky Function
        if(functype == 'Bohachevsky'):
            for i in range(self.dim-1):
                fitness += X[i]**2 + 2*(X[i+1]**2) - 0.3*math.cos(3*math.pi*X[i]) - 0.4*math.cos(4*math.pi*X[i+1]) + 0.7
        
        #Ackley Function
        if(functype == 'Ackley'):
            sumval1 = 0.
            sumval2 = 0.
            for i in range(self.dim):
                sumval1 += X[i]**2
                sumval2 += math.cos(2*math.pi*X[i])
            fitness = 20 - 20*math.exp(-0.2*math.sqrt(1/self.dim*sumval1)) + math.e - math.exp(1/self.dim*sumval2)

        return fitness

#------------------------------------------------
    ##  3. Main
    def main(self) -> None:
        
        self.Eval(self.population)
        self.minsort(self.population)
        
        countGen    = []
        avefitness  = []
        bestfitness = []

        for GenLoop in range(self.GenNumber):
            
            self.Reproduction()
            self.Allzero(self.Gpar)
            self.Cal_grav(self.parent, self.Gpar)
            
            self.make_Mirror()
            self.Eval(self.mirror)
            self.BestParents()
            self.Allzero(self.Gbest)
            self.Cal_grav(self.bestparents, self.Gbest)
            
            self.REXstar()
            self.Eval(self.children)
            self.minsort(self.children)
            self.Replace()
            self.minsort(self.population)

            if(GenLoop == 0):
                self.best = copy.deepcopy(self.population[0])
            
            if(GenLoop > 0):
                if(Gene.fitness(self.population[0]) < Gene.fitness(self.best)):
                    self.best = copy.deepcopy(self.population[0])
            print(GenLoop, Gene.fitness(self.best))

            countGen.append(GenLoop)
            bestfitness.append(Gene.fitness(self.best))
            avefitness.append(self.average())
            if (math.sqrt((Gene.fitness(self.best))**2) <= 1e-7):
                print(GenLoop)
                print(Gene.gene(self.best))
                break
        
        plt.figure()
        plt.plot(countGen, bestfitness,   color='black', linestyle='solid',   label='best')
        plt.plot(countGen, avefitness,    color='black', linestyle='dotted',  label='average')
        plt.xlabel('Generation[-]')
        plt.ylabel('Fitness[-]')
        plt.legend()
        plt.savefig('FitnessCurve.png')
        plt.clf()

#------------------------------------------------
    ##  4. reproduction
    def Reproduction(self) -> None:
        for i in range(self.ParentNumber):
            if(i < NUM_ELITE):
                idx = i                                                 
            else:
                idx = np.random.randint(0, self.PopNumber)
                while True:
                    judge = True
                    for j in range(i):
                        if(idx == self.idx_tmp[j]):
                            idx     = np.random.randint(0, self.PopNumber)
                            judge   = False
                    if(judge == True):
                        break
            self.parent[i]  = copy.deepcopy(self.population[idx])
            self.idx_tmp[i] = idx
            
#------------------------------------------------
    ##  5. calculate gravity center
    def Cal_grav(self, individual, gpoint) -> None:
        for i in range(len(individual)):
            gpoint += Gene.gene(individual[i])
        gpoint /= len(individual)

#------------------------------------------------
    ##  6. make mirror individuals
    def make_Mirror(self) -> None:
        for i in range(self.ParentNumber):
            gene = 2.0 * self.Gpar - Gene.gene(self.parent[i])
            Gene.set_gene(self.mirror[i], gene)
        
#------------------------------------------------
    ##  7. evaluation
    def Eval(self, individual) -> None:
        for i in range(len(individual)):
            Gene.set_fitness(individual[i], self.Cal_fitness(Gene.gene(individual[i]), self.functype))

#------------------------------------------------
    ##  8. sampling the best individuals between parents and mirrors
    def BestParents(self) -> None:
        Parent_Mirror = self.parent + self.mirror
        self.minsort(Parent_Mirror)
        self.bestparents = copy.deepcopy(Parent_Mirror[:self.ParentNumber])

#------------------------------------------------
    ##  10. sort
    def minsort(self, Individual) -> None:
        fitness = np.zeros(len(Individual))
        for i in range(len(Individual)):
            fitness[i]  = Gene.fitness(Individual[i])
        sortind         = np.argsort(fitness)
        Individualtmp   = ['individual']*len(Individual)
        for i in range(len(Individual)):
            Individualtmp[i] = copy.deepcopy(Individual[sortind[i]])
        for i in range(len(Individual)):
            Individual[i] = copy.deepcopy(Individualtmp[i])

#------------------------------------------------
    ##  11. REXstar
    def REXstar(self) -> None:
        for i in range(self.ChildrenNumber):
            xi_t    = np.random.uniform(0.0, self.stepsize, self.dim)
            val_1   = xi_t * (self.Gbest - self.Gpar)
            val_2   = np.zeros(self.dim)
            for parent in self.parent:
                xi      = np.random.uniform(-1.0*math.sqrt(3.0/self.ParentNumber), math.sqrt(3.0/self.ParentNumber))
                val_2   += xi * (Gene.gene(parent) - self.Gpar)
            gene = self.Gpar + val_1 + val_2
            Gene.set_gene(self.children[i], gene)
    
#------------------------------------------------
    ##  12. replace population with children
    def Replace(self) -> None:
        for i in range(self.ParentNumber):
            self.population[self.idx_tmp[i]] = copy.deepcopy(self.children[i])
        
#------------------------------------------------
    def average(self):
        aveval = 0.
        for i in range(self.PopNumber):
            aveval += Gene.fitness(self.population[i]) / self.PopNumber
        return aveval

#------------------------------------------------
    def Allzero(self, X):
        for i in range(len(X)):
            X[i] = 0.
