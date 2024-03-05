#------------------------------------------------
import Optimization

#------------------------------------------------
dim             = 20
stepsize        = 2.5
Generation      = 10000
PopNumber       = 20*dim
ParentNumber    = dim+1
ChildrenNumber  = 4*dim
Functype        = 'Rastrigin'
minval          = -5.12
maxval          = 5.12

#------------------------------------------------
mymodel = Optimization.GeneticAlgorithm(dim, stepsize, Generation, PopNumber, ParentNumber, ChildrenNumber, Functype, minval, maxval)
mymodel.main()
print('Done')