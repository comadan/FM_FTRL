# time pypy-2.4 -u runmodel.py | tee output_0.txt
from FM_FTRL_machine import *
import random
from math import log

#### RANDOM SEED ####
random.seed(5)  # seed random variable for reproducibility
#####################

####################
#### PARAMETERS ####
####################
reportFrequency = 1000000
trainingFile = "../data/train.csv"

fm_dim = 4
fm_initDev = .01
hashSalt = "salty"
    
alpha = .1
beta = 1.

alpha_fm = .05
beta_fm = 1.

p_D = 22
D = 2 ** p_D

L1 = 1.0
L2 = .1
L1_fm = 2.0
L2_fm = 1.0

dropoutRate = .8

n_epochs = 5


####
start = datetime.now()

# initialize a FM learner
learner = FM_FTRL_machine(fm_dim, fm_initDev, L1, L2, L1_fm, L2_fm, D, alpha, beta, alpha_fm = alpha_fm, beta_fm = beta_fm, dropoutRate = dropoutRate)

print("Start Training:")
for e in range(n_epochs):
    
    # if it is the first epoch, then don't use L1_fm or L2_fm
    if e == 0:
        learner.L1_fm = 0.
        learner.L2_fm = 0.
    else:
        learner.L1_fm = L1_fm
        learner.L2_fm = L2_fm
    
    cvLoss = 0.
    cvCount = 0.
    progressiveLoss = 0.
    progressiveCount = 0.
    for t, date, ID, x, y in data(trainingFile, D, hashSalt):
        if date == 30:
            p = learner.predictWithDroppedOutModel(x)
            loss = logLoss(p, y)
            cvLoss += loss
            cvCount += 1.
        else:
            p = learner.dropoutThenPredict(x)
            loss = logLoss(p, y)
            learner.update(x, p, y)
            
            progressiveLoss += loss
            progressiveCount += 1.
            if t % reportFrequency == 0:                
                print("Epoch %d\tcount: %d\tProgressive Loss: %f" % (e, t, progressiveLoss / progressiveCount))
        
    print("Epoch %d finished.\tvalidation loss: %f\telapsed time: %s" % (e, cvLoss / cvCount, str(datetime.now() - start)))
    
# save the weights
w_outfile = "param.w.txt"
w_fm_outfile = "param.w_fm.txt"
learner.write_w(w_outfile)
learner.write_w_fm(w_fm_outfile)
        



