import pandas
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from random import choice

import statsmodels.formula.api as smf
import statsmodels.api         as sm

def generate_fakedataframe(healthy_params,sick_params,num_toxicgenes,numdesigns=1000,pathwaylen=4,numgenes=20):
    healthy_mean, healthy_sd = healthy_params
    sick_mean, sick_sd       = sick_params
    
    pathwaylen = 4
    genenames  = ['gene%d'%x for x in range(numgenes)]
    geneidxs   = range(numgenes)

    fakedata = np.zeros(shape=(numdesigns,numgenes))

    for i in range(numdesigns):
        genes = set()
        for j in range(pathwaylen):
            geneidx = choice(geneidxs)
            genes.add(geneidx)
        for geneidx in genes:
            fakedata[i,geneidx] = 1.0
    
    toxic_genes = genenames[0:num_toxicgenes]
    toxic_idxs  = [genenames.index(x) for x in toxic_genes]
    growth_rates = np.zeros(numdesigns)

    sick_designs = np.zeros(numdesigns,dtype=bool)

    for i in range(numdesigns):
        sick_flag = False
        for toxic_idx in toxic_idxs:
            if fakedata[i,toxic_idx] != 0:
                sick_flag = True
        if sick_flag:
            growth_rates[i] = np.random.normal(size=1,loc=sick_mean,scale=sick_sd)[0]
            sick_designs[i] = True
        else:
            growth_rates[i] = np.random.normal(size=1,loc=healthy_mean,scale=healthy_sd)[0]

    df = pandas.DataFrame(fakedata,columns=genenames)
    df['growth_rate'] = pandas.Series(growth_rates, index=df.index)
    df['sick_design'] = pandas.Series(sick_designs, index=df.index) 
        
    return df,toxic_genes
    
def trial_stats(numtrials,healthy_params,sick_params,num_toxicgenes,numdesigns,pathwaylen,numgenes,verbose=False):
    accs = []
    ppvs = []
    for trial in range(numtrials):
        # Generate data
        df,toxic_genes = generate_fakedataframe(healthy_params,sick_params,num_toxicgenes,numdesigns=numdesigns,pathwaylen=4,numgenes=20)
        genes = [n for n in df.columns if n not in ['growth_rate', 'sick_design']]

        # Build Model
        model_txt = 'growth_rate ~ ' + ' + '.join(genes)
        res = smf.ols(formula=model_txt, data=df).fit()
        
        # Measure performance of model
        pred_toxic_genes = [x for x in res.pvalues.keys() if res.pvalues[x] < 0.005 and x != 'Intercept']

        TOT = len(genes)
        P = len(toxic_genes)
        N = TOT - P
        TP = float(len([x for x in pred_toxic_genes if x     in toxic_genes]))
        FP = float(len([x for x in pred_toxic_genes if x not in toxic_genes]))

        TN = float(len([x for x in genes if (x not in toxic_genes) and (x not in pred_toxic_genes)]))
        FN = float(len([x for x in genes if (x     in toxic_genes) and (x not in pred_toxic_genes)]))
        
        acc = (TP + TN) / (P + N)
        if (TP + FP) == 0: 
            ppv = 0
        else:
            ppv = TP / (TP + FP)
            
        if verbose:
            print res.summary2()
            print "TOXIC:", toxic_genes
            print "PRED: ", pred_toxic_genes
            print 'TP, FP, TN, FN, TOT = (%d, %d, %d, %d,   %d)'%(TP, FP, TN, FN, TOT)
            print 'acc, ppv = (%f , %f)'%(acc,ppv)
        accs.append(acc)
        ppvs.append(ppv)
    return [np.mean(accs), np.mean(ppvs)]

def check_numgenes():
    #Check how many TOTAL genes are tested in the experiment.
    numtrials = 50
    stats = []
    for property in range(20,200,20):
        acc,ppv = trial_stats(numtrials,(0.9,0.3),(0.6,0.4),4,500,4,property)
        stats.append([property,acc,ppv])
    statsm = np.array(stats)
    plt.plot(statsm.T[0],statsm.T[1],label='Accuracy')
    plt.plot(statsm.T[0],statsm.T[2],label="PPV")
    plt.xlabel("Num Total Genes)")
    plt.ylabel("Stat measure")
    plt.ylim([0.0,1.0])
    plt.legend(loc='center right')
    plt.show()   
      
    
    print statsm.T[0],statsm.T[1],statsm.T[2]


check_numgenes() 