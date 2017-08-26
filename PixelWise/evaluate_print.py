import numpy as np
import StatusSaver as log
log = log.StatusSaver

def print_cscore(epoch,pred,actual):
    pred = get_label(pred)
    actual = get_label(actual)
    n = len(actual)
    n_class = len(np.unique(actual))
    cm = np.zeros([n_class,n_class])
    for i in range(n):
        cm[actual[i],pred[i]] += 1
    # avg_cm = old_cm + cm
    # avg_cm /= eval_epoch
    pr = get_pr(cm)
    f1 = get_f1(pr)

    log("status.txt",["Results for epoch: %d"%(epoch),"CM: ",cm,"PR: ",pr, "F1:",f1, "Avg F1: %.2f\n"%(np.mean(f1,0))])
    # print("F1:\n",np.mean(f1),"\n")

def get_label(mat):
    label = np.argmax(mat,1)
    return label

def get_pr(cm):
    n = cm.shape[0]
    pr = np.zeros([n,2])
    for i in range(n):
        if cm[i,i] == 0:
            pr[i,0] = 0
            pr[i,1] = 1
        else:
            pr[i,0] = ( cm[i,i] / np.sum(cm[:,i]) + 1e-5 ) * 100
            pr[i,1] = ( cm[i,i] / np.sum(cm[i,:]) + 1e-5 ) * 100
    return pr

def get_f1(pr):
    n = pr.shape[0]
    fscore = np.zeros([n,1])
    for i in range(n):
        p = pr[i,0]
        r = pr[i,1]
        if (p+r) == 0:
            fscore[i,0] = 0
        else:
            fscore[i,0] = (2*p*r)/(p+r)
    return fscore