from sklearn.metrics import confusion_matrix

def precision(true_labels, predictions):
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel() 
    return tp/(tp + fp)

def recall(true_labels, predictions):
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel() 
    return tp/(tp + fn)

def f1_score(true_labels, predictions) :
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel() 
    prec = tp/(tp + fp)
    reca = tp/(tp + fn)
    
    return 2*(prec*reca)/(prec + reca)
