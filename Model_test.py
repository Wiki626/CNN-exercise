import classifier_function as cf
import os
import re

output = cf.image_classifier((os.path.join(os.getcwd(), 'test_set')), (os.getcwd() + '\CNN_model.h5'))

def confusion(input):
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    bad_results = 0
    for k in input:
        if re.search('.+not_tank.+', k):
            if str(input.get(k)) == '[[0.]]':
                true_neg += 1
            elif str(input.get(k)) == '[[1.]]':
                false_neg += 1
            else:
                bad_results += 1
        else:
            if str(input.get(k)) == '[[0.]]':
                false_pos += 1
            elif str(input.get(k)) == '[[1.]]':
                true_pos += 1
            else:
                bad_results += 1
    print('Model returned %s true positives %s false positives %s true negatives, %s false negatives, and %s bad '
          'results outside of the predicted range.' %(true_pos, false_pos, true_neg, false_neg, bad_results))
    return(true_pos, false_pos, false_neg, true_neg)

def interpret_results(tp, fp, fn, tn):
    pyes = ((tp + fn) / (tp + fp + fn + tn)) * ((tp + fp) / (tp + fp + fn + tn))
    pno = ((fp + tn) / (tp + fp + fn + tn)) * ((tn + fn) / (tp + fp + fn + tn))
    pe = pyes + pno
    p0 = (tp + tn) / (tp + fp + fn + tn)
    k = (p0 - pe) / (1 - pe)
    if k < 0:
        print("Model has a Kappa score of {}. No agreement.".format(k))
    elif k < 0.2:
        print("Model has a Kappa score of {}. Fair agreement.".format(k))
    elif k < 0.4:
        print("Model has a Kappa score of {}. Slight agreement.".format(k))
    elif k < 0.6:
        print("Model has a Kappa score of {}. Moderate agreement.".format(k))
    elif k < 0.8:
        print("Model has a Kappa score of {}. Substantial agreement.".format(k))
    else:
        print("Model has a Kappa score of {}. Near-perfect agreement.".format(k))

tp, fp, fn, tn = confusion(output)
print(interpret_results(tp, fp, fn, tn))