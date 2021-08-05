import sklearn
from sklearn import metrics
import numpy as np


# UCF101, softmax
known_as_known = np.array([1026, 1026, 1023, 1022, 1020, 1016, 1003])
known_as_unknown = np.array([0, 0, 3, 4, 6, 10, 23])

unknown_as_known = np.array([3142, 2098, 1719, 1461, 1227, 995, 634])
unknown_as_unknown = np.array([140, 1184, 1563, 1821, 2055, 2287, 2648])

fpr = (unknown_as_known)/(known_as_known+unknown_as_known)
tpr = (known_as_known)/(known_as_known+unknown_as_known)

np.insert(fpr, 0, 0.0)
np.insert(fpr, fpr.shape[0], 1.0)

np.insert(tpr, 0, 0.0)
np.insert(tpr, fpr.shape[0], 1.0)

print(fpr)
print(tpr)

print(metrics.auc(fpr, tpr))