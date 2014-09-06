from sklearn import svm
from sklearn import linear_model


class linear_svm_lr(object):
    r"""
    Binary classifier that combines Linear Support Vector Machines and
    Logistic Regression.
    """
    def __init__(self, X, t):
        self.clf1 = svm.LinearSVC(class_weight='auto')
        self.clf1.fit(X, t)
        t1 = self.clf1.decision_function(X)
        self.clf2 = linear_model.LogisticRegression(class_weight='auto')
        self.clf2.fit(t1[..., None], t)

    def __call__(self, x):
        t1_pred = self.clf1.decision_function(x)
        return self.clf2.predict_proba(t1_pred[..., None])[:, 1]
