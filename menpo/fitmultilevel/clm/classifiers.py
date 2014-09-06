from sklearn import svm
from sklearn import linear_model


def linear_svm_lr(X, t):
    r"""
    Binary classifier that combines Linear Support Vector Machines and
    Logistic Regression.
    """
    clf1 = svm.LinearSVC(class_weight='auto')
    clf1.fit(X, t)
    t1 = clf1.decision_function(X)
    clf2 = linear_model.LogisticRegression(class_weight='auto')
    clf2.fit(t1[..., None], t)

    def linear_svm_predict(x):
        t1_pred = clf1.decision_function(x)
        return clf2.predict_proba(t1_pred[..., None])[:, 1]

    return linear_svm_predict
