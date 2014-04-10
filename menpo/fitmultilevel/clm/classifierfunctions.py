from sklearn import svm
from sklearn import linear_model


def classifier(X, t, classifier_type, **kwargs):
    r"""
    General binary classifier function. Provides a consistent signature for
    specific implementation of binary classifier functions.

    Parameters
    ----------
    X: (n_samples, n_features) ndarray
        Training vectors.

    t: (n_samples, 1) ndarray
        Binary class labels.

    classifier_type: closure
        Closure implementing a particular type of binary classifier.

    Returns
    -------
    classifier_closure: function
        The classifier.
    """
    if hasattr(classifier_type, '__call__'):
        classifier_closure = classifier_type(X, t, **kwargs)
        return classifier_closure
    else:
        raise ValueError("classifier_type can only be a closure defining "
                         "a particular classifier technique. Several "
                         "examples of such closures can be found in "
                         "`menpo.fitmultilevel.clm.classifierfunctions` "
                         "(linear_svm_lr, ...).")


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
