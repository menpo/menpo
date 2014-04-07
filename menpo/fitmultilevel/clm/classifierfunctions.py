from sklearn import svm
from sklearn import linear_model


#TODO: Document me
def classifier(X, t, classifier_type, **kwargs):
    r"""
    """
    if hasattr(classifier_type, '__call__'):
        classifier_closure = classifier_type(X, t, **kwargs)

        # note that, self is required as this closure is assigned to an object
        def classifier_object_method(self, x):
            return classifier_closure(x)

        return classifier_closure
    else:
        raise ValueError("classifier_type can only be: a closure defining "
                         "a particular classifier technique. Several "
                         "examples of such closures can be found in "
                         "`menpo.fitmultilevel.clm.classifierfunctions` "
                         "(linear_svm, ...).")


#TODO: Document me
def linear_svm(X, t):
    r"""
    Linear Support Vector Machine classifier
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
