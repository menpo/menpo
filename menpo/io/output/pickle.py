import cPickle as pickle


def pickle_export(obj, file_handle):
    pickle.dump(obj, file_handle, protocol=2)
