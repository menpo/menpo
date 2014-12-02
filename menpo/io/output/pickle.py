import cPickle as pickle


def pickle_export(file_handle, obj):
    pickle.dump(obj, file_handle, protocol=2)
