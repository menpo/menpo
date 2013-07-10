import os.path


def pybug_src_dir_path():
    return os.path.split(os.path.abspath(__file__))[0][:-5]


def data_dir_path():
    return os.path.join(pybug_src_dir_path(), 'data')


def data_path_to(data):
    return os.path.join(data_dir_path(), data)
