try:
    import cPickle as pickle
except ImportError:
    import pickle
import gzip
from .base import Importer


class PickleImporter(Importer):

    def build(self):
        with open(self.filepath, 'rb') as f:
            x = pickle.load(f)
        return x


class GZipPickleImporter(Importer):

    def build(self):
        with gzip.open(self.filepath, 'rb') as f:
            x = pickle.load(f)
        return x
