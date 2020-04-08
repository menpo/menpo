import contextlib
from math import ceil
import os
from pathlib import Path

try:
    from subprocess import DEVNULL
except ImportError:
    DEVNULL = open(os.devnull, "wb")

try:
    from urllib2 import urlopen  # Py2
except ImportError:
    from urllib.request import urlopen  # Py3


def _norm_path(filepath):
    r"""
    Uses all the tricks in the book to expand a path out to an absolute one.
    """
    return Path(
        os.path.abspath(
            os.path.normpath(os.path.expandvars(os.path.expanduser(str(filepath))))
        )
    )


def _possible_extensions_from_filepath(filepath):
    r"""
    Generate a list of possible extensions from the given filepath. Since
    filenames can contain '.' characters and some extensions are compound
    (e.g. '.pkl.gz'), there may be many possible extensions for a given
    path. Generate a list possible extensions, preferring longer extensions.

    Parameters
    ----------
    filepath : `Path`
        A pathlib Path.

    Returns
    -------
    possible_extensions : `list` of `str`
        A list of extensions **with** leading '.' characters and converted
        to lowercase.
    """
    suffixes = filepath.suffixes
    return ["".join(suffixes[i:]).lower() for i in range(len(suffixes))]


def _normalize_extension(extension):
    r"""
    Simple function that takes a given extension string and ensures that it
    is lower case and contains the leading period e.g. ('.jpg')

    Parameters
    ----------
    extension : `str`
        The string extension.

    Returns
    -------
    norm_extension : `str`
        The normalized extension, lower case with '.' prefix.
    """
    if extension is None:
        return None
    elif extension[0] is not ".":
        # Account for the fact the user may only have passed the extension
        # without the proceeding period
        extension = "." + extension
    return extension.lower()


@contextlib.contextmanager
def _call_subprocess(process):
    r"""
    Call a subprocess and automatically clean up/wait for the various
    pipe interfaces, {stderr, stdout, stdin}.

    Parameters
    ----------
    process : `subprocess.POpen`
        The subprocess POpen object to automatically close.

    Yields
    ------
    The ``subprocess.POpen`` object back for processing.
    """
    try:
        yield process
    finally:
        for stream in (process.stdout, process.stdin, process.stderr):
            if stream:
                stream.close()
        process.wait()


def copy_and_yield(fsrc, fdst, length=1024*1024):
    """copy data from file-like object fsrc to file-like object fdst"""
    while 1:
        buf = fsrc.read(length)
        if not buf:
            break
        fdst.write(buf)
        yield


def download_file(url, destination, verbose=False):
    r"""
    Download a file from a URL to a path, optionally reporting the progress

    Parameters
    ----------
    url : `str`
        The URL of a remote resource that should be downloaded
    destination : `Path`
        The path on disk that the file will be downloaded to
    verbose : `bool`, optional
        If ``True``, report the progress of the download dynamically.
    """
    from menpo.visualize.textutils import print_progress, bytes_str
    req = urlopen(url)
    chunk_size_bytes = 512 * 1024

    with open(str(destination), 'wb') as fp:

        # Retrive a generator that we can keep yielding from to download the
        # file in chunks.
        copy_progress = copy_and_yield(req, fp, length=chunk_size_bytes)

        if verbose:
            # wrap the download object with print progress to log the status
            n_bytes = int(req.headers['content-length'])
            n_items = int(ceil((1.0 * n_bytes) / chunk_size_bytes))
            prefix = 'Downloading {}'.format(bytes_str(n_bytes))
            copy_progress = print_progress(copy_progress, n_items=n_items,
                                           show_count=False, prefix=prefix)

        for _ in copy_progress:
            pass

    req.close()
