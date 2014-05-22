import urllib2
import cStringIO
import os
import scipy.io as sio
import glob
import tempfile
import shutil
import zipfile
from collections import namedtuple

# Container for bounding box
from menpo.shape import PointCloud

BoundingBox = namedtuple('BoundingBox', ['detector', 'groundtruth'])
# Where the bounding boxes should be fetched from
bboxes_url = 'http://ibug.doc.ic.ac.uk/media/uploads/competitions/bounding_boxes.zip'


def download_ibug_bounding_boxes(path=None, verbose=False):
    r"""Downloads the bounding box information provided on the iBUG website
    and unzips it to the path.
    
    Parameters
    ----------
    path : `str`, optional
        The path that the bounding box files should be extracted to.
        If None, the current directory will be used.
    """
    if path is None:
        path = os.getcwd()
    else:
        path = os.path.abspath(os.path.expanduser(path))
    if verbose:
        print('Acquiring bounding box information from iBUG website...')
    try:
        remotezip = urllib2.urlopen(bboxes_url)
        zipinmemory = cStringIO.StringIO(remotezip.read())
        ziplocal = zipfile.ZipFile(zipinmemory)
    except Exception as e:
        print('Unable to grab bounding boxes (are you online?)')
        raise e
    if verbose:
        print('Extracting to {}'.format(os.path.join(path, 'Bounding Boxes')))
    try:
        ziplocal.extractall(path=path)
        if verbose:
            print('Done.')
    except Exception as e:
        if verbose:
            print('Unable to save.'.format(e))
        raise e


def import_bounding_boxes(boxes_path):
    r"""
    Imports the bounding boxes at boxes_path, returning a dict
    where the key is a filename and the value is a BoundingBox.
    
    Parameters
    ----------
    boxes_path : str
        A path to a bounding box .mat file downloaded from the
        iBUG website.
        
    Returns
    -------
    dict:
        Mapping of filenames to bounding boxes

    """
    bboxes_mat = sio.loadmat(boxes_path)
    bboxes = {}
    for bb in bboxes_mat['bounding_boxes'][0, :]:
        fname, detector_bb, gt_bb = bb[0, 0]
        bboxes[str(fname[0])] = BoundingBox(
            PointCloud(detector_bb.reshape([2, 2])[:, ::-1]),
            PointCloud(gt_bb.reshape([2, 2])[:, ::-1]))
    return bboxes


def import_all_bounding_boxes(boxes_dir_path=None, verbose=True):
    r"""
    Imports all the bounding boxes contained in boxes_dir_path.
    If the path is False, the bounding boxes are downloaded from the
    iBUG website directly.
    
    
    """
    temp_path = None
    if boxes_dir_path is None:
        print('No path provided - acuqiring zip to tmp dir...')
        temp_path = tempfile.mkdtemp()
        download_ibug_bounding_boxes(path=temp_path, verbose=verbose)
        boxes_dir_path = os.path.join(temp_path, 'Bounding Boxes')
    prefix = 'bounding_boxes_'
    bbox_paths = glob.glob(os.path.join(boxes_dir_path, prefix + '*.mat'))
    bboxes = {}
    for bbox_path in bbox_paths:
        db = os.path.splitext(os.path.split(bbox_path)[-1])[0][len(prefix):]
        if verbose:
            print('Importing {}'.format(db))
        bboxes[db] = import_bounding_boxes(bbox_path)
    if verbose:
        print('Cleaning up...')
    if temp_path:
        # If we downloaded, clean it up!
        shutil.rmtree(temp_path)
    if verbose:
        print('Done.')
    return bboxes

