import warnings
import os
import numpy as np
import subprocess as sp
import re
from pathlib import Path

from menpo.image.base import normalize_pixels_range, channels_to_front
from menpo.image import Image
from menpo.base import LazyList

from ..utils import DEVNULL, _call_subprocess


_FFMPEG_CMD = lambda: str(Path(os.environ.get('MENPO_FFMPEG_CMD', 'ffmpeg')))
_FFPROBE_CMD = lambda: str(Path(os.environ.get('MENPO_FFPROBE_CMD', 'ffprobe')))


def ffmpeg_importer(filepath, normalize=True, exact_frame_count=True, **kwargs):
    r"""
    Imports videos by streaming frames from a pipe using FFMPEG. Returns a
    :map:`LazyList` that gives lazy access to the video on a per-frame basis.

    There are two important environment variables that can be set to alter
    the behaviour of this function:

        ================== ======================================
        ENV Variable       Definition
        ================== ======================================
        MENPO_FFMPEG_CMD   The path to the 'ffmpeg' executable.
        MENPO_FFPROBE_CMD  The path to the 'ffprobe' executable.
        ================== ======================================

    Parameters
    ----------
    filepath : `Path`
        Absolute filepath of the video.
    normalize : `bool`, optional
        If ``True``, normalize between 0.0 and 1.0 and convert to float. If
        ``False`` just return whatever ffmpeg imports.
    exact_frame_count: `bool`, optional
        If ``True``, the import fails if ffprobe is not available
        (reading from ffmpeg's output returns inexact frame count)
    \**kwargs : `dict`, optional
        Any other keyword arguments.

    Returns
    -------
    image : :map:`LazyList`
        A :map:`LazyList` containing :map:`Image` or subclasses per frame
        of the video.
    """
    reader = FFMpegVideoReader(filepath, normalize=normalize, exact_frame_count=exact_frame_count)
    ll = LazyList.init_from_index_callable(lambda x: Image.init_from_channels_at_back(reader[x]), len(reader))
    ll.fps = reader.fps

    return ll


def ffmpeg_types():
    r"""The supported FFMPEG types.

    Returns
    -------
    supported_types : `dict`
        A dictionary of extensions supported by the FFMPEG importer.
    """
    # Need to check this
    ffmpeg_exts = ['.avi', '.mp4', '.mpg', '.mpeg', '.wmv', '.mov', '.mkv']
    return {ext: ffmpeg_importer for ext in ffmpeg_exts}


class FFMpegVideoReader(object):
    """
    Read a video using ffmpeg and handle state to allow seeking.

    Parameters
    ----------
    filepath : `Path`
        Absolute path to the video
    normalize : `bool`, optional
        If ``True``, the resulting range of the pixels of the returned
        frames is normalized.
    exact_frame_count : `bool`, optional
        If True, the import fails if ffmprobe is not available
        (reading from ffmpeg's output returns inexact frame count)
    """
    def __init__(self, filepath, normalize=False, exact_frame_count=True):
        self.filepath = filepath
        self.normalize = normalize
        self.exact_frame_count = exact_frame_count
        self._pipe = None
        if self.exact_frame_count:
            try:
                infos = video_infos_ffprobe(self.filepath)
            except:
                raise ValueError('ffprobe not available, unable to get exact '
                                 'frame count. If you want to use an '
                                 'approximate frame number, set exact_frame_'
                                 'count to False and proceed at your own '
                                 'risk.')
        else:
            infos = video_infos_ffmpeg(self.filepath)
        self.duration = infos['duration']
        self.width = infos['width']
        self.height = infos['height']
        self.n_frames = infos['n_frames']
        self.fps = infos['fps']
        # contains the index of the last read frame
        # the index is updated in _open_pipe, _read_one_frame and _trash_frames
        self.index = -1

    def _shutdown_pipe(self):
        if self._pipe is not None:
            if self._pipe.stdout:
                self._pipe.stdout.close()
            if self._pipe.stderr:
                self._pipe.stderr.close()
            if self._pipe.stdin:
                self._pipe.stdin.close()
        self._pipe = None

    def __del__(self):
        r"""
        Close the pipe if open.
        """
        self._shutdown_pipe()

    def __len__(self):
        return self.n_frames

    def _open_pipe(self, frame=None):
        r"""
        Open a pipe at the time just before the specified frame

        Parameters
        ----------
        frame : `int`, optional
            If ``None``, pipe opened from the beginning of the video
            otherwise, pipe opened at the time corresponding to that frame

        Note
        ----
        Since v.2.1 of ffmpeg, this is frame-accurate
        """
        if frame is not None and frame > 0:
            time = str(frame / float(self.fps))
            command = [_FFMPEG_CMD(),
                       '-ss', time,
                       '-i', str(self.filepath),
                       '-f', 'image2pipe',
                       '-pix_fmt', 'rgb24',
                       '-vcodec', 'rawvideo', '-']
        else:
            command = [_FFMPEG_CMD(),
                       '-i', str(self.filepath),
                       '-f', 'image2pipe',
                       '-pix_fmt', 'rgb24',
                       '-vcodec', 'rawvideo', '-']
            frame = 0

        self._shutdown_pipe()
        self._pipe = sp.Popen(command, stdout=sp.PIPE, stdin=DEVNULL,
                              stderr=DEVNULL,
                              bufsize=10**8)  # Is this buffer the correct size?
        # We have not yet read the specified frame
        self.index = frame - 1

    def __iter__(self):
        r"""
        Iterate through all frames of the video in order

        Only opens the pipe once at the beginning
        """
        self.index = 0
        for index in range(self.n_frames):
            yield self[index]

    def __getitem__(self, index):
        r"""
        Get a specific frame from the video
        """
        # If the user is reading consecutive frames, or a frame later in the
        # video, do not reopen a pipe
        if self._pipe is None or self._pipe.poll() is not None or index <= self.index:
            self._open_pipe(frame=index)
        else:
            to_trash = index - self.index - 1
            if to_trash > 0:
                self._trash_frames(to_trash)

        return self._read_one_frame()

    def _trash_frames(self, n_frames):
        r"""
        Reads and trashes the data corresponding to ``n_frames``
        """
        _ = self._pipe.stdout.read(self.height*self.width*3*n_frames)
        self._pipe.stdout.flush()
        self.index += n_frames

    def _read_one_frame(self):
        r"""
        Reads one frame from the opened ``self._pipe`` and converts it to
        a numpy array

        Returns
        -------
        image : :map:`Image`
            Image of shape ``(self.height, self.width, 3)``
        """
        raw_data = self._pipe.stdout.read(self.height*self.width*3)
        frame = np.frombuffer(raw_data, dtype=np.uint8)
        frame = frame.reshape((self.height, self.width, 3))
        self._pipe.stdout.flush()
        self.index += 1

        if self.normalize:
            frame = normalize_pixels_range(frame)

        return frame


def video_infos_ffmpeg(filepath):
    r"""
    Parses the information from a video using ffmpeg.
    Uses subprocess to get the information through a pipe.

    Parameters
    ----------
    filepath : `Path`
        absolute path to the video file which information to extract

    Returns
    -------
    infos : `dict`
        keys are width, height (size of the frames)
        duration (duration of the video in seconds)
        n_frames
    """
    warnings.warn('Estimating number of frames using ffmpeg duration which '
                  'may be inaccurate for certain types of encodings. Try '
                  'setting the MENPO_FFPROBE_CMD environment variable to '
                  'define the path to ffprobe.')

    # Read information using ffmpeg - the call below intentionally causes
    # an error about no output from FFMPEG in order to terminate faster - hence
    # reading the output from stderr.
    command = [_FFMPEG_CMD(), '-i', str(filepath), '-']
    with _call_subprocess(sp.Popen(command, stdout=DEVNULL,
                                   stderr=sp.PIPE)) as pipe:
        raw_infos = pipe.stderr.read().decode()

    # Note: we use '\d+\.?\d*' so we can match both int and float for the fps
    video_info = re.search(
        r"Video:.*(?P<width> \d+)x(?P<height>\d+).*(?P<fps> \d+\.?\d*) fps",
        raw_infos, re.DOTALL).groupdict()

    # Some videos don't have a valid duration
    time = re.search(
        r"Duration:\s(?P<hours>\d+?):(?P<minutes>\d+?):(?P<seconds>\d+\.\d+?),",
        raw_infos, re.DOTALL)
    if time is None:
        raise ValueError('Unable to determine duration for video - please '
                         'install and use ffprobe for accurate frame count '
                         'computation.')

    # Get the duration in seconds and convert size to ints
    time = time.groupdict()
    hours = float(time['hours'])
    minutes = float(time['minutes'])
    seconds = float(time['seconds'])
    duration = 60*60*hours + 60*minutes + seconds

    fps = round(float(video_info['fps']))
    n_frames = round(duration*fps)
    width = int(video_info['width'])
    height = int(video_info['height'])

    # Create the resulting dictionary
    infos = {'duration': duration, 'width': width, 'height': height,
             'n_frames': n_frames, 'fps': fps}

    return infos


def video_infos_ffprobe(filepath):
    """
    Parses the information from a video using ffprobe
    Uses subprocess to get the information through a pipe

    Parameters
    ----------
    filepath : `Path`
        Absolute path to the video file which information to extract

    Returns
    -------
    infos : `dict`
        keys are width, height (size of the frames)
        duration (duration of the video in seconds)
        n_frames
    """
    expected_keys = {'width', 'height', 'avg_frame_rate', 'duration',
                     'nb_read_frames'}

    p = sp.Popen(
        [_FFPROBE_CMD(), '-v', 'quiet',        # Quiet output
         '-count_frames',                      # Count the number of complete frames
         '-select_streams', 'v:0',             # Only show the first stream
         '-show_entries',                      # Show the entries below
         'stream=height,width,nb_read_frames,duration,avg_frame_rate',
         '-of', 'default=noprint_wrappers=1',  # Output format is just to print key=value pairs
         str(filepath)],
        stdin=sp.PIPE,
        stdout=sp.PIPE,
        stderr=sp.PIPE,
    )
    with _call_subprocess(p) as pipe:
        stdout_output = pipe.stdout.readlines()
    del p

    # Split the key=value pairs on the '='
    kv_dict = dict([v.decode().strip().split('=') for v in stdout_output])
    found_keys = set(kv_dict.keys())

    if found_keys ^ expected_keys:
        raise ValueError('Not all of the expected values were returned. '
                         'Expected {} but found {}.'.format(
            expected_keys, found_keys))

    kv_dict['n_frames'] = int(kv_dict.pop('nb_read_frames'))
    kv_dict['width'] = int(kv_dict['width'])
    kv_dict['height'] = int(kv_dict['height'])
    # Some videos may not have a valid duration
    if kv_dict['duration'] == 'N/A':
        kv_dict['duration'] = None
    else:
        kv_dict['duration'] = float(kv_dict['duration'])
    fps = kv_dict.pop('avg_frame_rate').split('/')
    try:
        kv_dict['fps'] = float(fps[0]) / float(fps[1])
    except ZeroDivisionError:
        kv_dict['fps'] = None

    return kv_dict
