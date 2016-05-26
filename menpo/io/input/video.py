import numpy as np
import subprocess as sp
import re

from menpo.image.base import normalize_pixels_range, channels_to_front
from menpo.image import Image
from menpo.base import LazyList


def ffmpeg_importer(filepath, normalise=True, **kwargs):
    r"""
    Imports videos using the FFMPEG plugin from the imageio library. Returns a
    :map:`LazyList` that gives lazy access to the video on a per-frame basis.

    Parameters
    ----------
    filepath : `Path`
        Absolute filepath of the video.
    normalise : `bool`, optional
        If ``True``, normalise between 0.0 and 1.0 and convert to float. If
        ``False`` just return whatever imageio imports.
    \**kwargs : `dict`, optional
        Any other keyword arguments.

    Returns
    -------
    image : :map:`LazyList`
        A :map:`LazyList` containing :map:`Image` or subclasses per frame
        of the video.
    """
    reader = FFMpegVideoReader(filepath, normalise=normalise)
    ll = LazyList.init_from_index_callable(reader.__getitem__, len(reader))
    ll.fps = reader.fps

    return ll


def ffmpeg_types():
    r"""The supported FFMPEG types.

    Returns
    -------
    supported_types : `dict`
        A dictionary of extensions to the :map:`imageio_ffmpeg_importer`.
    """
    # Need to check this
    ffmpeg_exts = ['.avi', '.mp4']
    return {ext: ffmpeg_importer for ext in ffmpeg_exts}


class FFMpegVideoReader:
    """ Uses ffmpeg to read a video
    """
    def __init__(self, video_path, normalise=False):
        """Read a video as a list using ffmpeg

        Parameters
        ----------
        video_path: str
            absolute path to the video
        normalise: bool, default is False
            if True, the resulting range of the pixels of the returned frames is normalised
        """
        self.video_filename = str(video_path)
        self.normalise = normalise
        try:
            infos = video_infos_ffprobe(self.video_filename)
        except:
            infos = video_infos_ffmpeg(self.video_filename)
        self.duration = infos['duration']
        self.width = infos['width']
        self.height = infos['height']
        self.n_frames = infos['n_frames']
        self.fps = infos['fps']
        self._pipe = None
        self.index = -1  # contains the index of the last read frame
        # the index is updated in _open_pip, _read_one_frame and _trash_frames

    def __del__(self):
        """Close the pipe if open...
        """
        if self._pipe is not None:
            self._pipe.stdout.close()

    def __len__(self):
        return self.n_frames

    def _open_pipe(self, frame=None):
        """Open a pipe at the time just before the specified frame

        Parameters
        ----------
        frame: {int, None}, default is None
            if None, pipe opened from the beginning of the video
            otherwise, pipe opened at the time corresponding to that frame

        Note
        ----
        Since v.2.1 of ffmpeg, this is frame-accurate
        """
        if frame is not None and frame > 0:
            time = str(frame / self.fps)
            command = ['ffmpeg',
                       '-ss', time,
                       '-i', self.video_filename,
                       '-f', 'image2pipe',
                       '-pix_fmt', 'rgb24',
                       '-vcodec', 'rawvideo', '-']
        else:
            command = ['ffmpeg',
                       '-i', self.video_filename,
                       '-f', 'image2pipe',
                       '-pix_fmt', 'rgb24',
                       '-vcodec', 'rawvideo', '-']
            frame = 0

        if self._pipe is not None:
            self._pipe.stdout.close()
        self._pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)
        self.index = frame - 1  # We have not yet read the specified frame

    def __iter__(self):
        """Iterate through all frames of the video in order

        Only opens the pipe once at the beginning
        """
        self._open_pipe(frame=None)
        for self.index in range(self.n_frames):
            yield self._read_one_frame()

    def __getitem__(self, index):
        """ Get a specific frame from the video

        """
        # If the user is reading consecutive frames, or a frame later in the video, do not reopen a pipe
        if (self._pipe is None) or (index <= self.index):
            self._open_pipe(index)
        else:
            to_trash = index - self.index - 1
            if to_trash != 0:
                self._trash_frames(to_trash)

        return self._read_one_frame()

    def _trash_frames(self, n_frames):
        """Reads and trashs the data corresponding to `n_frames`
        """
        _ = self._pipe.stdout.read(self.height*self.width*3*n_frames)
        self._pipe.stdout.flush()
        self.index += n_frames

    def _read_one_frame(self):
        """Reads one frame from the opened self._pipe and converts it to numpy

        Returns
        -------
        Menpo image of shape (self.height, self.width, 3)
        """
        raw_data = self._pipe.stdout.read(self.height*self.width*3)
        frame = np.fromstring(raw_data, dtype=np.uint8)
        frame = frame.reshape((self.height, self.width, 3))
        frame = channels_to_front(frame)
        self._pipe.stdout.flush()
        self.index += 1

        if self.normalise:
            return Image(normalize_pixels_range(frame), copy=False)
        else:
            return Image(frame, copy=False)


def video_infos_ffmpeg(video_filename):
    """ Parses the information from a video using ffmpeg
    Uses subprocesses to get the information through a pipe

    Parameters
    ----------
    video_filename: str
        absolute path to the video file which information to extract

    Returns
    -------
    infos: dict
        keys are width, height (size of the frames)
        duration (duration of the video in seconds)
        n_frames
    """
    # Read information using ffmpeg
    command = ['ffmpeg', '-i', video_filename, '-']
    pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
    pipe.stdout.readlines()
    raw_infos = pipe.stderr.read().decode()
    pipe.terminate()

    # parse the information
    # Note: we use '\d+\.?\d*' so we can match both int and float for the fps
    video_info = re.search(r"Video:.*(?P<width> \d+)x(?P<height>\d+).*(?P<fps> \d+\.?\d*) fps",
                           raw_infos, re.DOTALL).groupdict()
    time = re.search(r"Duration:\s(?P<hours>\d+?):(?P<minutes>\d+?):(?P<seconds>\d+\.\d+?),",
                     raw_infos, re.DOTALL).groupdict()

    # Get the duration in seconds and convert size to ints
    hours = float(time['hours'])
    minutes = float(time['minutes'])
    seconds = float(time['seconds'])
    duration = 60*60*hours + 60*minutes + seconds

    fps = round(float(video_info['fps']))
    n_frames = round(duration*fps)
    width = int(video_info['width'])
    height = int(video_info['height'])

    # Create the resulting dictionary
    infos = {'duration': duration, 'width': width, 'height': height, 'n_frames': n_frames, 'fps': fps}

    return infos


def video_infos_ffprobe(video_filename):
    """ Parses the information from a video using ffprobe
    Uses subprocesses to get the information through a pipe

    Parameters
    ----------
    video_filename: str
        absolute path to the video file which information to extract

    Returns
    -------
    infos: dict
        keys are width, height (size of the frames)
        duration (duration of the video in seconds)
        n_frames
    """
    try:
        # Seems cleaner to me but adds an unnecessary json dependency
        import json
        p = sp.Popen(
                ['ffprobe', '-show_format', '-show_streams', '-print_format', 'json', video_filename],
                stdin=sp.PIPE,
                stdout=sp.PIPE,
                stderr=sp.PIPE,
            )
        # We are reading the data (printed in json format) from the pipe,
        data = json.loads(''.join(l.decode() for l in p.stdout.readlines()))
        # For some reason the information is in a list?
        data['streams'] = data['streams'][0]

    except ImportError:
        p = sp.Popen(
            ['ffprobe', '-show_format', '-show_streams', video_filename],
            stdin=sp.PIPE,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
        )
        # Store all the information in a dictionary
        data = dict(streams=[])
        stream = {}
        is_stream = is_format = False
        for line in p.stdout.readlines():
            # result.append(line)
            line = line.decode().strip()
            if line == '[STREAM]':
                is_stream = True
                continue
            if line == '[/STREAM]':
                is_stream = False
                data['streams'] = stream
                stream = {}
                continue
            if line == '[FORMAT]':
                is_format = True
                continue
            if line == '[/FORMAT]':
                break
            tokens = line.split('=')
            if is_stream:
                stream[tokens[0]] = tokens[1]
            if is_format:
                data[tokens[0]] = tokens[1]

    # Keep only the relevant parts
    width = int(data['streams']['width'])
    height = int(data['streams']['height'])
    n_frames = int(data['streams']['nb_frames'])
    duration = float(data['streams']['duration'])
    fps = float(eval(data['streams']['avg_frame_rate']))

    infos = {'duration': duration, 'width': width, 'height': height, 'n_frames': n_frames, 'fps': fps}

    return infos