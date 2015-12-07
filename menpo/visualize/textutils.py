from __future__ import division, print_function
from collections import deque
from datetime import datetime
import sys
from time import time


def progress_bar_str(percentage, bar_length=20, bar_marker='=', show_bar=True):
    r"""
    Returns an `str` of the specified progress percentage. The percentage is
    represented either in the form of a progress bar or in the form of a
    percentage number. It can be combined with the :func:`print_dynamic`
    function.

    Parameters
    ----------
    percentage : `float`
        The progress percentage to be printed. It must be in the range
        ``[0, 1]``.
    bar_length : `int`, optional
        Defines the length of the bar in characters.
    bar_marker : `str`, optional
        Defines the marker character that will be used to fill the bar.
    show_bar : `bool`, optional
        If ``True``, the `str` includes the bar followed by the percentage,
        e.g. ``'[=====     ] 50%'``

        If ``False``, the `str` includes only the percentage,
        e.g. ``'50%'``

    Returns
    -------
    progress_str : `str`
        The progress percentage string that can be printed.

    Raises
    ------
    ValueError
        ``percentage`` is not in the range ``[0, 1]``
    ValueError
        ``bar_length`` must be an integer >= ``1``
    ValueError
        ``bar_marker`` must be a string of length 1

    Examples
    --------
    This for loop: ::

        n_iters = 2000
        for k in range(n_iters):
            print_dynamic(progress_bar_str(float(k) / (n_iters-1)))

    prints a progress bar of the form: ::

        [=============       ] 68%
    """
    if percentage < 0:
        raise ValueError("percentage is not in the range [0, 1]")
    elif percentage > 1:
        percentage = 1
    if not isinstance(bar_length, int) or bar_length < 1:
        raise ValueError("bar_length must be an integer >= 1")
    if not isinstance(bar_marker, str) or len(bar_marker) != 1:
        raise ValueError("bar_marker must be a string of length 1")
    # generate output string
    if show_bar:
        str_param = "[%-" + str(bar_length) + "s] %d%%"
        bar_percentage = int(percentage * bar_length)
        return str_param % (bar_marker * bar_percentage, percentage * 100)
    else:
        return "%d%%" % (percentage * 100)


def print_dynamic(str_to_print):
    r"""
    Prints dynamically the provided `str`, i.e. the `str` is printed and then
    the buffer gets flushed.

    Parameters
    ----------
    str_to_print : `str`
        The string to print.
    """
    sys.stdout.write("\r{}".format(str_to_print.ljust(80)))
    sys.stdout.flush()


def bytes_str(num):
    r"""
    Converts bytes to a human readable format. For example: ::

        print_bytes(12345) returns '12.06 KB'
        print_bytes(123456789) returns '117.74 MB'

    Parameters
    ----------
    num : `int`
        The size in bytes.

    Raises
    ------
    ValueError
        num must be int >= 0
    """
    if not isinstance(num, int) or num < 0:
        raise ValueError("num must be int >= 0")
    for x in ['bytes', 'KB', 'MB', 'GB']:
        if num < 1024.0:
            return "{0:3.2f} {1:s}".format(num, x)
        num /= 1024.0
    return "{0:3.2f} {1:s}".format(num, 'TB')


def print_progress(iterable, prefix='', n_items=None, offset=0,
                   show_bar=True, show_count=True, show_eta=True,
                   end_with_newline=True):
    r"""
    Print the remaining time needed to compute over an iterable.

    To use, wrap an existing iterable with this function before processing in
    a for loop (see example).

    The estimate of the remaining time is based on a moving average of the last
    100 items completed in the loop.

    Parameters
    ----------
    iterable : `iterable`
        An iterable that will be processed. The iterable is passed through by
        this function, with the time taken for each complete iteration logged.
    prefix : `str`, optional
        If provided a string that will be prepended to the progress report at
        each level.
    n_items : `int`, optional
        Allows for ``iterator`` to be a generator whose length will be assumed
        to be `n_items`. If not provided, then ``iterator`` needs to be
        `Sizable`.
    offset : `int`, optional
        Useful in combination with ``n_items`` - report back the progress as
        if `offset` items have already been handled. ``n_items``  will be left
        unchanged.
    show_bar : `bool`, optional
        If False, The progress bar (e.g. [=========      ]) will be hidden.
    show_count : `bool`, optional
        If False, The item count (e.g. (4/25)) will be hidden.
    show_eta : `bool`, optional
        If False, The estimated time to finish (e.g. - 00:00:03 remaining)
        will be hidden.
    end_with_newline : `bool`, optional
        If False, there will be no new line added at the end of the dynamic
        printing. This means the next print statement will overwrite the
        dynamic report presented here. Useful if you want to follow up a
        print_progress with a second print_progress, where the second
        overwrites the first on the same line.

    Raises
    ------
    ValueError
        ``offset`` provided without ``n_items``

    Examples
    --------
    This for loop: ::

        from time import sleep
        for i in print_progress(range(100)):
            sleep(1)

    prints a progress report of the form: ::

        [=============       ] 70% (7/10) - 00:00:03 remaining
    """
    if n_items is None and offset != 0:
        raise ValueError('offset can only be set when n_items has been'
                         ' manually provided.')
    if prefix != '':
        prefix = prefix + ': '
        bar_length = 10
    else:
        bar_length = 20
    n = n_items if n_items is not None else len(iterable)

    timings = deque([], 100)
    time1 = time()
    for i, x in enumerate(iterable, 1 + offset):
        yield x
        time2 = time()
        timings.append(time2 - time1)
        time1 = time2
        remaining = n - i
        duration = datetime.utcfromtimestamp(sum(timings) / len(timings) *
                                             remaining)
        bar_str = progress_bar_str(i / n, bar_length=bar_length,
                                   show_bar=show_bar)
        count_str = ' ({}/{})'.format(i, n) if show_count else ''
        eta_str = " - {} remaining".format(duration.strftime('%H:%M:%S')) \
            if show_eta else ''
        print_dynamic('{}{}{}{}'.format(prefix, bar_str, count_str, eta_str))

    # the iterable has now finished - to make it clear redraw the progress with
    # a done message. We also hide the eta at this stage.
    count_str = ' ({}/{})'.format(n, n) if show_count else ''
    bar_str = progress_bar_str(1, bar_length=bar_length, show_bar=show_bar)
    print_dynamic('{}{}{} - done.'.format(prefix, bar_str, count_str))

    if end_with_newline:
        print('')
