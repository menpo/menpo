import sys


def progress_bar_str(percentage, bar_length=20, bar_marker='=', show_bar=True):
    r"""
    Returns an str for the specified progress percentage. It can be combined
    with the print_dynamic function.

    Parameters
    ----------
    percentage : float
        The percentage that will be included in the output string. It must be
        in the range [0, 1].
    bar_length : int, optional
        Defines the length of the bar in characters.

        Default: 20
    bar_marker : str, optional
        Defines the marker that will be used to fill the bar.

        Default: '='
    show_bar : bool, optional
        If True, the str includes the bar and then the percentage,
        e.g. '[=====     ] 50%'
        If False, the str includes only the percentage,
        e.g. '50%'

    Returns
    -------
    progress_str : str
        The progress percentage string.

    Raises
    ------
    ValueError
        percentage is not in the range [0, 1]
    ValueError
        bar_length must be an integer >= 1
    ValueError
        bar_marker must be a string of length 1
    """
    if percentage > 1 or percentage < 0:
        raise ValueError("percentage is not in the range [0, 1]")
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


def print_dynamic(str_to_print=''):
    r"""
    Dynamically prints the given string. This means that it prints the string
    and then flushes the buffer.

    Parameters
    ----------
    str_to_print : str
        The string to print.
    """
    sys.stdout.write("\r%s" % str_to_print)
    sys.stdout.flush()


def print_bytes(num):
    r"""
    Returns a string of size provided in num with the appropriate format.
    e.g. print_bytes(12345) returns '12.06 KB'
         print_bytes(123456789) returns '117.74 MB'

    Parameter
    ---------
    num : int > 0
        The size in bytes.
    """
    if not isinstance(num, int) or num < 0:
        raise ValueError("num must be int >= 0")
    for x in ['bytes', 'KB', 'MB', 'GB']:
        if num < 1024.0:
            return "{0:3.2f} {1:s}".format(num, x)
        num /= 1024.0
    return "{0:3.2f} {1:s}".format(num, 'TB')
