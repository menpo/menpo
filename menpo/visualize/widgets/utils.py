from IPython.html.widgets import TextWidget


class IntListTextWidget():
    r"""
    Basic widget that returns a `list` of `int` numbers. It uses
    `IPython.html.widgets.TextWidget()` and converts its value to a `list` of
    `int`.

    Parameters
    -----------
    value : `str` or `list` of `int`, Optional
        The initial value of the widget.

    description : `str`, Optional
        The description of the widget.

    Raises
    ------
    ValueError
        value must be str or list
    """
    def __init__(self, value='', description=''):
        if isinstance(value, list):
            val = _convert_list_to_str(value)
        elif isinstance(value, str):
            val = value
        else:
            raise ValueError("value must be str or list")
        self.text_wid = TextWidget(value=val, description=description)

    @property
    def value(self):
        r"""
        The value fo the widget.
        """
        return _convert_str_to_list_int(str(self.text_wid.value))

    @property
    def description(self):
        r"""
        The description of the widget.
        """
        return self.text_wid.description

    @property
    def model_id(self):
        r"""
        The id of the widget.
        """
        return self.text_wid.model_id


class FloatListTextWidget(IntListTextWidget):
    r"""
    Basic widget that returns a `list` of `float` numbers. It uses
    `IPython.html.widgets.TextWidget()` and converts its value to a `list` of
    `float`.

    Parameters
    -----------
    value : `str` or `list` of `int`, Optional
        The initial value of the widget.

    description : `str`, Optional
        The description of the widget.

    Raises
    ------
    ValueError
        value must be str or list
    """
    @property
    def value(self):
        r"""
        The value fo the widget.
        """
        return _convert_str_to_list_float(str(self.text_wid.value))


def _convert_list_to_str(l):
    r"""
    Function that converts a given list of numbers to a string. For example:
        convert_list_to_str([1, 2, 3]) returns '1, 2, 3'
    """
    if isinstance(l, list):
        return str(l)[1:-1]
    else:
        return ''


def _convert_str_to_list_int(s):
    r"""
    Function that converts a given string to a list of int numbers. For example:
        _convert_str_to_list_int('1, 2, 3') returns [1, 2, 3]
    """
    if isinstance(s, str):
        return [int(i[:-1]) if i[-1] == ',' else int(i) for i in s.split()]
    else:
        return []


def _convert_str_to_list_float(s):
    r"""
    Function that converts a given string to a list of float numbers. For example:
        _convert_str_to_list_float('1, 2, 3') returns [1.0, 2.0, 3.0]
    """
    if isinstance(s, str):
        return [float(i[:-1]) if i[-1] == ',' else float(i) for i in s.split()]
    else:
        return []
