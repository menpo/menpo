r"""
Uses the mlabwrap package to create a global object called ``mlab``. This
object represents an instance of Matlab. The object facilitates communicating
with the Matlab instance. Methods can be called via the object as if they
were being called directly in Matlab.

Examples
--------
Calling the Matlab ``version`` method:

    >>> mlab.version()
    '8.1.0.604 (R2013a)'

Add external script to path and call it. Assumes the script exists at the path
specified. The script is called ``test`` and returns ``5``:

    >>> mlab.addpath('/home/user/matlab_scripts/')
    '/usr/local/MATLAB/R2013a/toolbox/local:/home/user/matlab_scripts/'
    >>> mlab.test()
    5
"""
from mlabwrap import MlabInstance
mlab = MlabInstance.get_instance()