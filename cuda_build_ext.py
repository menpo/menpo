# CUDA compilation is adapted from the source
# https://github.com/rmcgibbo/npcuda-example

import os
from distutils.command.build_ext import build_ext
from os import path
from os.path import join as pjoin
import re

def find_in_path(name, path):
    """
    Find a file in a search path
    """
    
    # adapted from:
    #   http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda():
    """
    Locate the CUDA environment on the system
    
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    
    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """
    
    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            return None
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in cudaconfig.iteritems():
        if not os.path.exists(v):
            return None

    return cudaconfig
CUDA = locate_cuda()

def convert_to_cuda_pyx(pyx_filename):
    """
    Generate the CUDA equivalent for pyx_filename
    """
    
    module_path, pyx_shortname = path.split(pyx_filename)
    if pyx_shortname.startswith("cpp"):
        cupyx_filename = pjoin(module_path, "cu" + pyx_shortname[3:])
    else:
        cupyx_filename = pjoin(module_path, "cu" + pyx_shortname)
    
    # Read the pyx file
    with open(pyx_filename, "r") as f:
        pyx_content = f.read()
    
    # Look for C/C++ headers
    tmp_sources = re.findall(
            r'\ncdef extern from "(?P<directory>c(pp?))/(?P<source_no_ext>[^/]+).h"',
            pyx_content)
    
    # Try to replace C/C++ header by CUDA header if it exists
    for sourcetuple in tmp_sources:
        directory = sourcetuple[0]
        source_no_ext = sourcetuple[2]
        tmp_source_name = pjoin(module_path, pjoin("cu", source_no_ext + ".h"))
        if os.path.isfile(tmp_source_name):
            pyx_content = re.sub(
                    r"{}/{}.(cpp|c\s|c$)".format(directory, source_no_ext),
                    r"cu/{}.cu".format(source_no_ext),
                    pyx_content)
            pyx_content = pyx_content.replace(
                    directory + "/" + source_no_ext + ".h",
                    "cu/" + source_no_ext + ".h")
    
    # Save the pyx file
    with open(cupyx_filename, "w+") as f:
        f.write(pyx_content)
    
    return cupyx_filename

def customize_compiler_for_nvcc(self):
    """
    inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """
    
    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        # CU module case
        elif isinstance(extra_postargs, dict) and 'gcc' in extra_postargs:
            postargs = extra_postargs['gcc']
        # When generating extensions for C/C++ modules
        # cythonize does not define extra_postargs['gcc']
        else:
            postargs = extra_postargs

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile

# run the customize_compiler
class cuda_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)
