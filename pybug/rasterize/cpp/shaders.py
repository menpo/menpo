import os
import glob

"""
This short script finds all GLSL shader files in this directory and
creates a header of string literals.
Shader files are considered as all files in the shader subdirectory.

For example,

./shader/myshader.frag

generates:

const GLchar myshader_frag_str [] = "shader contents here"...

in the header file 'shaders.h'
"""

header_file = 'shaders.h'
c_folder = os.path.dirname(os.path.abspath(__file__))
shaders_folder = os.path.join(c_folder, 'shaders')
header_filepath = os.path.join(c_folder, header_file)

class Shader:
    def __init__(self, path):
        self.path = path
        self.shader_type = os.path.splitext(path)[-1][1:]
        self.name = os.path.splitext(os.path.split(path)[-1])[0]
        with open(path) as f:
            self.lines = f.readlines()
        self._c_string = convert_to_c_literal(self.lines)

    @property
    def c_literal(self):
        return 'const GLchar {}_{}_str [] = {};\n'.format(
            self.name, self.shader_type, self._c_string)


def convert_to_c_literal(lines):
    lines_in_quotes = ["\"{}\\n\"\n".format(l.strip()) for l in lines]
    return reduce(lambda a, b: a + b, lines_in_quotes)


def build_c_shaders():
    shader_paths = glob.glob(os.path.join(shaders_folder, '*'))
    shaders = [Shader(s) for s in shader_paths]
    lines = reduce(lambda a, b: a + b, [s.c_literal for s in shaders])
    with open(header_filepath, 'w') as f:
        f.write(lines)

if __name__ == '__main__':
    build_c_shaders()