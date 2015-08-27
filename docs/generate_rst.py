from inspect import isclass, isfunction, ismodule
from functools import partial

is_func_or_partial = lambda f: isfunction(f) or isinstance(f, partial)


def write_docs_for_module(module, path, modules_to_skip=None,
                          generate_index=False):
    if modules_to_skip is None:
        modules_to_skip = {}
    module_name = module.__name__
    doc_dir = path / module_name
    if not doc_dir.is_dir():
        doc_dir.mkdir()
    for k, v in module.__dict__.iteritems():
        if ismodule(v):
            file_to_doc = docs_for_module(k, v, module_name,
                                          generate_index=generate_index)
            if len(file_to_doc) == 0 or k in modules_to_skip:
                continue
            mod_dir = doc_dir / k
            if not mod_dir.is_dir():
                mod_dir.mkdir()
            for f_name in file_to_doc:
                doc_file = mod_dir / (f_name + '.rst')
                with open(str(doc_file), 'wb') as f:
                    f.write(file_to_doc[f_name])


def docs_for_module(module_name, module, package_name, generate_index=False):
    file_to_doc = {}
    for k, v in module.__dict__.iteritems():
        if isclass(v):
            file_to_doc[k] = generate_class_rst(module_name, k,
                                                module.__name__, package_name)
        elif is_func_or_partial(v):
            file_to_doc[k] = generate_function_rst(module_name, k,
                                                   module.__name__,
                                                   package_name)
    # only make an index if there is something to index
    if generate_index and len(file_to_doc) > 0:
        file_to_doc['index'] = generate_module_index(module_name, module)
    return file_to_doc


def generate_module_index(module_name, module):
    breadcrumb = '.. _api-{}-index:\n\n'.format(module_name)
    title = ":mod:`{}`".format(module.__name__)
    title = "{}\n{}\n".format(title, '=' * len(title))
    toctree = "\n.. toctree::\n  :maxdepth: 1\n\n  "
    items = [i for i, v in module.__dict__.items() if isclass(v) or
             is_func_or_partial(v)]
    return breadcrumb + title + toctree + "\n  ".join(items)


def generate_class_rst(module_name, class_name, module, package_name):
    breadcrumb = '.. _{}-{}-{}:\n\n'.format(package_name, module_name,
                                            class_name)
    current_module = '.. currentmodule:: {}\n\n'.format(module)
    title = "{}\n{}\n".format(class_name, '=' * len(class_name))
    body = (".. autoclass:: {}\n  :members:\n  :inherited-members:"
            "\n  :show-inheritance:\n".format(class_name))
    return breadcrumb + current_module + title + body


def generate_function_rst(module_name, function_name, module, package_name):
    breadcrumb = '.. _{}-{}-{}:\n\n'.format(package_name, module_name,
                                            function_name)
    current_module = '.. currentmodule:: {}\n\n'.format(module)
    title = "{}\n{}\n".format(function_name, '=' * len(function_name))
    body = ".. autofunction:: {}\n".format(function_name)
    return breadcrumb + current_module + title + body



if __name__ == '__main__':
    from pathlib import Path
    import menpo

    path = Path(__file__).parent / 'source' / 'api'

    # Flip generate_index to True to make index.rst files too!
    write_docs_for_module(menpo, path, generate_index=False,
                          modules_to_skip={'_version'})
