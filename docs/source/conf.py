# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'ml-utils'
copyright = '2021, Filip Bronić'
author = 'Filip Bronić'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        # 'searchbox.html',
        # 'donate.html',
    ]
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# html_theme_options = {
#     'headerbordercolor': 'gray'
# }


# -- Skip select files
modules_to_skip = ["model.test"]


def skip_modules(app, what, name, obj, skip, options):
    print("Skipping ", skip, name)
    return ((what == "module") and (name in modules_to_skip)) or skip
    # if (what == "module") and name in modules_to_skip:
    #     return True
    # else:
    #     return skip


def text_process(app, what, name, obj, options, lines):
    print(what, name, lines)


def setup(app):
    app.connect('autodoc-skip-member', skip_modules)
    app.connect('autodoc-process-docstring', text_process)
