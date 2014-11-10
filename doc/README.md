# Documentation for particles Dynare module

The documentation is written with sphinx, so you need to have
[Python](https://www.python.org/) on your system with
[sphinx](http://sphinx-doc.org/). You also need to install the [Read
the docs](https://readthedocs.org/) theme and the sphinx matlab
domain, this can be done with pip in a terminal:

```
 ~$ pip install sphinx_rtd_theme
 ~$ pip install sphinxcontrib-matlab
```

To obtain the documentation as an html file in subfolder
```build/html``` you just have to use the provided Makefile:

```
 ~$ make html
```