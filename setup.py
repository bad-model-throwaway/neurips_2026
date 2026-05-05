# Build Cython extensions:
#   python3 setup.py build_ext --inplace

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "agents._cartpole_cy",
        ["agents/_cartpole_cy.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "agents._pointmass_cy",
        ["agents/_pointmass_cy.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="cartpole_mpc",
    ext_modules=cythonize(extensions, language_level=3),
)
