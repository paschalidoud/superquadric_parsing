"""Setup mesh-fusion"""

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension


def get_extensions():
   return cythonize([
       Extension(
           "learnable_primitives.fast_sampler._sampler",
           [
               "learnable_primitives/fast_sampler/_sampler.pyx",
               "learnable_primitives/fast_sampler/sampling.cpp"
           ],
           language="c++11",
           libraries=["stdc++"],
           extra_compile_args=["-std=c++11", "-O3"]
       )
   ])


def setup_package():
    setup(
        name="learnable_primitives",
        ext_modules=get_extensions()
    )

if __name__ == "__main__":
    setup_package()
