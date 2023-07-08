from setuptools import setup, Extension
from torch.utils import cpp_extension
import os
import subprocess
import math

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"


p = int(subprocess.run("cat /proc/cpuinfo | grep cores | head -1", shell=True, check=True, text=True, stdout=subprocess.PIPE).stdout.split(" ")[2])

subprocess.call(["python", "generate.py", "--module", "--search", "--p", str(p)])

setup(
    name='cQIGen',
    ext_modules=[cpp_extension.CppExtension(
        'cQIGen', ['backend.cpp'],
        extra_compile_args = ["-O3", "-mavx", "-mavx2", "-mfma", "-march=native", "-ffast-math", "-ftree-vectorize", "-faligned-new", "-std=c++17", "-fopenmp", "-fno-signaling-nans", "-fno-trapping-math"]
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
