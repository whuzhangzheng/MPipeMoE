import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

cxx_flags = ["-I/usr/include", '-fPIC'] 
cuda_flags = ["-I/usr/include"] 
link_args = []

DEBUG = 0
if DEBUG:
    cxx_flags.extend(['-g', '-O0'])
    cuda_flags.extend(['-g', '-G'])
    link_args.extend(['-g'])
ext_libs = []

authors = [
        'Zheng Zhang',
        'Donglin Yang',
        'Yaqi Xia',
        'Liang Ding',
        'Dacheng Tao',
        'Xiaobo Zhou',
        'Dazhao Cheng'
]

#if os.environ.get('USE_NCCL', '0') == '1':
cxx_flags.append('-DUSE_NCCL')
cuda_flags.append('-DUSE_NCCL')

ext_libs.append('nccl')

if "NCCL_HOME" in os.environ:
    NCCL_HOME = os.environ["NCCL_HOME"]
    cxx_flags += [f"-I{NCCL_HOME}/include"]
    cuda_flags += [f"-I{NCCL_HOME}/include"]
    link_args += [f"-L{NCCL_HOME}/lib"]

if __name__ == '__main__':
    setuptools.setup(
        name='pipemoe',
        version='1.0.0',
        description='An efficient Mixture-of-Experts system for PyTorch',
        author=', '.join(authors),
        author_email='zzhang3031@whu.edu.cn',
        license='Apache-2',
        url='https://github.com/whuzhangzheng/MPipeMoE',
        packages=['pipemoe'],
        ext_modules=[
            CUDAExtension(
                name='pipemoe_cuda', 

                sources=[
                    'cuda/micro_compute.cu',
                    'cuda/micro_sharded_compute.cu',
                    'cuda/stream_manager.cpp',
                    'cuda/initialize.cu',
                    'cuda/pmoe_cuda.cpp',
                    ],
                extra_compile_args={
                    'cxx': cxx_flags,
                    'nvcc': cuda_flags
                    },
                extra_link_args = link_args,
                libraries=ext_libs
                )
            ],
        cmdclass={
            'build_ext': BuildExtension
        })
