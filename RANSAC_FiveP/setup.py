from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from os.path import join

project_root = 'essential_matrix'  # 它定义了project_root变量，指定了项目的根目录

# 使用列表推导式将 源文件的路径 保存在sources变量中
# 一个CUDA源文件(.cu)和一个C++源文件(.cpp)
sources = [join(project_root, file) for file in ["essential_matrix.cu",
                                                 "essential_matrix_wrapper.cpp"]]

# 最后，通过调用setup函数来配置扩展库的信息。
# 其中，name参数指定了库的名称，ext_modules参数指定了要构建的扩展模块，这里使用了CUDAExtension类，并传入了之前定义的源文件列表。
# cmdclass参数指定了在构建过程中使用的命令类，这里使用了BuildExtension类。

setup(
    name='essential_matrix',
    ext_modules=[
        CUDAExtension('essential_matrix',
                      sources),  # extra_compile_args, extra_link_args
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    })

# 在终端中导航到setup.py文件所在的目录。
