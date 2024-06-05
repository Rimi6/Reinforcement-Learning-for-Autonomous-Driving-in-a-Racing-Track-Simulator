from setuptools import setup

setup(name='f110gym',
      version='0.2.1',
      author='Hongrui Zheng',
      author_email='billyzheng.bz@gmail.com',
      url='https://f1tenth.org',
      package_dir={'': 'f110gym'},
      install_requires=['gym==0.17.0',
		        'numpy<=1.24.3,>=1.18.0',
                        'Pillow>=9.0.1',
                        'scipy>=1.7.3',
                        'numba>=0.55.2',
                        'pyyaml>=5.3.1',
                        'pyglet<1.5',
                        'pyopengl']
      )
