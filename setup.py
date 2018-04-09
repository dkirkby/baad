from setuptools import setup

setup(
    name='coadd',
    version='0.1',
    description='Optimal Coaddition of Binned Data',
    url='http://github.com/dkirkby/coadd',
    author='David Kirkby',
    author_email='dkirkby@uci.edu',
    license='BSD3',
    packages=['coadd'],
    install_requires=['numpy', 'scipy'],
    include_package_data=False,
    zip_safe=False,
    # Use pytest for testing.
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)