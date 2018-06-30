from setuptools import setup

setup(
    name='baad',
    version='0.1dev',
    description='Bayesian Addition of Astronomical Data',
    url='http://github.com/dkirkby/baad',
    author='David Kirkby',
    author_email='dkirkby@uci.edu',
    license='BSD3',
    packages=['baad'],
    install_requires=['numpy', 'scipy'],
    include_package_data=False,
    zip_safe=False,
    # Use pytest for testing.
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
