from setuptools import setup

# ---------------- VARIABLES ------------------------

AUTHOR = 'Pierre-Jean POLLOT'

EMAIL = ''

VERSION = '1.0.0'

MAIN_PACKAGE = 'gp_sandbox'

SUB_PACKAGES = [
    'test',
    'gaussian_processes',
    'bayesian_optimization',
    'utils',
]

SCRIPTS = [

]

URL = 'https://github.com/pjpollot/gp_sandbox'

DESCRIPTION = 'A Python package towards Gaussian Processes and Bayesian Optimization'


# -------------------- SCRIPT ---------------------

packages = [
    MAIN_PACKAGE + '.' + package for package in SUB_PACKAGES
]

setup(
    name=MAIN_PACKAGE,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    packages=packages,
    scripts=SCRIPTS,
    url=URL,
    license='LICENSE.md',
    description=DESCRIPTION,
    long_description=open('README.md').read(),
)
