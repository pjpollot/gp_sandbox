from setuptools import setup

# ---------------- VARIABLES ------------------------

AUTHOR = 'Pierre-Jean POLLOT'

EMAIL = 'pierre-jean.pollot@laposte.net'

VERSION = '0.1.0'

MAIN_PACKAGE = 'gp_sandbox'

SUB_PACKAGES = [
    'test',
    'gaussian_processes',
    'optimization',
    'utils',
]

SCRIPTS = [

]

URL = ''

DESCRIPTION = ''

packages = [
    MAIN_PACKAGE + '.' + package for package in SUB_PACKAGES
]


# -------------------- RUNNING ---------------------

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