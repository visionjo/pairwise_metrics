# -*- coding: utf-8 -*-
from setuptools import setup, find_packages, Command
import io
import os
import sys

# Package meta-data.
NAME = 'pairmetrics'
DESCRIPTION = 'Package to evaluate and visualize results and distances of pairwise samples.'
URL = 'https://github.com/visionjo/pairwise_metrics'
EMAIL = 'robinson.jo@husky.neu.edu'
AUTHOR = 'Joseph Robinson'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.1.0'

# What packages are required for this module to be executed?
REQUIRED = [
    # 'requests', 'maya', 'records',
]

# What packages are optional?
EXTRAS = {
    # 'fancy feature': ['django'],
}

here = os.path.abspath(os.path.dirname(__file__))

try:
    # Import the README and use it as the long-description.
    # Note: this will only work if 'README.md' is present in your MANIFEST.in file!
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    LICENSE = f.read()

class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()

setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=readme,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    # If package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    python_requires=REQUIRES_PYTHON,
    include_package_data=True,
    extras_require=EXTRAS,
    license='MIT',

    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*", "docs"]),
)

