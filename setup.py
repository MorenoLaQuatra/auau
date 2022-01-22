
from setuptools import setup, find_packages

version = open('VERSION').read().strip()
license = open('LICENSE').read().strip()

setup(
    name = 'auau',
    version = version,
    license = license,
    author = 'Moreno La Quatra',
    author_email = 'moreno.laquatra@gmail.com',
    url = 'morenolaquatra.github.io',
    description = '(Au)dio (Au)gmentation = auau',
    long_description = open('README.md').read().strip(),
    packages = find_packages(),
    install_requires=[
        # put packages here
        'six',
        'librosa',
        'torchaudio',
        'numpy',
        'soundfile',
        'torch',
        'audiomentations',
        'torch-audiomentations'
    ],
    test_suite = 'tests',
    entry_points = {
	    'console_scripts': [
	        'packagename = packagename.__main__:main',
	    ]
	}
)