from distutils.core import setup

# Read the README file for long description
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='py_lets_be_rational',
    version='1.1.0',  # Increment version for new functionality
    packages=['py_lets_be_rational'],
    url='http://jaeckel.org',
    license='MIT',
    maintainer='vollib',
    maintainer_email='vollib@gammoncap.com',
    description='Pure python implementation of Peter Jaeckel\'s LetsBeRational with Asian option support.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy',
        # scipy removed - using internal norm_cdf implementation
    ],
    extras_require={
        'fast': ['numba>=0.30.0'],  # Optional Numba acceleration
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Financial and Insurance Industry', 
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Office/Business :: Financial',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, !=3.5.*',
)