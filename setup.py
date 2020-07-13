import os
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()


reqs = []
for ir in required:
    if ir[0:3] == 'git':
        name = ir.split('/')[-1]
        reqs += ['%s @ %s@master' % (name, ir)]
    else:
        reqs += [ir]


# If interested in benchmarking devito, we need the `examples` too
exclude = ['docs', 'tests']
try:
    if not bool(int(os.environ.get('DEVITO_BENCHMARKS', 0))):
        exclude += ['examples']
except (TypeError, ValueError):
    exclude += ['examples']

setup(name='devito',
      description="Finite Difference Tutorials using Devito.",
      long_description="""The Devito Book is a set of tutorials adapted from
      The Craft of Finite Difference Computing with Partial Differential Equations
      by Hans Petter Langtangen and Svein Linge. These tutorials use Devito, 
      a new tool for performing optimised Finite Difference (FD) computation 
      from high-level symbolic problem definitions. Devito performs automated code
      generation and Just-In-time (JIT) compilation based on symbolic
      equations defined in SymPy to create and execute highly
      optimised Finite Difference kernels on multiple computer
      platforms.""",
      url='http://www.devitoproject.org',
      author="Imperial College London",
      author_email='g.gorman@imperial.ac.uk',
      license='MIT',
      packages=find_packages(exclude=exclude),
      install_requires=reqs,
      test_suite='tests')