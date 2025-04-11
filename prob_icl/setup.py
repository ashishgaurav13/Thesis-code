from setuptools import setup, find_packages

setup(
    name='tools',
    version='0.0.1',
    description='Tools for inverse constraint learning',
    author='Ashish Gaurav',
    author_email='ashish.gaurav@uwaterloo.ca',
    license='Proprietary',
    packages=['tools'],
    include_package_data=True,
    package_data={
        'tools': ['assets/driving/*.png', 'assets/highD/*', 
                  'assets/exiD/*', 'assets/mujoco/*']
    },
    install_requires=[
        'pytest', 'genbadge', 'coverage', # Testing
        'pyinterval', 'dill', # Interval objects and pickling
        'pdoc', # Documentation generation
        'torch', 'numpy', 'pandas', 'scikit-learn', 'POT', 'tensorflow',
            # Neural networks and machine learning
        'gym', 'pyglet', 'Pillow', 'pygame', # Reinforcement learning
        'tensorboard', 'matplotlib', 'tqdm', 'plotly', 'seaborn', # Plotting & progress
        'numba', # JIT
        'joblib', 'mujoco',
        'beautifulsoup4', 'shapely', 'utm',
    ],
)