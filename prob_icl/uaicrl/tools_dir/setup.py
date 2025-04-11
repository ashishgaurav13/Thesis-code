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
        'torch==1.13.1', 'numpy==1.23.5', 'pandas==2.2.2', 'scikit-learn==1.5.2', 'POT', 'tensorflow==2.15.0',
            # Neural networks and machine learning
        'gym==0.25.2', 'pyglet', 'Pillow', 'pygame', # Reinforcement learning
        'tensorboard', 'matplotlib', 'tqdm', 'plotly', 'seaborn', # Plotting & progress
        'numba', # JIT
        'joblib', 'mujoco',
        'beautifulsoup4', 'shapely', 'utm',
    ],
)