from setuptools import setup, find_packages

setup(
    name='helper',
    version='0.0.1',
    packages=find_packages(include=['ct_sim', 'config.*', 'ct_sim.*']),
    install_requires=[
        'pandas==0.25.3',
        'names==0.3.0',
        'numpy==1.17.4',
        'matplotlib==3.1.2',
        'seaborn==0.9.0',
        'pymc3==3.7'
    ]
)