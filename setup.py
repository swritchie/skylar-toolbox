import setuptools 

setuptools.setup(
    name='skylar-toolbox',
    url='https://github.com/gopfsrisk/skylar-toolbox',
    author='Skylar Ritchie',
    packages=['skylar_toolbox'],
    zip_safe=False,
    install_requires=['scikit-learn', 'seaborn', 'tqdm'],
    extras_require={
        'catboost': ['catboost'],
        'exploratory_data_analysis': ['networkx'],
        'optuna': ['optuna'],
        'preprocessing': ['lifelines'],
        'complete': ['catboost', 'lifelines', 'networkx', 'optuna']})
