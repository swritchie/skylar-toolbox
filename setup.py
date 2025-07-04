import setuptools 

setuptools.setup(
    name='skylar-toolbox',
    url='https://github.com/gopfsrisk/skylar-toolbox',
    author='Skylar Ritchie',
    packages=['skylar_toolbox'],
    zip_safe=False,
    install_requires=['scikit-learn', 'seaborn', 'statsmodels', 'toolz', 'tqdm'],
    extras_require={
        'catboost': ['catboost'],
        'networkx': ['networkx'],
        'optuna': ['optuna'],
        'lifelines': ['lifelines'],
        'scikit-survival': ['scikit-survival'],
        'complete': ['catboost', 'lifelines', 'networkx', 'optuna']})
