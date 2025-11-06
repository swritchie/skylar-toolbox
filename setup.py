import setuptools 

setuptools.setup(
    name='skylar-toolbox',
    url='https://github.com/gopfsrisk/skylar-toolbox',
    author='Skylar Ritchie',
    packages=['skylar_toolbox'],
    zip_safe=False,
    install_requires=['feature_engine', 'scikit-learn', 'seaborn', 'statsmodels', 'toolz', 'tqdm'],
    extras_require={
        'catboost': ['catboost'],
        'lifelines': ['lifelines'],
        'networkx': ['networkx'],
        'optuna': ['optuna'],
        'scikit-survival': ['scikit-survival'],
        'complete': ['catboost', 'lifelines', 'networkx', 'optuna', 'scikit-survival']})
