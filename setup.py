from setuptools import setup, find_packages

setup(
    name='hof_cognitive_model_training',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'transformers',
        'datasets',
        'huggingface_hub'
    ],
    entry_points={
        'console_scripts': [
            'fine_tuning=hof_cognitive_model_training.fine_tuning:main',
            'upload_model=hof_cognitive_model_training.upload_model:main',
            'cognitive_chaining=hof_cognitive_model_training.cognitive_chaining:main',
        ],
    },
)
