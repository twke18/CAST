from setuptools import setup

core_requirements = [
    "torch",
    "dgl",
    "timm",
]

setup(
    name='cast_models',
    version='0.1',
    description='CAST',
    author='Tsung-Wei Ke',
    author_email='mackintoshtoast@gmail.com',
    url='https://twke18.github.io',
    install_requires=core_requirements,
    packages=[
        'cast_models',
    ],
)
