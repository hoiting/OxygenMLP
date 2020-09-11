from setuptools import setup, find_packages

setup(
    name='OxygenMLP',
    version='0.2',
    packages=find_packages(exclude=['tests*']),
#    license='MIT',
    description='neural network, multi-layer perceptron model that can predict oxygen abundance using strong emission lines',
    long_description=open('README.md').read(),
#    install_requires=['numpy', 'scklearn'],
    url='https://github.com/hoiting/OxygenMLP',
    author='I-Ting Ho',
    author_email='iting@mpia.de'
)
