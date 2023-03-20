from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

extras = dict()
extras["all"] = [
    "wandb>=0.10.30",
    "hydra-core>=1.1.0",
    "ipython",
    "ipykernel",
    "jupyter",
    "matplotlib",
    "seaborn",
    "rich"
]
keywords = ["weak supervision", "data programming", "deep learning", "pytorch", "pytorch lightning", "weasel", "hydra", "CMU"]

setup(
    name='weasel',
    version='0.1.0',
    packages=find_packages(
        exclude=["tests", "*.tests", "*.tests.*", "tests.*"]
    ),
    url='https://github.com/autonlab/weasel',
    license='Apache 2.0',
    author='Salva Rühling Cachay',
    author_email='salvaruehling@gmail.com',
    description='Learn any neural net for classification from multiple noisy heuristics only. No training labels!',
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=keywords,
    extras_require=extras,
    python_requires=">=3.7.0",
    install_requires=[
        "torch>=1.7.1",
        'pytorch-lightning==1.7.7',
        "scikit-learn",
        "pyyaml"
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],

)
