from setuptools import setup

setup(
    name="modelforge",
    version="0.1",
    packages=["modelforge"],
    package_data={
        "modelforge": [
            "dataset/yaml_files/*",
            "curation/yaml_files/*",
            "tests/data/potential_defaults/*",
            "tests/data/training_defaults/*",
        ]
    },
    url="https://github.com/choderalab/modelforge",
    license="MIT",
    author="Chodera lab, Marcus Wieder, Christopher Iacovella, and others",
    author_email="",
    description="A library for building and training neural network potentials",
    include_package_data=True,
)
