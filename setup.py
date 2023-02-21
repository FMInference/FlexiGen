from setuptools import setup, find_packages

install_requires = [
    "numpy", "tqdm", "pulp", "attrs",
    "transformers>=4.24", "torch>=1.12",
]

setup(name="flexgen",
      install_requires=install_requires,
      packages=find_packages(exclude=[
          "benchmark", "experiments", "playground", "scripts",
          "third_party",
      ]))
