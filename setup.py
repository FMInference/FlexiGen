from setuptools import setup, find_packages

install_requires = [
    "numpy", "tqdm", "pulp", "attrs",
    "transformers>=4.24", "torch>=1.12",
]

setup(
    name="flexgen",
    version='1.0',
    description='Running large language models like OPT-175B/GPT-3 on a single GPU. Focusing on high-throughput large-batch generation.',
    python_requires='>=3.7',
    install_requires=install_requires,
    packages=find_packages(exclude=[
        "benchmark", "experiments", "playground", "scripts",
        "third_party",
    ]),
)
