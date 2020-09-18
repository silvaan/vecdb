import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="vecdb",
  version="1.0.1",
  author="Silvan Junior",
  author_email="jrmirannda@gmail.com",
  description="A very simple library to store, retrieve, compare, update and delete embedding vectors.",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/jrmiranda/vecdb",
  packages=setuptools.find_packages(),
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
  python_requires='>=3.6',
)