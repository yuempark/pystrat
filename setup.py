import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pystrat",
    version="1.0.0",
    author="Yuem Park",
    author_email="yuempark@gmail.com",
    description="Convenient stratigraphic plotting and analysis in Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yuempark/pystrat",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
