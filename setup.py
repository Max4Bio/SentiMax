import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sentimax-pkg-Max4Bio",
    version="0.0.1",
    author="Max4Bio",
    author_email="maximilian.rosen@gmx.de",
    description="Next-Geration Sentiment package for german",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Max4Bio/SentiMax/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
