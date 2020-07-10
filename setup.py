import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TopicModellingPipeline",
    version="0.0.1",
    author="Arthur Karamov",
    author_email="ybiz177@gmail.com",
    description="tools for topic modelling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/karamov-git/TopicModellingPipeline",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GNU General Public License v3.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
