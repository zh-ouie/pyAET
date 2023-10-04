import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="pyaet", # Replace with your own name
    version="0.0.1",
    author="Long Yang",
    author_email="long_yang@tongji.edu.cn",
    description="AET reconstruction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yanglonggroup/pyAET",
    packages=setuptools.find_packages(),
    package_dir={"pyaet": "pyaet"},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
    entry_points={'console_scripts': [
            'pyaet = pyaet.main:main',],
        },
    data_files = [("", ["LICENSE.txt"])],
    python_requires='>=3.7',
    zip_safe=False,
)
