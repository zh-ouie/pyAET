import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = [
        line.strip()
        for line in fh
        if line.strip() and not line.strip().startswith("#")
    ]


setuptools.setup(
    name="pyaet",
    version="0.0.1",
    author="Long Yang",
    author_email="long_yang@tongji.edu.cn",
    description="Atomic electron tomography reconstruction and atom tracing workflow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yanglonggroup/pyAET",
    packages=setuptools.find_namespace_packages(
        include=["pyaet*", "preprocessing*"],
        exclude=["pyaet.old", "pyaet.old.*"],
    ),
    package_dir={"pyaet": "pyaet"},
    package_data={"pyaet.src": ["*.mat", "*.m"],
                  "preprocessing": ["data/*.mat", "build_parity_fftw.sh"]},
    extras_require={"gpu-fft": ["triton>=3.1; platform_system == 'Linux'"], "parity": ["cholespy==2.2.0"], "package-bm3d": ["bm3d==4.0.3"]},
    install_requires=install_requires,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
    data_files=[("", ["LICENSE"])],
    python_requires='>=3.10',
    zip_safe=False,
)
