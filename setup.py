from setuptools import setup, find_packages

setup(
    name="hydronet",
    version="0.1.0",
    packages=find_packages(),
    # Runtime dependencies. ``mpi4py`` is intentionally omitted — it isn't
    # imported anywhere in the package and requires an MPI runtime that's
    # painful to set up on Windows. ``pyHMT2D`` is also omitted because
    # it isn't available on PyPI; install it editable from a sibling clone
    # (see README).
    install_requires=[
        # Core
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        # Plotting / visualisation
        "matplotlib>=3.7.0",
        "plotly>=5.13.0",
        # Mesh / data I/O
        "vtk>=9.2.0",
        "meshio>=5.3.0",
        "gmsh>=4.12.0",
        "h5py>=3.10.0",
        "xarray>=2023.1.0",
        "netCDF4>=1.6.0",
        # Geospatial utilities (PINN / PI-DeepONet data pipelines)
        "geopandas>=1.0.0",
        "shapely>=2.0.0",
        "pyogrio>=0.7.0",
        "pyproj>=3.6.0",
        # Config + training-loop plumbing
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "tensorboard>=2.15.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
            "sphinx>=6.1.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    author="Xiaofeng Liu",
    author_email="xiaofengliu19@gmail.com",
    description="A deep learning framework for operator learning and solving 2D shallow water equations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/psu-efd/HydroNet",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.10",
)
