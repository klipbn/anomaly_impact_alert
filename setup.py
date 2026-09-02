from setuptools import setup, find_packages

setup(
    name="anomaly_impact_alert",
    version="0.4.12",
    description="Anomaly detection, impact explanation, forecasting, and alerting toolkit",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Alexey Voronko",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.3",
        "numpy>=1.21",
        "bottleneck>=1.3",
        "scipy>=1.7",
        "scikit-learn>=1.0",
        "matplotlib>=3.5",
        "prophet>=1.1",
        "statsmodels>=0.13",
        "requests>=2.0",
        "holidays>=0.20",
    ],
    include_package_data=True,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
)
