from setuptools import setup, find_packages

setup(
    name="anomaly_impact_alert",
    version="0.4.8",
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
        "scikit-learn>=1.0",
        "matplotlib>=3.5",
        "prophet>=1.1",
        "tqdm>=4.0",
        "requests>=2.0",
        "python-telegram-bot>=20.0",
    ],
    include_package_data=True,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
)
