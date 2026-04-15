from setuptools import setup, find_packages

setup(
    name="animRL",
    version="26.1",
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here, e.g., "numpy", "gym"
    ],
    author="Fatemeh Zargarbashi",
    author_email="fatemeh.zargarbashi@inf.ethz.ch",
    description="CMM 2026 - A3: Humanoid Motion Imitation with Deep Reinforcement Learning",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
)
