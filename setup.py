#!/usr/bin/env python3
import setuptools

setuptools.setup(
    name="IMFNet",
    version="0.0.0.dev0",
    packages=["imfnet", "imfnet.util"],
    package_dir={
        "imfnet": ".",
    },
)
