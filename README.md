# DeepFocus
Code for the PyTorch implementation of "DeepFocus: a Few-Shot Microscope Slide Auto-Focus using a Sample Invariant CNN-based Sharpness Function"

https://arxiv.org/abs/2001.00667
To be published at 2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI)

## Abstract
Autofocus (AF) methods are extensively used in biomicroscopy, for example to acquire timelapses, where the imaged objects tend to drift out of focus. AF algorithms determine an optimal distance by which to move the sample back into the focal plane. Current hardware-based methods require modifying the microscope and image-based algorithms either rely on many images to converge to the sharpest position or need training data and models specific to each instrument and imaging configuration. Here we propose DeepFocus, an AF method we implemented as a Micro-Manager plugin, and characterize its Convolutional neural network-based sharpness function, which we observed to be depth co-variant and sample-invariant. Sample invariance allows our AF algorithm to converge to an optimal axial position within as few as three iterations using a model trained once for use with a wide range of optical microscopes and a single instrument-dependent calibration stack acquisition of a flat (but arbitrary) textured object. From experiments carried out both on synthetic and experimental data, we observed an average precision, given 3 measured images, of 0.30 +- 0.16 micrometers with a 10x, NA 0.3 objective. We foresee that this performance and low image number will help limit photodamage during acquisitions with light-sensitive samples. 

## Requirements
The following python libraries are required. We advise the use of the conda package manager.
> numpy
> scipy
> scikit-image
> scikit-learn
> pytorch
> matplotlib
> numba
> lmfit
> grpcio
> protobuf
> fastai

For example, you can install all the requirements by using
> conda install --file detection/requirements.txt

## Installation
Copy-paste the contents of `javabinaries` in your Micro-manager main folder.

## How to run
Before starting Micro-Manager, run `python detection/run_grpc.py` to launch the processing server.

## Generate figures
To generate figures from the paper, run `python detection/calibration_fit.py` then `python detection/simulations_plot.py`

## Citation
For any use of the code or parts of the code, please cite:
>@INPROCEEDINGS{shajkofci2020deepfocus,
>    title={DeepFocus: a Few-Shot Microscope Slide Auto-Focus using a Sample Invariant CNN-based Sharpness Function},
>    author={Adrian Shajkofci and Michael Liebling},
>    year={2020},
>    isbn = {978-1-5386-9330-8/},
>    booktitle={2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI)}, 
>    pages={164--168}
>}
## Licence
This is free software: you can redistribute it and/or modify it under the terms of the BSD-3-Clause licence.
