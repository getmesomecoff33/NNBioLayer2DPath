# NNBioLayerPathRender

This repository contains the modifications and scripts to render the decision path through a BioLayer-Network. For more information please check out the original project:
[Brain-inspired Modular Training](https://github.com/KindXiaoming/BIMT).

Modifyed to work with Google-Quickdraw-Dataset.

## Installation

WIP

## Dependencies

See requierments.txt
For using the Gui-Version tkinter is requiert

Additional requierments can occure du to the used packages

## Usage

Make a new file named "config.json" to the projects top level dir.
Within the file include the required datadirs. Required are:

* DataPath: Path where all the data is located.
* RawData: Dirname within DataPath where raw data is located.
* ImagePath: For single image evaluation, path to the image.
* ImageName: For single image evaluation, name of the image.

Run prepardata.py to create a torch trainingsdataset
run net.py to beginn the trainingprocess

## Limitations

* Only denselayer of size 100 are supported
  * (except the outputlayer)
* Only grayscale images
* WB with darkmode-images

## Tested

Tested under *Fedora release 39 (Thirty Nine)*

## Disclaimer

The renderd desicsion path is not representative for the actual image classification. It merely shows the highest, non-negative weights impact of the network.

## Licence

This code is released under the MIT License.

## Contributing

Contributions to this project are welcome. If you'd like to contribute, please follow the standard GitHub fork and pull request workflow.
