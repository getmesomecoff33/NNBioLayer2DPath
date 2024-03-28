# NNBioLayerPathRender

This repository contains the modifications and scripts to render the decision path through a BioLayer-Network. For more information please check out the original project:
[Brain-inspired Modular Training](https://github.com/KindXiaoming/BIMT)

## Installation

WIP

## Dependencies

See requierments.txt

Additional requierments can occure du to the used packages

## Usage

### Train Network

To train a network run `netz.py`
All training related options can be found under the main-section of `netz.py`

### Show Rendered Path

In `netz.py` set line 11:

    SHOWFIGURE = True

### Save Rendered Path

In `netz.py`set line 12:

    SAVEFIGURE = True

By default the images are saved in ./results/mnist/

### Take a Screenshot

Run `makescreenshot.py`
Currently not working du to attemted privilege escalation by the packackage.

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
