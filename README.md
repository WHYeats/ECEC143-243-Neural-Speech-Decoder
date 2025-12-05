# ECE C143/243 Group Project (https://github.com/fwillett/speechBCI/tree/main/NeuralDecoder)

## Requirements
- python >= 3.9

## Installation

pip install -e .

Otherwise use the provided `environment.yaml`

## How to run

1. Convert the speech BCI dataset using [formatCompetitionData.ipynb](./notebooks/formatCompetitionData.ipynb)
2. Train model: `python ./scripts/train_model.py`

## Files Instructions
`./dataset/`: save the CompetitionDataset here. The dataset can be downlooaded from https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq.

`./notebook/show_lr.ipynb`: visualize the training stats.

`./src/neural_decoder/`: find models here.

`./logs/speech_logs/`: find the checkpoint and training stats here.

`./scripts/`: some training configurations.

## External Links:
Modelweights can be found https://drive.google.com/drive/folders/1gJrEZDsMa01PtgPK4-rhXLbdVTrtf5k7?usp=sharing. Modelweights should be downloaded in to `./logs/speech_logs/`.