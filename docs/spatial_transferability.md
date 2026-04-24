# Spatial Transferability Demo

This note records an optional demonstration workflow using the original GOFLOW
training code. It is not part of the paper benchmark and does not change the
model architecture, loss definitions, or normalization setup.

## Setup

- Source data: an LLC-format NetCDF file with `loggrad_T`, `U`, and `V`
- Architecture: original GOFLOW `unet`, `nbase=16`
- Training recipe: stage 0 L1 training followed by stage 1 spectral fine-tuning
- Spatial split: user-provided 256 x 256 boxes, one held out per fold
- Metrics: held-out velocity R2 and gradient R2, where gradient R2 is the
  average of vorticity and strain R2 used by the training script

The public training script now supports `--regions_file` and `--fold` so these
spatial splits can be reproduced without changing the model architecture or
loss definitions.

## Region File Format

Pass a JSON file containing a `regions` list. Each region is a four-element
box in `(y0, y1, x0, x1)` order:

```json
{
  "regions": [
    [0, 256, 0, 256],
    [0, 256, 256, 512]
  ]
}
```

With `--fold 0`, the first region is held out for testing and the remaining
regions are used for training. Use `--metrics_file` to save the selected
checkpoint metrics for each fold.

## Example Stage 1 Fold Map

The figure below is an example stage-1 fold map from a larger LLC-domain demo.
It shows held-out fold locations with the selected-checkpoint velocity and
gradient R2 values. It is included to illustrate limitations of the existing
GOFLOW approach under spatial transfer, not to introduce a new method.

![Stage 1 spatial transferability demo](figures/spatial_transfer_stage1.png)
