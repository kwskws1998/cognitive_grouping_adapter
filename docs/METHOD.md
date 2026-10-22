# Proposed operator, version 0.1

## Hypothesis

Grouping can improve recognition if it preserves evidence specific to an individual item instead of exposing only the shared group representation. The classifier receives both evidence channels; it is not trained to reproduce human crowding errors.

## Input and learned parameters

The frozen backbone returns a spatial tensor `F[B,C,H,W]`. The single adapter applies channel LayerNorm and a learned `C -> D` projection, with `D=64` by default. Let `Z` be the projected descriptors and `A=ReLU(Z)` the nonnegative amplitudes. The operator has no learned parameters. A learned `3D -> C` projection combines its outputs and adds them to `F`.

The up-projection starts at zero, making the complete adapter an exact identity initially. The first optimizer update changes the up-projection; gradients reach the down-projection after that path becomes nonzero.

## Local spread

For each source and each cardinal direction, normalize the descriptor vector and compare it only with the immediate valid neighbor:

`s(i,j) = exp(-||normalize(Z_i)-normalize(Z_j)||^2 / (2*tau^2))`

`b(i,direction) = b_min + (b_max-b_min)*s(i,j)`

The outgoing source profile is `A_i * exp(-distance/b(i,direction))`. Width is measured in feature-grid cells. It is neither a retinal receptive-field estimate nor an eccentricity parameter. Similarity at a distant endpoint cannot directly set the source's outgoing width.

## Directional landscapes and segmentation proxy

For each target location, construct separate horizontal and vertical finite-window landscapes by summing source profiles at half-cell sample positions. Default radius 2 means five source positions and nine sample positions along each axis. Invalid padded sources never contribute to the landscape or normalization; there is no wraparound.

On each side of the center, locate the nearest strict local minimum. This is a piecewise-differentiable discrete selection, not a globally smooth approximation. If there is no side minimum, the side threshold is the center activity and the candidate central basin has zero extent. If there are no other active sources on that side, that side is explicitly treated as isolated.

The side numerator integrates positive landscape excess above its threshold, restricted to the central basin. The denominator integrates the whole valid side window. Trapezoidal weights are explicit. The segmentation proxy `p` is the minimum over the two sides of both axes, clipped to `[0,1]`. An inactive target has `p=0`; an active isolated target has `p=1`.

## Evidence channels

`U = A*p`

`S = A-U`

`G = mean of horizontal/vertical normalized spatial aggregation`

`F_adapted = F + W_up(concat(U,S,G))`

`U+S=A` is an accounting identity. It does not prove that either tensor contains psychologically identified private/shared neural activity. The nontrivial claim concerns whether the context-dependent split improves recognition and produces the predicted intervention pattern.

## Explicit departures from the original CODE model

- The paper's author implementation is a continuous-sampled 1D item model. This is a proposed directional extension over a 2D feature grid.
- Neural amplitudes vary; original item profiles are peak-normalized. We preserve amplitudes for adaptation.
- This implementation uses a bounded window and central basin. It does not reproduce the original global 0.05 region cutoff, derivative-based threshold search, behavioral guessing correction or eccentricity scaling.
- The author's code computes excess in the pooled landscape. This implementation also uses landscape excess, but bounds the numerator to the central basin to avoid assigning a neighboring stronger peak to the center.
- The outputs are adapter features, not predicted human accuracies.
- Gradients are valid away from local-minimum switching ties. No claim of everywhere differentiability is made.

## Required falsification

Compare the full split against fixed split, no-unique, fixed-width, uniform spatial and pairwise boundary controls. Use the same backbone, head, split, augmentation, sample budget and search budget. A gain over linear probing alone is insufficient. Measure clean accuracy and controlled clutter sensitivity; then run multiple real-data seeds. If a boundary control accounts for the gain, the proposed cognitive interpretation has not been isolated.
