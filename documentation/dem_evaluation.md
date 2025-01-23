Updated 1/22/2025 -

All image and spatial augments work individually, as set in **SPATIAL_AUG_INDICES** and **IMAGE_AUG_INDICES** within config.py. When working together, however, some indices are no longer activated - resized crop, for example, seems to not always be present and needs to be investigated further to insure.

Input DEM data is consistently aligned correctly with RGB data on main. After augmentations, DEM remains aligned with RGB.

Using print statements, DEM has been confirmed to be integrated to the model consistently when activated.

## Sample IOU Results w/o DEM & w/ all augs:
Jaccard index: 0.566
Test avg loss: 1.739
IoU for BACKGROUND: 0.000
IoU for POND: 0.767
IoU for WETLAND: 0.367
IoU for DRY BOTTOM - TURF: 0.345
IoU for DRY BOTTOM - MESIC PRAIRIE: 0.122

Training result: [0.6516666412353516],
average: 0.652, standard deviation: 0.000

Test result: [0.5659226775169373],
average: 0.566, standard deviation:0.000

## Sample IOU Results w/ DEM & w/ all augs:
Jaccard index: 0.422
Test avg loss: 1.767
IoU for BACKGROUND: 0.000
IoU for POND: 0.634
IoU for WETLAND: 0.228
IoU for DRY BOTTOM - TURF: 0.422
IoU for DRY BOTTOM - MESIC PRAIRIE: 0.091

Training result: [0.6246926188468933],
average: 0.625, standard deviation: 0.000

Test result: [0.4223737418651581],
average: 0.422, standard deviation:0.000

The results of the DEM tests return lower than running without the DEM. Randomness is involved in these tests, however, and this may not be an accurate reading of the effectiveness of DEM vs. no DEM.