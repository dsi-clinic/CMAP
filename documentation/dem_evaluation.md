# Performance
Updated 1/27/2025 -

All image and spatial augments work individually, as set in **SPATIAL_AUG_INDICES** and **IMAGE_AUG_INDICES** within config.py. When working together, however, some indices are no longer activated - resized crop, for example, seems to not always be present and needs to be investigated further to insure.

Input DEM data is consistently aligned correctly with RGB data on main. After augmentations, DEM remains aligned with RGB.

Using print statements, DEM has been confirmed to be integrated to the model consistently when activated.

Running with all spatial augs(0-6) and some of the image augs(0-3) is the default test and what achieves best results without DEM. It seems that using all augs on the non-DEM dataset achieves similar results to running img augs 0-3 on the DEM dataset. It is worth investigating whether those specific image augmentations are making the DEM input worse specifically, but improving the RGB input. If this is the case, one possible solution might be running those on just the RGB images and not the DEM inputs.

# NON-DEM TESTS
## Sample IOU Results w/o DEM & spatial augs 0-6 & image augs 0-3:
Jaccard index: 0.510
Test avg loss: 1.726
IoU for BACKGROUND: 0.000
IoU for POND: 0.739
IoU for WETLAND: 0.311
IoU for DRY BOTTOM - TURF: 0.380
IoU for DRY BOTTOM - MESIC PRAIRIE: 0.124

Training result: [0.633846640586853],
average: 0.634, standard deviation: 0.000

Test result: [0.5102429986000061],
average: 0.510, standard deviation:0.000

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

# DEM TESTS
## Sample IOU Results w/DEM & spatial augs 0-6 & image augs 0-3:
Jaccard index: 0.545
Test avg loss: 1.788
IoU for BACKGROUND: 0.000
IoU for POND: 0.724
IoU for WETLAND: 0.381
IoU for DRY BOTTOM - TURF: 0.438
IoU for DRY BOTTOM - MESIC PRAIRIE: 0.086

Training result: [0.6484153866767883],
average: 0.648, standard deviation: 0.000

Test result: [0.5446481108665466],
average: 0.545, standard deviation:0.000

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

# Processes Involved
Updated 1/28/2025:
The DEM files are normalized on an image by image basis, though the RGB files are normalized across the entire population of images. This is working as intended. The Bare Earth DEM is used rather than the Hydro-Enforced DEM. This is wonderful, as long as it is being processed correctly. Data and Kane County documentation can be found in /net/projects/cmap/data/kane-county-data.

Castillo et al.(2014, doi: 10.1002/esp.3595) has more detail on normalization techniques necessary, and recommends normalizing slope data. In this case, the filled/difference DEM is the equivalent of slope data.


# Next Steps:
* Adjust plot.py to display difference DEM correctly
* Run diagnostic tests on these configurations:
    * DEM w/ all spatial augs
    * DEM w/ no spatial augs
    * DEM w/ 3/ spatial augs
    * DEM w/ no augs at all
    * Adjust if augs are seen to be unfit
* Run equivalent diagnostic tests without DEM
    * Num_trials to be used?



## ONE SPATIAL CURRENTLY MUST BE APPLIED
DEM w/ all spatial:
