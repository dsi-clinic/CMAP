from torchgeo.datasets import RasterDataset


class KaneCounty(RasterDataset):
    filename_glob = "mask_*.tif"
    filename_regex = r"""
        ^mask
        _label(?P<label>\d+)
        _(?P<shape_idx>\d+)
        _m
        _(?P<quadrangle>\d+)
        _(?P<quarter_quad>[a-z]+)
        _(?P<utm_zone>\d+)
        _(?P<resolution>\d+)
        _(?P<date>\d+)
        (?:_(?P<processing_date>\d+))?
        \..*$
    """
    all_bands = ["Label"]
    is_image = False
    colors = {
        0: (0, 0, 0, 0),
        1: (215, 80, 48, 255),
        2: (49, 102, 80, 255),
        3: (239, 169, 74, 255),
        4: (100, 107, 99, 255),
        5: (89, 53, 31, 255),
        6: (2, 86, 105, 255),
        7: (207, 211, 205, 255),
        8: (195, 88, 49, 255),
        9: (144, 70, 132, 255),
        10: (29, 51, 74, 255),
        11: (71, 64, 46, 255),
        12: (114, 20, 34, 255),
        13: (37, 40, 80, 255),
        14: (94, 33, 41, 255),
        15: (255, 255, 255, 255),
    }

    labels = {
        0: "BACKGROUND",
        1: "POND",
        2: "WETLAND",
        3: "DRY BOTTOM - TURF",
        4: "DRY BOTTOM - MESIC PRAIRIE",
        5: "DEPRESSIONAL STORAGE",
        6: "DRY BOTTOM - WOODED",
        7: "POND - EXTENDED DRY",
        8: "PICP PARKING LOT",
        9: "DRY BOTTOM - GRAVEL",
        10: "UNDERGROUND",
        11: "UNDERGROUND VAULT",
        12: "PICP ALLEY",
        13: "INFILTRATION TRENCH",
        14: "BIORETENTION",
        15: "UNKNOWN",
    }

    labels_inverse = {v: k for k, v in labels.items()}
