from torchgeo.datasets import RasterDataset


class KaneCounty(RasterDataset):
    filename_glob = "mask_m*.tif"
    filename_regex = r"""
        ^mask_m
        _(?P<quadrangle>\d+)
        _(?P<quarter_quad>[a-z]+)
        _(?P<utm_zone>\d+)
        _(?P<resolution>\d+)
        _(?P<date>\d+)
        (?:_(?P<processing_date>\d+))?
        \..*$
    """
    all_bands = ["R", "G", "B", "NIR"]
    rgb_bands = ["R", "G", "B"]
    is_image = False
    cmap = {
        0: (255, 255, 255, 255),
        1: (0, 197, 255, 255),
        2: (0, 168, 132, 255),
        3: (38, 115, 0, 255),
        4: (76, 230, 0, 255),
        5: (163, 255, 115, 255),
        6: (255, 170, 0, 255),
        7: (255, 0, 0, 255),
        8: (156, 156, 156, 255),
        9: (0, 166, 130, 255),
        10: (115, 115, 0, 255),
        11: (230, 230, 0, 255),
        12: (255, 255, 115, 255),
        13: (197, 0, 255, 255),
        14: (124, 211, 255, 255),
        15: (226, 0, 124, 255),
    }

    labels = {
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
        15: "",
    }

    labels_inverse = {v: k for k, v in labels.items()}
