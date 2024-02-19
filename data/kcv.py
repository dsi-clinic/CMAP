from torchgeo.datasets import VectorDataset


class KaneCounty(VectorDataset):
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
    # all_bands = ["R", "G", "B", "NIR"]
    # rgb_bands = ["R", "G", "B"]
    # is_image = False

    label_names = {
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

    labels_inverse = {v: k for k, v in label_names.items()}
