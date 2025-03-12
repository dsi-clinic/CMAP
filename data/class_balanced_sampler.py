"""Custom samplers for sampling patches from geospatial datasets.

Samplers avoid areas with missing labels by handling bounding boxes smaller than
the patch size, and balances the prevelance of classes within batches.
"""

import torch
from torchgeo.datasets import BoundingBox
from torchgeo.samplers import BatchGeoSampler
from torchgeo.samplers.constants import Units
from torchgeo.samplers.utils import (
    _to_tuple,
    get_random_bounding_box,
    tile_to_chips,
)


class ClassBalancedRandomBatchGeoSampler(BatchGeoSampler):
    """Samples batches of elements from a region of interest randomly.

    This is modified from RandomBatchGeoSampler so it can process bounding boxes
    that are smaller than the given size and thus provide data balanced between
    background and features. Note that randomly sampled chips may overlap.
    """

    def __init__(self, config) -> None:
        """Initialize a new Sampler instance.

        Args:
            config: A dictionary containing the following keys:
                - dataset: dataset to index from
                - size: dimensions of each patch
                - batch_size: number of samples per batch
                - length: number of samples per epoch
                - roi: region of interest to sample from
                - units: defines if size is in pixel or CRS units
                - NUM_CLASSES: number of classes in dataset [config.NUM_CLASSES]
        """
        self.config = config
        self.dataset = config["dataset"]
        super().__init__(config["dataset"], config.get("roi"))
        self.size = _to_tuple(config["size"])

        if config.get("units") is None or config.get("units") == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)

        self.batch_size = config["batch_size"]
        self.length = 0
        self.hits, self.adjusted_weight_tensor = self.calculate_hits_and_areas()

        if config.get("length") is not None:
            self.length = config["length"]

    def calculate_hits_and_areas(
        self,
    ):
        """Calculate hits (candidate patches) and  adjusted areas (weights) for the dataset.
            Weights are based on hits size (bounds) and per-class area. Larger hits and less frequent classes are oversampled.

        Returns:
            hits: List of hits within the region of interest.
            adjusted_areas: Tensor of adjusted areas corresponding to each hit.
        """
        hits = []
        weight = []
        context_x = self.size[1] / 2
        context_y = self.size[0] / 2

        for hit in self.index.intersection(tuple(self.roi), objects=True):
            bounds = BoundingBox(*hit.bounds)
            if (
                bounds.maxx - bounds.minx >= self.size[1]
                and bounds.maxy - bounds.miny >= self.size[0]
            ):
                shape_bounds = self.get_shape_bounds(bounds, context_x, context_y)
                if (
                    shape_bounds.maxx - shape_bounds.minx >= self.size[1]
                    or shape_bounds.maxy - shape_bounds.miny >= self.size[0]
                ):
                    rows, cols = tile_to_chips(shape_bounds, self.size)
                    self.length += rows * cols
                else:
                    self.length += 1
                hits.append(hit)

                fixed_bbox = get_random_bounding_box(
                    bounds, self.size, self.res
                )  
                sample = self.dataset[fixed_bbox]
                patch_mask_int = (
                    sample["mask"].squeeze().long()
                ) 
                total_pixels = patch_mask_int.numel()
                class_pixel_counts = [
                    (patch_mask_int == i).sum().item()
                    for i in range(self.config["NUM_CLASSES"])
                ]
                patch_distribution = {
                    i: class_pixel_counts[i] / total_pixels
                    for i in range(self.config["NUM_CLASSES"])
                }
                underrepresented_score = patch_distribution.get(
                    3, 0
                ) + patch_distribution.get(4, 0)
                adjusted_weight = shape_bounds.area * (1 + 10 * underrepresented_score)
                weight.append(adjusted_weight)

        adjusted_weight_tensor = torch.tensor(weight, dtype=torch.float)
        if torch.sum(adjusted_weight_tensor) == 0:
            adjusted_weight_tensor += 1

        return hits, adjusted_weight_tensor

    def get_shape_bounds(self, bounds, context_x, context_y):
        """Get adjusted shape bounds considering context.

        Args:
            bounds: Original bounding box bounds.
            context_x: Context size in x direction.
            context_y: Context size in y direction.

        Returns:
            BoundingBox: Adjusted bounding box.
        """
        return BoundingBox(
            bounds.minx + context_x,
            bounds.maxx - context_x,
            bounds.miny + context_y,
            bounds.maxy - context_y,
            bounds.mint,
            bounds.maxt,
        )

    def __iter__(self):
        """Return a batch of indices of a dataset.

        Yields:
            batch of (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        for _ in range(len(self)):
            batch = []
            for _ in range(self.batch_size):
                idx = torch.multinomial(
                    self.adjusted_weight_tensor, 1
                ).item()  # Weighted sampling happens here
                hit = self.hits[idx]
                bounds = BoundingBox(*hit.bounds)
                bounding_box = get_random_bounding_box(bounds, self.size, self.res)
                batch.append(bounding_box)
            yield batch

    def __len__(self):
        """Returns the number of samples that this sampler will draw.

        Returns:
            int: The length of the sampler.
        """
        return self.length // self.batch_size
