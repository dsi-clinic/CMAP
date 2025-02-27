"""Custom samplers for sampling patches from geospatial datasets.

Samplers avoid areas with missing labels by handling bounding boxes smaller than
the patch size. This ensures balanced sampling between background and feature
areas.
"""

import math

import torch
from torchgeo.datasets import BoundingBox
from torchgeo.samplers import BatchGeoSampler, GeoSampler
from torchgeo.samplers.constants import Units
from torchgeo.samplers.utils import (
    _to_tuple,
    get_random_bounding_box,
    tile_to_chips,
)


class BalancedRandomBatchGeoSampler(BatchGeoSampler):
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
        """
        super().__init__(config["dataset"], config.get("roi"))
        self.size = _to_tuple(config["size"])

        if config.get("units") is None or config.get("units") == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)

        self.batch_size = config["batch_size"]
        self.length = 0
        self.hits, self.areas = self.calculate_hits_and_areas()

        if config.get("length") is not None:
            self.length = config["length"]

    def calculate_hits_and_areas(
        self,
    ):  # Balancing (between background and features, i.e any class) happens here! We are creating different sized patches?!
        """Calculate hits and areas for the dataset.

        Returns:
            hits: List of hits within the region of interest.
            areas: Tensor of areas corresponding to each hit.
        """
        hits = []
        areas = []
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
                areas.append(shape_bounds.area)

        areas_tensor = torch.tensor(areas, dtype=torch.float)
        if torch.sum(areas_tensor) == 0:
            areas_tensor += 1

        return hits, areas_tensor

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
                idx = torch.multinomial(self.areas, 1)
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


class BalancedGridGeoSampler(GeoSampler):
    """Samples elements in a grid-like fashion.

    This is modified from GridGeoSampler so it can process bounding boxes
    that are smaller than the given size and thus provide data balanced between
    background and features.
    """

    def __init__(self, config) -> None:
        """Initialize a new Sampler instance.

        Args:
            config: A dictionary containing the following keys:
                - dataset: dataset to index from
                - size: dimensions of each patch
                - stride: distance to skip between each patch
                - roi: region of interest to sample from
                - units: defines if size and stride are in pixel or CRS units
        """
        super().__init__(config["dataset"], config.get("roi"))
        self.size = _to_tuple(config["size"])
        self.stride = _to_tuple(config["stride"])

        if config.get("units") is None or config.get("units") == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)
            self.stride = (self.stride[0] * self.res, self.stride[1] * self.res)

        self.hits, self.length = self.calculate_hits_and_length()

    def calculate_hits_and_length(self):
        """Calculate hits and total length for the dataset.

        Returns:
            hits: List of hits within the region of interest.
            length: Total number of samples.
        """
        hits = []
        total_length = 0
        context_x = self.size[1] / 2
        context_y = self.size[0] / 2

        for hit in self.index.intersection(tuple(self.roi), objects=True):
            bounds = BoundingBox(*hit.bounds)
            hit_bounds, rows, cols = self.get_hit_bounds_and_dimensions(
                bounds, context_x, context_y
            )

            if rows > 0 and cols > 0:
                hit.bounds = hit_bounds
                hits.append(hit)
                total_length += rows * cols

        return hits, total_length

    def get_hit_bounds_and_dimensions(self, bounds, context_x, context_y):
        """Get adjusted hit bounds and calculate dimensions.

        Args:
            bounds: Original bounding box bounds.
            context_x: Context size in x direction.
            context_y: Context size in y direction.

        Returns:
            tuple: Adjusted bounding box, number of rows, number of columns.
        """
        minx, maxx = bounds.minx, bounds.maxx
        miny, maxy = bounds.miny, bounds.maxy
        mint, maxt = bounds.mint, bounds.maxt

        cols = self.calculate_dimension(
            (minx, maxx), self.size[1], self.stride[1], context_x
        )
        rows = self.calculate_dimension(
            (miny, maxy), self.size[0], self.stride[0], context_y
        )

        if cols == 0 or rows == 0:
            return bounds, 0, 0

        return BoundingBox(minx, maxx, miny, maxy, mint, maxt), rows, cols

    def calculate_dimension(self, interval, size, stride, context):
        """Calculate the dimension for sampling.

        Args:
            interval: Minimum and maximum values of the dimension.
            size: Size of the patch.
            stride: Stride between patches.
            context: Context size to be subtracted.

        Returns:
            int: Number of patches in the dimension.
        """
        min_val, max_val = interval
        diff = max_val - min_val
        if diff >= size * 2:
            max_val -= context
            min_val += context
            return math.ceil((diff - size) / stride) + 1
        if diff >= size:
            extra_context = (diff - size) / 2
            max_val -= extra_context
            min_val = max_val - size
            return 1
        return 0

    def __iter__(self):
        """Return the index of a dataset.

        Yields:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        for hit in self.hits:
            bounds = BoundingBox(*hit.bounds)
            mint, maxt = bounds.mint, bounds.maxt
            rows, cols = tile_to_chips(bounds, self.size, self.stride)

            for i in range(rows):
                miny, maxy = self.get_min_max(
                    (bounds.miny, bounds.maxy),
                    (i, rows),
                    self.size[0],
                    self.stride[0],
                )
                for j in range(cols):
                    minx, maxx = self.get_min_max(
                        (bounds.minx, bounds.maxx),
                        (j, cols),
                        self.size[1],
                        self.stride[1],
                    )
                    yield BoundingBox(minx, maxx, miny, maxy, mint, maxt)

    def get_min_max(self, interval, indices, size, stride):
        """Get the min and max values for a given dimension.

        Args:
            interval: Minimum and maximum values of the dimension.
            indices: Current index and maximum index.
            size: Size of the patch.
            stride: Stride between patches.

        Returns:
            tuple: (min_val, max_val) adjusted for the current index.
        """
        min_val, max_val = interval
        idx, max_idx = indices
        min_val = min_val + idx * stride
        max_val = min_val + size
        if idx == max_idx - 1:
            min_val = max_val - size
        return min_val, max_val

    def __len__(self):
        """Return the number of samples over the ROI.

        Returns:
            int: Number of patches that will be sampled.
        """
        return self.length
