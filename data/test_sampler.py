"""Custom samplers for sampling patches from geospatial datasets.

Samplers avoid areas with missing labels by handling bounding boxes smaller than
the patch size. This ensures balanced sampling between background and feature
areas.
"""

from collections.abc import Iterator

import torch
from torchgeo.datasets import BoundingBox, GeoDataset
from torchgeo.samplers import BatchGeoSampler
from torchgeo.samplers.constants import Units
from torchgeo.samplers.utils import (
    _to_tuple,
    get_random_bounding_box,
    tile_to_chips,
)


class RandomBatchGeoSampler(BatchGeoSampler):
    """Samples batches of elements from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random :term:`chips <chip>` as possible. Note that
    randomly sampled chips may overlap.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: tuple[float, float] | float,
        batch_size: int,
        length: int | None = None,
        roi: BoundingBox | None = None,
        units: Units = Units.PIXELS,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        .. versionchanged:: 0.3
           Added ``units`` parameter, changed default to pixel units

        .. versionchanged:: 0.4
           ``length`` parameter is now optional, a reasonable default will be used

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch`
            batch_size: number of samples per batch
            length: number of samples per epoch
                (defaults to approximately the maximal number of non-overlapping
                :term:`chips <chip>` of size ``size`` that could be sampled from
                the dataset)
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` is in pixel or CRS units
        """
        super().__init__(dataset, roi)
        self.size = _to_tuple(size)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)

        self.batch_size = batch_size
        self.length = 0
        self.hits = []
        areas = []
        print("Debug: Initializing RandomBatchGeoSampler")
        print("Debug: len of index", len(self.index))
        print("Debug: roi", self.roi)
        for hit in self.index.intersection(tuple(self.roi), objects=True):
            bounds = BoundingBox(*hit.bounds)
            print("Debug: Processing hit with bounds", bounds)
            print("Debug: hit", hit)
            if (
                bounds.maxx - bounds.minx >= self.size[1]
                and bounds.maxy - bounds.miny >= self.size[0]
            ):
                print("Debug: Bounds are large enough for the given size")
                if bounds.area > 0:
                    rows, cols = tile_to_chips(bounds, self.size)
                    print(f"Debug: Calculated rows: {rows}, cols: {cols}")
                    self.length += rows * cols
                else:
                    print("Debug: Bounds area is zero or negative")
                    self.length += 1
                self.hits.append(hit)
                areas.append(bounds.area)
                print("Debug: Appended hit and area", hit, bounds.area)
            else:
                print("Debug: Bounds are too small for the given size")
        if length is not None:
            self.length = length
            print("Debug: Overriding length with provided value", length)

        # torch.multinomial requires float probabilities > 0
        self.areas = torch.tensor(areas, dtype=torch.float)
        print("Debug: Areas tensor", self.areas)
        if torch.sum(self.areas) == 0:
            self.areas += 1
            print("Debug: Adjusted areas to avoid zero sum")

    def __iter__(self) -> Iterator[list[BoundingBox]]:
        """Return the indices of a dataset.

        Returns:
            batch of (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        for _ in range(len(self)):
            # Choose a random tile, weighted by area
            idx = torch.multinomial(self.areas, 1)
            hit = self.hits[idx]
            bounds = BoundingBox(*hit.bounds)

            # Choose random indices within that tile
            batch = []
            for _ in range(self.batch_size):
                bounding_box = get_random_bounding_box(bounds, self.size, self.res)
                batch.append(bounding_box)

            yield batch

    def __len__(self) -> int:
        """Return the number of batches in a single epoch.

        Returns:
            number of batches in an epoch
        """
        return self.length // self.batch_size
