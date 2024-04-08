import random
from collections.abc import Iterator
from typing import Optional, Union

import torch
from torchgeo.datasets import BoundingBox, GeoDataset
from torchgeo.samplers import BatchGeoSampler, GeoSampler
from torchgeo.samplers.constants import Units
from torchgeo.samplers.utils import _to_tuple, tile_to_chips


class BalancedRandomBatchGeoSampler(BatchGeoSampler):
    """Samples batches of elements from a region of interest randomly.

    This is modified from RandomBatchGeoSampler so it can process bounding boxes
    that are smaller than the given size and thus provide data balanced between
    background and features. Note that randomly sampled chips may overlap.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[tuple[float, float], float],
        batch_size: int,
        length: Optional[int] = None,
        roi: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
    ) -> None:
        """Initialize a new Sampler instance.

        Args:
            dataset: dataset to index from
            size: dimensions of each patch, can either be:
                * a single ``float`` - in which case the same value is used for the height and
                width dimension
                * a ``tuple`` of two floats - in which case, the first *float* is used for the
                height dimension, and the second *float* for the width dimension
            batch_size: number of samples per batch
            length: number of samples per epoch
                (defaults to approximately the maximal number of non-overlapping
                chips of size ``size`` that could be sampled from
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
        for hit in self.index.intersection(tuple(self.roi), objects=True):
            bounds = BoundingBox(*hit.bounds)
            if (
                bounds.maxx - bounds.minx >= self.size[1]
                or bounds.maxy - bounds.miny >= self.size[0]
            ):
                if bounds.area > 0:
                    rows, cols = tile_to_chips(bounds, self.size)
                    self.length += rows * cols
            else:
                self.length += 1
            self.hits.append(hit)
            areas.append(bounds.area)
        if length is not None:
            self.length = length

        # torch.multinomial requires float probabilities > 0
        self.areas = torch.tensor(areas, dtype=torch.float)
        if torch.sum(self.areas) == 0:
            self.areas += 1

    def __iter__(self) -> Iterator[list[BoundingBox]]:
        """Return a batch of indices of a dataset.

        Returns:
            batch of (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        # get a list of all items in the region of interest
        items = list(self.index.intersection(tuple(self.roi), objects=True))

        for _ in range(len(self)):
            batch = []
            for _ in range(self.batch_size):
                # Choose a random item and get a bounding box for it
                sample = random.choice(items)
                bounds = BoundingBox(*sample.bounds)
                minx, maxx, miny, maxy = get_bounding_box(bounds, self.size, self.roi)
                batch.append(
                    BoundingBox(minx, maxx, miny, maxy, bounds.mint, bounds.maxt)
                )

            yield batch

    def __len__(self):
        """
        Returns the number of samples that this sampler will draw.

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

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[tuple[float, float], float],
        stride: Union[tuple[float, float], float],
        roi: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
    ) -> None:
        """Initialize a new Sampler instance.

        Args:
            dataset: dataset to index from
            size: dimensions of each patch, can either be:
                * a single ``float`` - in which case the same value is used for the height and
                width dimension
                * a ``tuple`` of two floats - in which case, the first *float* is used for the
                height dimension, and the second *float* for the width dimension
            stride: distance to skip between each patch
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` and ``stride`` are in pixel or CRS units
        """
        super().__init__(dataset, roi)
        self.size = _to_tuple(size)
        self.stride = _to_tuple(stride)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)
            self.stride = (self.stride[0] * self.res, self.stride[1] * self.res)

        self.hits = []
        for hit in self.index.intersection(tuple(self.roi), objects=True):
            self.hits.append(hit)

        self.length = 0
        for hit in self.hits:
            bounds = BoundingBox(*hit.bounds)
            if (
                bounds.maxx - bounds.minx >= self.size[1]
                or bounds.maxy - bounds.miny >= self.size[0]
            ):
                rows, cols = tile_to_chips(bounds, self.size, self.stride)
                self.length += rows * cols
            else:
                self.length += 1

    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        # For each tile...
        for hit in self.hits:
            bounds = BoundingBox(*hit.bounds)
            mint = bounds.mint
            maxt = bounds.maxt
            if (
                bounds.maxx - bounds.minx >= self.size[1]
                and bounds.maxy - bounds.miny >= self.size[0]
            ):
                rows, cols = tile_to_chips(bounds, self.size, self.stride)

                # For each row...
                for i in range(rows):
                    miny = bounds.miny + i * self.stride[0]
                    maxy = miny + self.size[0]
                    if maxy > self.roi.maxy:
                        miny, maxy = self.roi.maxy - self.size[0], self.roi.maxy

                    # For each column...
                    for j in range(cols):
                        minx = bounds.minx + j * self.stride[1]
                        maxx = minx + self.size[1]
                        if maxx > self.roi.maxx:
                            minx, maxx = self.roi.maxx - self.size[1], self.roi.maxx

                        yield BoundingBox(minx, maxx, miny, maxy, mint, maxt)
            else:
                minx, maxx, miny, maxy = get_bounding_box(bounds, self.size, self.roi)
                yield BoundingBox(minx, maxx, miny, maxy, mint, maxt)

    def __len__(self) -> int:
        """Return the number of samples over the ROI.

        Returns:
            number of patches that will be sampled
        """
        return self.length


def get_bounding_box(bounds, size, roi):
    """Compute the bounding box of a patch sampled from a tile which is
        smaller than the given size.
    Args:
        bounds: bounding box of tile
        size: size of output patch

    Returns: boundaries of x- and y-axis:
        minx, maxx, miny, maxy
    """
    # compute the center of the tile
    center_x = (bounds.minx + bounds.maxx) / 2
    center_y = (bounds.miny + bounds.maxy) / 2

    # compute the bounding box around the center with the given size
    minx = center_x - size[1] / 2
    maxx = center_x + size[1] / 2
    if minx < roi.minx:
        minx, maxx = roi.minx, roi.minx + size[1]
    elif maxx > roi.maxx:
        minx, maxx = roi.maxx - size[1], roi.maxx
    miny = center_y - size[0] / 2
    maxy = center_y + size[0] / 2
    if miny < roi.miny:
        miny, maxy = roi.miny, roi.miny + size[0]
    elif maxy > roi.maxy:
        miny, maxy = roi.maxy - size[0], roi.maxy
    return minx, maxx, miny, maxy
