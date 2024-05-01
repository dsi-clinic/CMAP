import math
from collections.abc import Iterator
from typing import Optional, Union

import torch
from torchgeo.datasets import BoundingBox, GeoDataset
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
                * a single ``float`` - in which case the same value is used for
                  the height andwidth dimension
                * a ``tuple`` of two floats - in which case, the first *float*
                  is used for the height dimension, and the second *float* for
                  the width dimension
            batch_size: number of samples per batch
            length: number of samples per epoch
                (defaults to approximately the maximal number of non-overlapping
                chips of size ``size`` that could be sampled from
                the dataset)
            roi: region of interest to sample from
                (minx, maxx, miny, maxy, mint, maxt)
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
        context_x = self.size[1] / 2
        context_y = self.size[0] / 2
        for hit in self.index.intersection(tuple(self.roi), objects=True):
            bounds = BoundingBox(*hit.bounds)
            if (
                bounds.maxx - bounds.minx >= self.size[1]
                and bounds.maxy - bounds.miny >= self.size[0]
            ):
                # calculate length using shape size without context
                shape_bounds = BoundingBox(
                    bounds.minx + context_x,
                    bounds.maxx - context_x,
                    bounds.miny + context_y,
                    bounds.maxy - context_y,
                    bounds.mint,
                    bounds.maxt,
                )
                if (
                    shape_bounds.maxx - shape_bounds.minx >= self.size[1]
                    or shape_bounds.maxy - shape_bounds.miny >= self.size[0]
                ):
                    rows, cols = tile_to_chips(shape_bounds, self.size)
                    self.length += rows * cols
                else:
                    self.length += 1
                self.hits.append(hit)
                areas.append(shape_bounds.area)
        if length is not None:
            self.length = length

        self.areas = torch.tensor(areas, dtype=torch.float)
        if torch.sum(self.areas) == 0:
            self.areas += 1

    def __iter__(self) -> Iterator[list[BoundingBox]]:
        """Return a batch of indices of a dataset.

        Returns:
            batch of (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """

        for _ in range(len(self)):
            batch = []
            for _ in range(self.batch_size):
                # Choose a random tile, weighted by area
                idx = torch.multinomial(self.areas, 1)
                hit = self.hits[idx]
                bounds = BoundingBox(*hit.bounds)

                # Choose random indices within that tile
                bounding_box = get_random_bounding_box(
                    bounds, self.size, self.res
                )
                batch.append(bounding_box)

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
                * a single ``float`` - in which case the same value is used for
                  the height andwidth dimension
                * a ``tuple`` of two floats - in which case, the first *float*
                  is used for the height dimension, and the second *float* for
                  the width dimension
            stride: distance to skip between each patch
            roi: region of interest to sample from
                (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` and ``stride`` are in pixel or CRS units
        """
        super().__init__(dataset, roi)
        self.size = _to_tuple(size)
        self.stride = _to_tuple(stride)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)
            self.stride = (self.stride[0] * self.res, self.stride[1] * self.res)

        context_x = self.size[1] / 2
        context_y = self.size[0] / 2
        self.hits = []
        self.length = 0
        for hit in self.index.intersection(tuple(self.roi), objects=True):
            bounds = BoundingBox(*hit.bounds)
            minx, maxx = bounds.minx, bounds.maxx
            miny, maxy = bounds.miny, bounds.maxy
            mint, maxt = bounds.mint, bounds.maxt
            rows, cols = 0, 0
            # get rid of extra context
            x_diff = maxx - minx
            if x_diff >= self.size[1] * 2:
                maxx -= context_x
                minx += context_x
                cols = math.ceil((x_diff - self.size[1]) / self.stride[1]) + 1
            elif x_diff >= self.size[1]:
                extra_context = (x_diff - self.size[1]) / 2
                maxx -= extra_context
                minx = maxx - self.size[1]
                cols = 1
            else:
                continue

            y_diff = maxy - miny
            if y_diff >= self.size[0] * 2:
                maxy -= context_y
                miny += context_y
                rows = math.ceil((y_diff - self.size[0]) / self.stride[0]) + 1
            elif y_diff >= self.size[0]:
                extra_context = (y_diff - self.size[0]) / 2
                maxy -= extra_context
                miny = maxy - self.size[0]
                rows = 1
            else:
                continue

            hit.bounds = BoundingBox(minx, maxx, miny, maxy, mint, maxt)
            self.hits.append(hit)
            self.length += rows * cols

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
                if bounds.maxy - bounds.miny - self.size[0] < 0.1:
                    rows = 1
                if bounds.maxx - bounds.minx - self.size[1] < 0.1:
                    cols = 1

                # For each row...
                for i in range(rows):
                    miny = bounds.miny + i * self.stride[0]
                    maxy = miny + self.size[0]
                    if i == rows - 1:
                        maxy = bounds.maxy
                        miny = maxy - self.size[0]

                    # For each column...
                    for j in range(cols):
                        minx = bounds.minx + j * self.stride[1]
                        maxx = minx + self.size[1]
                        if j == cols - 1:
                            maxx = bounds.maxx
                            minx = maxx - self.size[1]

                        yield BoundingBox(minx, maxx, miny, maxy, mint, maxt)

    def __len__(self) -> int:
        """Return the number of samples over the ROI.

        Returns:
            number of patches that will be sampled
        """
        return self.length
