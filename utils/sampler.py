import random
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Optional, Union

import torch
from torch.nn import functional as F
from torchgeo.datasets import BoundingBox, GeoDataset
from torchgeo.datasets.utils import _list_dict_to_dict_list
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
                # Choose a random item
                sample = random.choice(items)
                bounds = BoundingBox(*sample.bounds)
                if (
                    bounds.maxx - bounds.minx >= self.size[1]
                    or bounds.maxy - bounds.miny >= self.size[0]
                ):
                    sample_maxx, sample_maxy = bounds.maxx, bounds.maxy

                    # Choose random bounding box within that tile
                    random_bounds = get_random_bounding_box(
                        bounds, self.size, self.res
                    )
                    bounds = BoundingBox(
                        random_bounds.minx,
                        min(sample_maxx, random_bounds.maxx),
                        random_bounds.miny,
                        min(sample_maxy, random_bounds.maxy),
                        random_bounds.mint,
                        random_bounds.maxt,
                    )

                batch.append(bounds)

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
                    maxy = min(miny + self.size[0], bounds.maxy)

                    # For each column...
                    for j in range(cols):
                        minx = bounds.minx + j * self.stride[1]
                        maxx = min(minx + self.size[1], bounds.maxx)

                        yield BoundingBox(minx, maxx, miny, maxy, mint, maxt)
            else:
                yield bounds

    def __len__(self) -> int:
        """Return the number of samples over the ROI.

        Returns:
            number of patches that will be sampled
        """
        return self.length


def get_padding_size(sample_shape: Sequence[int], size: int):
    """Get padding size for 2D samples.

    Args:
        sample_shape: list of lengths
        size: the output size

    Returns:
        4-elements tuple
    """
    left = (size - sample_shape[-1]) // 2
    right = size - left - sample_shape[-1]
    top = (size - sample_shape[-2]) // 2
    bottom = size - top - sample_shape[-2]
    return (left, right, top, bottom)


def collate_samples(
    samples: Iterable[dict[Any, Any]], size: int
) -> dict[Any, Any]:
    """Stack a list of samples along a new axis. Samples smaller than the given
    size are padded with 0.

    Useful for forming a mini-batch of samples to pass to DataLoader.

    Args:
        samples: list of samples

    Returns:
        a single sample
    """
    collated: dict[Any, Any] = _list_dict_to_dict_list(samples)
    for key, value in collated.items():
        if isinstance(value[0], torch.Tensor):
            value = [
                F.pad(sample, get_padding_size(sample.shape, size))
                for sample in value
            ]
            collated[key] = torch.stack(value)
    return collated
