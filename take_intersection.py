from torchgeo.datasets import GeoDataset, BoundingBox

def take_intersection(dataset1: GeoDataset, dataset2: GeoDataset) -> GeoDataset:
     '''Given two GeoDatasets, performs the & operation and returns a GeoDataset'''
    datasets = [dataset1, dataset2]

    data_crs = dataset1.crs
    data_res = dataset1.res

def merge_dataset_indices(datasets) -> None:
    i = 0
    ds1, ds2 = datasets
    for hit1 in ds1.index.intersection(ds1.index.bounds, objects=True):
        for hit2 in ds2.index.intersection(hit1.bounds, objects=True):
            box1 = BoundingBox(*hit1.bounds)
            box2 = BoundingBox(*hit2.bounds)
            self.index.insert(i, tuple(box1 & box2)) # what is this line doing
            i += 1

    if i == 0:
        raise RuntimeError("Datasets have no spatiotemporal intersection")

class IntersectionDataset(GeoDataset):
    """Dataset representing the intersection of two GeoDatasets.

    This allows users to do things like:

    * Combine image and target labels and sample from both simultaneously
      (e.g., Landsat and CDL)
    * Combine datasets for multiple image sources for multimodal learning or data fusion
      (e.g., Landsat and Sentinel)
    * Combine image and other raster data (e.g., elevation, temperature, pressure)
      and sample from both simultaneously (e.g., Landsat and Aster Global DEM)

    These combinations require that all queries are present in *both* datasets,
    and can be combined using an :class:`IntersectionDataset`:

    .. code-block:: python

       dataset = landsat & cdl

    .. versionadded:: 0.2
    """

[docs]    def __init__(
        self,
        dataset1: GeoDataset,
        dataset2: GeoDataset,
        collate_fn: Callable[
            [Sequence[dict[str, Any]]], dict[str, Any]
        ] = concat_samples,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """Initialize a new IntersectionDataset instance.

        When computing the intersection between two datasets that both contain model
        inputs (such as images) or model outputs (such as masks), the default behavior
        is to stack the data along the channel dimension. The *collate_fn* parameter
        can be used to change this behavior.

        Args:
            dataset1: the first dataset
            dataset2: the second dataset
            collate_fn: function used to collate samples
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version

        Raises:
            RuntimeError: if datasets have no spatiotemporal intersection
            ValueError: if either dataset is not a :class:`GeoDataset`

        .. versionadded:: 0.4
            The *transforms* parameter.
        """
        super().__init__(transforms)
        self.datasets = [dataset1, dataset2]
        self.collate_fn = collate_fn

        for ds in self.datasets:
            if not isinstance(ds, GeoDataset):
                raise ValueError("IntersectionDataset only supports GeoDatasets")

        self.crs = dataset1.crs
        self.res = dataset1.res

        # Merge dataset indices into a single index
        self._merge_dataset_indices()


    def _merge_dataset_indices(self) -> None:
        """Create a new R-tree out of the individual indices from two datasets."""
        i = 0
        ds1, ds2 = self.datasets
        for hit1 in ds1.index.intersection(ds1.index.bounds, objects=True):
            for hit2 in ds2.index.intersection(hit1.bounds, objects=True):
                box1 = BoundingBox(*hit1.bounds)
                box2 = BoundingBox(*hit2.bounds)
                self.index.insert(i, tuple(box1 & box2))
                i += 1

        if i == 0:
            raise RuntimeError("Datasets have no spatiotemporal intersection")

[docs]    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve image and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of data/labels and metadata at that index

        Raises:
            IndexError: if query is not within bounds of the index
        """
        if not query.intersects(self.bounds):
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        # All datasets are guaranteed to have a valid query
        samples = [ds[query] for ds in self.datasets]

        sample = self.collate_fn(samples)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


[docs]    def __str__(self) -> str:
        """Return the informal string representation of the object.

        Returns:
            informal string representation
        """
        return f"""\
{self.__class__.__name__} Dataset
    type: IntersectionDataset
    bbox: {self.bounds}
    size: {len(self)}"""


    @property
    def crs(self) -> CRS:
        """:term:`coordinate reference system (CRS)` of both datasets.

        Returns:
            The :term:`coordinate reference system (CRS)`.
        """
        return self._crs

    @crs.setter
    def crs(self, new_crs: CRS) -> None:
        """Change the :term:`coordinate reference system (CRS)` of both datasets.

        Args:
            new_crs: New :term:`coordinate reference system (CRS)`.
        """
        self._crs = new_crs
        self.datasets[0].crs = new_crs
        self.datasets[1].crs = new_crs

    @property
    def res(self) -> float:
        """Resolution of both datasets in units of CRS.

        Returns:
            Resolution of both datasets.
        """
        return self._res

    @res.setter
    def res(self, new_res: float) -> None:
        """Change the resolution of both datasets.

        Args:
            new_res: New resolution.
        """
        self._res = new_res
        self.datasets[0].res = new_res
        self.datasets[1].res = new_res


