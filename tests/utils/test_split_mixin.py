import unittest
from typing import Optional

import numpy as np
import pandas as pd
import torch

from dear.utils.split_mixin import DataframeSplitMixin, DatasetType


class DummyDataframeDataset(torch.utils.data.Dataset, DataframeSplitMixin):
    def __init__(
        self,
        *args,
        dataframe_size: int,
        validation_size: int | float,
        random_seed: int = 42,
        dataset_subsample: Optional[int] = None,
        **kwargs
    ):
        self.df = pd.DataFrame(
            np.random.randint(0, 100, size=(dataframe_size, 3)), columns=list("ABC")
        )
        super().__init__(*args, **kwargs)
        self.split(
            self.df,
            validation_size=validation_size,
            dataset_subsample=dataset_subsample,
            random_seed=random_seed,
        )
        self.current_split = self.change_dataset_type(DatasetType.TRAIN)

    def __len__(self) -> int:
        return len(self.df)

    def change_dataset_type(self, dataset_type: DatasetType) -> DatasetType:
        self.df = self.get_split(dataset_type)
        return dataset_type


class TestDataframeSplitMixin(unittest.TestCase):
    def test_split_int(self):
        df_size = 100
        val_size = 30
        dummy = DummyDataframeDataset(dataframe_size=df_size, validation_size=val_size)
        self.assertEqual(df_size - val_size, len(dummy))

    def test_split_subsample_size(self):
        df_size = 100
        subsample_size = 20
        dummy = DummyDataframeDataset(
            dataframe_size=df_size,
            validation_size=0,
            dataset_subsample=subsample_size,
        )
        dstype = DatasetType.TRAIN_SUBSAMPLED
        self.assertEqual(dummy.change_dataset_type(dstype), dstype)
        self.assertEqual(subsample_size, len(dummy))

    def test_split_float(self):
        df_size = 100
        val_size = 0.3
        dummy = DummyDataframeDataset(dataframe_size=df_size, validation_size=val_size)
        self.assertEqual(df_size - val_size * df_size, len(dummy))


if __name__ == "__main__":
    unittest.main()
