import shutil
import tempfile
import unittest
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from dear.datasets.dataset_factory import DatasetFactory
from dear.datasets.dear_dataset import DEARBaseDataset
from tests.datasets.test_dear_dataset import mock_dear_dataset


@DatasetFactory.register()
class DummyDEARDataset(DEARBaseDataset):
    def process_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        return metadata

    def get_label(self, row) -> List[Tuple[int, float]]:
        pass


class DatasetFactoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.factory = DatasetFactory()
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.metadata_df = mock_dear_dataset(cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def test_register_and_create(self):
        assert len(self.factory.registry) >= 1
        dummy_object = self.factory.create("DummyDEARDataset", base_path=self.temp_dir)
        self.assertIsInstance(
            dummy_object,
            DummyDEARDataset,
        )

    def test_key_error(self):
        self.assertRaises(
            KeyError,
            self.factory.create,
            class_name="ClassNotInRegistry",
        )


if __name__ == "__main__":
    unittest.main()
