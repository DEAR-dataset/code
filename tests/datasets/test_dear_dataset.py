import random
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from parameterized import parameterized
from scipy.special import factorial

from dear.datasets.dear_dataset import (
    DRRDEARDataset,
    EnvironmentDEARDataset,
    IndoorOutdoorDEARDataset,
    RT60DEARDataset,
    SNRDEARDataset,
    SpeechDEARDataset,
    StationaryTransientNoiseDEARDataset,
)


def poisson(x):
    return (np.exp(-0.6) * 0.6**x) / factorial(x)


def rt_drr():
    return poisson(random.uniform(0, 2)), np.random.normal(5, 5)


def mock_dear_dataset(base_path: Path, files: int = 128):
    sample_rate = 48_000
    duration = 30
    file_path = base_path / "files"
    file_path.mkdir(exist_ok=False)
    multiclass_columns = [
        "environmentClass",
        *[f"targetSNR_{i}" for i in range(1, 31)],
        *[f"nTargetsActive_{i}" for i in range(1, 31)],
        "indoorOrOutdoor",
        "nTargets",
        "targetRT60",
        "targetDRR",
    ]
    metadata = pd.DataFrame(
        columns=["id", "split", "stationaryOrTransientNoise", *multiclass_columns]
    )
    for i in range(files):
        row = pd.Series()
        file_uuid = uuid.uuid4()
        row["id"] = f"{file_uuid}"
        row["split"] = "development"
        if random.choices([False, True], weights=[10, 2])[0]:
            row["stationaryOrTransientNoise"] = random.choices(
                ["stationary", "transient"], weights=[10, 4]
            )[0]
            for column in multiclass_columns:
                row[column] = ""
        else:
            row["environmentClass"] = random.choices(
                ["Nature", "Transport", "Domestic", "Professional", "Leisure"],
                [20, 20, 16, 10, 2],
            )[0]
            row["stationaryOrTransientNoise"] = ""
            row["indoorOrOutdoor"] = random.choice(["indoor", "outdoor"])
            row["nTargets"] = random.choices([1, 2, 3], [40, 15, 10])[0]
            if row["nTargets"] > 1:
                row["targetRT60"], row["targetDRR"] = "", ""
            else:
                row["targetRT60"], row["targetDRR"] = rt_drr()
            for segment in range(1, 31):
                row[f"nTargetsActive_{segment}"] = random.choice(
                    range(row["nTargets"] + 1)
                )
                row[f"targetSNR_{segment}"] = np.random.normal(0, 15)
        torchaudio.save(
            file_path / f"{file_uuid}.wav",
            torch.rand((1, sample_rate * duration - 10)),
            sample_rate=sample_rate,
        )
        metadata.loc[len(metadata), :] = row
    metadata.to_csv(base_path / "DEAR_metadata.csv")
    return metadata


class TestDearDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.metadata_df = mock_dear_dataset(cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    @parameterized.expand(
        (
            ("EnvironmentDEARDataset", EnvironmentDEARDataset, {}),
            (
                "StationaryTransientNoiseDEARDataset",
                StationaryTransientNoiseDEARDataset,
                {},
            ),
            ("RT60DEARDataset", RT60DEARDataset, {}),
            ("IndoorOutdoorDEARDataset", IndoorOutdoorDEARDataset, {}),
            ("DRRDEARDataset", DRRDEARDataset, {}),
            ("SNRDEARDataset", SNRDEARDataset, {}),
            ("SpeechPresentDEARDataset", SpeechDEARDataset, {"speech_present": True}),
            ("SpeakersActiveDEARDataset", SpeechDEARDataset, {"speech_present": False}),
        )
    )
    def test_dear_evaluation_tasks(self, name, dataclass, kwargs):
        dataset = dataclass(base_path=self.temp_dir, **kwargs)
        for entry, label in dataset:
            self.assertTrue(isinstance(entry, torch.Tensor))


if __name__ == "__main__":
    unittest.main()
