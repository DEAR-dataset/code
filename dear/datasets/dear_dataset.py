import logging
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from pandas import DataFrame

from ..utils.split_mixin import DataframeSplitMixin, DatasetType
from .dataset_factory import DatasetFactory


class TargetVariableType(Enum):
    """Enum for the different types of target variables."""

    CONTINUOUS = "CONTINUOUS"
    DISCRETE = "DISCRETE"


class ResampleMethod(Enum):
    SINC_INTERP_HANN = "sinc_interp_hann"
    SINC_INTERP_KAISER = "sinc_interp_kaiser"


class DEARBaseDataset(torch.utils.data.Dataset, DataframeSplitMixin, ABC):
    def __init__(
        self,
        base_path: Path = Path("/data/DEAR"),
        split: DatasetType = DatasetType.TRAIN,
        resample_method: ResampleMethod = ResampleMethod.SINC_INTERP_HANN,
        transform: Optional[Callable] = None,
        output_sample_rate: Optional[int] = 44_100,
        apply_windowing: bool = False,
        target_variable_type: TargetVariableType = TargetVariableType.DISCRETE,
    ):
        self.logger = logging.getLogger("DEARDataset")
        if isinstance(base_path, str):
            base_path = Path(base_path)
        metadata_df = pd.read_csv(base_path / "DEAR_metadata.csv")
        self.metadata = self.process_metadata(metadata_df)
        self.split(self.metadata)
        self.change_dataset_type(split)
        self.base_path = base_path
        self.output_sample_rate = output_sample_rate
        self.resample_method = resample_method
        self.apply_windowing = apply_windowing
        self.transform = transform
        self.target_variable_type = target_variable_type

    def split(
        self,
        dataframe: DataFrame,
        validation_size: int | float = -1.0,
        random_seed: Optional[int] = 42,
        dataset_subsample: Optional[int] = None,
    ):
        if validation_size != -1.0:
            raise RuntimeWarning(
                "Split is defined in the metadata of the DEAR dataset, the parameter"
                "`validation_size` thus has no effect"
            )
        if dataset_subsample is not None:
            raise NotImplementedError()
        full_df = dataframe
        train_df = full_df[full_df.split == "development"]
        test_df = full_df[full_df.split == "test"]
        self.split_df = {
            DatasetType.FULL: full_df,
            DatasetType.TRAIN: train_df,
            DatasetType.VALIDATION: test_df,
        }

    def change_dataset_type(self, dataset_type: DatasetType) -> DatasetType:
        dataset_type = DatasetType(dataset_type)
        if dataset_type not in self.split_df:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        self.metadata = self.get_split(dataset_type)
        return dataset_type

    @abstractmethod
    def process_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Method not implemented in the baseclass.")

    @abstractmethod
    def get_label(self, row) -> List[Tuple[int, float]]:
        raise NotImplementedError("Method not implemented in the baseclass.")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(index, int):
            raise TypeError(f"Index must be of type int but got {type(index)}")
        audio_entry = self.metadata.iloc[index]
        waveform, sample_rate = torchaudio.load(
            self.base_path / "files" / f"{audio_entry['id']}.wav"
        )
        if (
            self.output_sample_rate is not None
            and self.output_sample_rate != sample_rate
        ):
            resample_transform = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.output_sample_rate,
                resampling_method=self.resample_method.value,
            )
            waveform = resample_transform(waveform)
        segments, labels = self.slice_item_into_segments(
            data_tensor=waveform,
            label_list=self.get_label(audio_entry),
        )
        if self.apply_windowing:
            segments = self.apply_window_function(segments)
        if self.transform:
            segments = torch.stack(
                [self.transform(x_i) for x_i in torch.unbind(segments, dim=0)],
                dim=0,
            )
        return segments, labels

    def slice_item_into_segments(
        self,
        data_tensor: torch.Tensor,
        label_list: List[Tuple[int, float]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Slices an element and associated labels into segments.

        Takes in the label list for one file of the DEAR database and
        slices them in segments with associated times.

        :param data_tensor: Tensor containing the 30s of DEAR generated audio
        :param label_list: List with tuples with semantics (start_s, label)
        :return: Tensor with audio segments and a tensor with labels
        """
        segments = []
        labels = []
        for i, (start_s, label) in enumerate(label_list):
            start_index = int(start_s * self.output_sample_rate)
            if i + 1 < len(label_list):
                next_start = label_list[i + 1][0]
                end_index = int(next_start * self.output_sample_rate)
                if end_index <= data_tensor.shape[1]:
                    labels.append(label)
                    segment = data_tensor[:, start_index:end_index]
                    segments.append(segment)
                else:
                    self.logger.warning("Found labels outside of tensor size")
                    break
            else:
                labels.append(label)
                segment = data_tensor[:, start_index:]
                expected_length = self.output_sample_rate
                segment_length = segment.shape[-1]
                if segment_length < expected_length:
                    segment = torch.nn.functional.pad(
                        segment, (0, expected_length - segment_length)
                    )
                segments.append(segment)
        segments = torch.stack(segments, dim=0)
        labels = torch.Tensor(labels)
        labels = torch.unsqueeze(labels, -1)
        return segments, labels

    def apply_window_function(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Apply window function to the signals.

        Note: this method only works BEFORE transforming to frequency domain
        :param signal: Signal or Signals in the time domain to apply windowing
        :return: Windowed signal
        """
        window = torch.hann_window(signal.shape[-1])
        window = window[None, :].repeat(*signal.shape[:-1], 1)
        signal = signal * window
        return signal


@DatasetFactory.register()
class EnvironmentDEARDataset(DEARBaseDataset):
    def process_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        metadata = metadata[metadata["environmentClass"].notna()]
        metadata["environmentClassLabel"], _ = pd.factorize(
            metadata["environmentClass"]
        )
        return metadata

    def get_label(self, row) -> List[Tuple[int, float]]:
        return [
            (0, int(row["environmentClassLabel"])),
        ]


@DatasetFactory.register()
class IndoorOutdoorDEARDataset(DEARBaseDataset):
    def process_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        metadata = metadata[metadata["indoorOrOutdoor"].notna()]
        metadata["indoorOrOutdoorLabel"], _ = pd.factorize(metadata["indoorOrOutdoor"])
        return metadata

    def get_label(self, row) -> List[Tuple[int, float]]:
        return [
            (0, int(row["indoorOrOutdoorLabel"])),
        ]


@DatasetFactory.register()
class StationaryTransientNoiseDEARDataset(DEARBaseDataset):
    def process_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        metadata = metadata[metadata["stationaryOrTransientNoise"].notna()]
        metadata["stationaryOrTransientNoise"], _ = pd.factorize(
            metadata["stationaryOrTransientNoise"]
        )
        return metadata

    def get_label(self, row) -> List[Tuple[int, float]]:
        return [
            (0, int(row["stationaryOrTransientNoise"])),
        ]


class DEARSegmentDataset(DEARBaseDataset, ABC):
    def process_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Remove all rows containing no targets.

        :param metadata: DataFrame containing metadata for the DEAR dataset
        :return: Cleaned DataFrame with rows containing no segment targets removed
        """
        if any(metadata.nTargets.isna()):
            self.logger.warning(
                f"Expected no NaN targets but got:\n"
                f"{metadata[metadata.nTargets.isna()]}"
            )
        return metadata

    def get_segment_labels(
        self,
        row: pd.Series,
        target_name: str,
    ) -> Tuple:
        data = row[[f"{target_name}{i}" for i in range(1, 31)]]
        data.replace("", np.nan)
        return tuple(zip(range(0, 30, 1), data))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        segments, labels = super().__getitem__(index)
        return (
            segments[~torch.any(labels.isnan(), dim=1)],
            labels[~torch.any(labels.isnan(), dim=1)],
        )


@DatasetFactory.register()
class SNRDEARDataset(DEARSegmentDataset):
    def get_label(self, row: pd.Series) -> List:
        """
        Get the SNR label for the current audio file.

        For each audio file we have 30 1s segments with an SNR target to regress to
        :param row: the row of the loaded dataset
        :return: list of tuples with semantics (start_of_label_s, label value)
        """
        return list(self.get_segment_labels(row, target_name="targetSNR_"))


@DatasetFactory.register()
class SpeechDEARDataset(DEARSegmentDataset):
    def __init__(self, speech_present: bool = False, **kwargs):
        """
        Instantiate speech present dataset.

        This class either provides classification targets/labels if the flag
        speech_present is set, or a regression target on the number of active speakers
        in a segment if the flag is not set.
        :param speech_present: If it's a classification target (speech/no speech present
            ) or if it's a regression target on the number of active speakers (0-3)
        """
        self.speech_present = speech_present
        super().__init__(**kwargs)

    def get_label(self, row) -> Tuple:
        """
        Get the speech present label for the current audio file.

        :param row: the row of the loaded dataset
        :return: list of tuples with semantics (start_of_label_s, label value)
        """
        segment_labels = self.get_segment_labels(row, target_name="nTargetsActive_")
        if self.speech_present:
            segment_labels = tuple(
                (index, int(label > 0)) for index, label in segment_labels
            )
        return segment_labels


class DEARActiveSegmentDataset(DEARSegmentDataset, ABC):
    def process_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Only keep rows containing just one target.

        :param metadata: DataFrame containing metadata for the DEAR dataset
        :return: Cleaned DataFrame with rows containing only one active target
        """
        metadata = metadata[metadata.nTargets == 1]
        return metadata

    def get_active_segment_labels(
        self,
        row: pd.Series,
        label: float,
    ) -> Tuple:
        data = torch.tensor([label for _ in range(1, 31)])
        activity = torch.tensor(
            np.array([row[f"nTargetsActive_{i}"] < 1 for i in range(1, 31)])
        )
        data[activity] = torch.nan
        return tuple(zip(range(0, 30, 1), data))


@DatasetFactory.register()
class DRRDEARDataset(DEARActiveSegmentDataset):
    def get_label(self, row: pd.Series) -> List:
        """
        Return a DRR regression target for the current row.

        :param row: the row of the loaded dataset
        :return: tuple of tuples with semantics
            (start_of_label_s, DRR for target)
        """
        labels = self.get_active_segment_labels(row, label=float(row["targetDRR"]))
        return list(labels)


@DatasetFactory.register()
class RT60DEARDataset(DEARActiveSegmentDataset):
    def get_label(self, row: pd.Series) -> List:
        """
        Return a RT60 regression target for the current row.

        :param row: the row of the loaded dataset
        :return: tuple of tuples with semantics
            (start_of_label_s, RT60 for target)
        """
        labels = self.get_active_segment_labels(
            row, label=np.log(float(row["targetRT60"]))
        )
        return list(labels)
