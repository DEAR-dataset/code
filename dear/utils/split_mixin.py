import abc
from enum import Enum
from typing import Optional

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split


class DatasetType(Enum):
    """Enum for the different dataset splits."""

    TRAIN = "TRAIN"
    TRAIN_SUBSAMPLED = "TRAIN_SUBSAMPLED"
    VALIDATION = "VALIDATION"
    FULL = "FULL"


class DataframeSplitMixin:
    split_df = None

    def split(
        self,
        dataframe: DataFrame,
        validation_size: int | float,
        random_seed: Optional[int],
        dataset_subsample: Optional[int] = None,
    ):
        """
        Split the data frame into train and validation dataframe.

        :param dataframe: Dataframe to derive the splits from
        :param validation_size: Validation size in number of samples [int] or as a
            percentage of the original dataframe [float]
        :param dataset_subsample: How many samples to take from the train dataset
        :param random_seed: Seed for the initialisation of a random generator
        """
        full_df = dataframe
        if validation_size > 0:
            train_df, validation_df = train_test_split(
                dataframe,
                test_size=validation_size,
                random_state=random_seed,
            )
        else:
            train_df = full_df
            validation_df = pd.DataFrame(columns=full_df.columns)
        if dataset_subsample is not None:
            train_sub_df = train_df.sample(
                n=dataset_subsample,
                random_state=random_seed,
            )
        else:
            train_sub_df = pd.DataFrame(columns=full_df.columns)
        self.split_df = {
            DatasetType.FULL: full_df,
            DatasetType.TRAIN: train_df,
            DatasetType.TRAIN_SUBSAMPLED: train_sub_df,
            DatasetType.VALIDATION: validation_df,
        }

    def get_split(self, split: DatasetType) -> DataFrame:
        if self.split_df is None:
            raise NameError("No splits have been made, please call `split` first")
        return self.split_df[split]

    @abc.abstractmethod
    def change_dataset_type(self, dataset_type: DatasetType) -> DatasetType:
        raise NotImplementedError("Not implemented in base mixin")
