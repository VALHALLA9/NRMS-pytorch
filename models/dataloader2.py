from dataclasses import dataclass, field
import polars as pl
import numpy as np
import torch
from torch.utils.data import Dataset

from utils._articles_behaviors import map_list_article_id_to_value
from utils._python import (
    repeat_by_list_values_from_matrix,
    create_lookup_objects,
)

from utils._constants import (

    DEFAULT_USER_COL,
    DEFAULT_IMPRESSION_TIMESTAMP_COL,
    DEFAULT_ARTICLE_PUBLISHED_TIMESTAMP_COL,
    

    DEFAULT_ARTICLE_ID_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    

    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_LABELS_COL,
)


@dataclass
class NewsrecDataLoader(Dataset):
    """
    A DataLoader for news recommendation.
    """

    behaviors: pl.DataFrame
    history_column: str
    article_dict: dict[int, any]
    unknown_representation: str
    eval_mode: bool = False
    batch_size: int = 32
    inview_col: str = DEFAULT_INVIEW_ARTICLES_COL
    labels_col: str = DEFAULT_LABELS_COL
    user_col: str = DEFAULT_USER_COL
    use_time_features: bool = True
    kwargs: field(default_factory=dict) = None

    def __post_init__(self):
        """
        Post-initialization method. Loads the data and sets additional attributes.
        """
        self.lookup_article_index, self.lookup_article_matrix = create_lookup_objects(
            self.article_dict, unknown_representation=self.unknown_representation
        )
        self.unknown_index = [0]
        self.X, self.y = self.load_data()
        if self.kwargs is not None:
            self.set_kwargs(self.kwargs)

    def __len__(self) -> int:
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):

        raise ValueError("Function '__getitem__' needs to be implemented.")

    def load_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Load data with or without labels"""
        if self.eval_mode and self.labels_col not in self.behaviors.columns:
            # For test set without labels
            X = self.behaviors.with_columns(
                pl.col(self.inview_col).list.len().alias("n_samples")
            )
            # Create dummy labels
            n_samples = X["n_samples"].sum()
            y = pl.DataFrame({"dummy_labels": [[0.0] * n for n in X["n_samples"]]})
            return X, y
        else:
            # Original training/validation mode
            X = self.behaviors.drop(self.labels_col).with_columns(
                pl.col(self.inview_col).list.len().alias("n_samples")
            )
            y = self.behaviors[self.labels_col]
            return X, y

    def set_kwargs(self, kwargs: dict):
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class NRMSDataLoader(NewsrecDataLoader):

    use_time_features: bool = False
    history_size: int = 50  
    
    def _get_time_features(self, article_ids, current_time):
        """计算时间差特征"""
        time_deltas = []
        for aid in article_ids:
            if aid in self.article_times:
                delta = (current_time - self.article_times[aid]).total_seconds() / 3600
                time_deltas.append(delta)
            else:
                time_deltas.append(0.0)
        return np.array(time_deltas, dtype=np.float32)
    
    def __getitem__(self, idx):

        his_input_title, pred_input_title = super().__getitem__(idx)
        
        if not self.use_time_features:
            return (his_input_title, pred_input_title)
            
        # time
        current_time = self.behaviors[DEFAULT_IMPRESSION_TIMESTAMP_COL].iloc[idx]
        his_time_delta = self._get_time_features(
            self.behaviors[self.history_column].iloc[idx],
            current_time
        )
        pred_time_delta = self._get_time_features(
            self.behaviors[DEFAULT_INVIEW_ARTICLES_COL].iloc[idx],
            current_time
        )
        
        return (his_input_title, pred_input_title, his_time_delta, pred_time_delta)


    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.pipe(
            map_list_article_id_to_value,
            behaviors_column=self.history_column,
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        ).pipe(
            map_list_article_id_to_value,
            behaviors_column=self.inview_col,
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        )

    def __getitem__(self, idx) -> tuple[tuple[np.ndarray], np.ndarray]:
            batch_X = self.X[idx * self.batch_size : (idx + 1) * self.batch_size].pipe(
                self.transform
            )
            batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
            
            if self.eval_mode:
                repeats = np.array(batch_X["n_samples"])
                
                if "dummy_labels" in batch_y.columns:
                    batch_y = np.concatenate(batch_y["dummy_labels"].to_list())
                else:
                    batch_y = np.array(batch_y.explode().to_list()).reshape(-1, 1)


                history_data = batch_X[self.history_column].to_list()
                for i in range(len(history_data)):
                    if len(history_data[i]) < self.history_size:  
                        history_data[i].extend([0] * (self.history_size - len(history_data[i])))
                    elif len(history_data[i]) > self.history_size:
                        history_data[i] = history_data[i][:self.history_size]
                    
                his_input_title = repeat_by_list_values_from_matrix(
                    history_data,
                    matrix=self.lookup_article_matrix,
                    repeats=repeats,
                )
                pred_input_title = self.lookup_article_matrix[
                    batch_X[self.inview_col].explode().to_list()
                ]
            else:
                batch_y = np.array(batch_y.to_list())
                his_input_title = self.lookup_article_matrix[
                    batch_X[self.history_column].to_list()
                ]
                pred_input_title = self.lookup_article_matrix[
                    batch_X[self.inview_col].to_list()
                ]
                pred_input_title = np.squeeze(pred_input_title, axis=2)
                
            his_input_title = np.squeeze(his_input_title, axis=2)
            return (his_input_title, pred_input_title), batch_y.reshape(-1, 1)


@dataclass
class NRMSDataLoaderPretransform(NewsrecDataLoader):
    """
    In the __post_init__ pre-transform the entire DataFrame. This is useful for
    when data can fit in memory, as it will be much faster once training.
    Note, it might not be as scaleable.
    """

    def __post_init__(self):
        super().__post_init__()
        self.X = self.X.pipe(
            map_list_article_id_to_value,
            behaviors_column=self.history_column,
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        ).pipe(
            map_list_article_id_to_value,
            behaviors_column=self.inview_col,
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        )

    def __getitem__(self, idx) -> tuple[tuple[np.ndarray], np.ndarray]:
        """
        his_input_title:    (samples, history_size, document_dimension)
        pred_input_title:   (samples, npratio, document_dimension)
        batch_y:            (samples, npratio)
        """
        batch_X = self.X[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        if self.eval_mode:
            repeats = np.array(batch_X["n_samples"])
            batch_y = np.array(batch_y.explode().to_list()).reshape(-1, 1)
            his_input_title = repeat_by_list_values_from_matrix(
                batch_X[self.history_column].to_list(),
                matrix=self.lookup_article_matrix,
                repeats=repeats,
            )
            pred_input_title = self.lookup_article_matrix[
                batch_X[self.inview_col].explode().to_list()
            ]
        else:
            batch_y = np.array(batch_y.to_list())
            his_input_title = self.lookup_article_matrix[
                batch_X[self.history_column].to_list()
            ]
            pred_input_title = self.lookup_article_matrix[
                batch_X[self.inview_col].to_list()
            ]
            pred_input_title = np.squeeze(pred_input_title, axis=2)

        his_input_title = np.squeeze(his_input_title, axis=2)
        return (his_input_title, pred_input_title), batch_y

