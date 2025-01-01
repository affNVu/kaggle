import polars as pl
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, make_scorer

input_path = 'inputs/train.parquet/*/*.parquet'
lf = pl.scan_parquet(input_path)
#
columns = lf.columns
features_cols = [x for x in columns if 'feature' in x]
responder_cols = [x for x in columns if 'responder' in x]
target_col = 'responder_6'
# dates 
dates = sorted(lf.select(['date_id']).unique().collect()['date_id'].to_list())
time_steps = sorted(lf.select(['time_id']).unique().collect()['time_id'].to_list())

class PurgedGroupTimeSeriesSplit:
    def __init__(
        self,
        train_days: int,
        test_days: int,
        gap: int = 0,
        max_n_splits: int = None,
        min_train_days: int = 1,
    ):
        self.train_days = train_days
        self.gap = gap
        self.test_days = test_days
        self.max_n_splits = max_n_splits
        self.min_train_days = min_train_days
        if train_days <= 0 or test_days <= 0 or min_train_days <= 0 or gap < 0:
            raise ValueError("Non positive input for days")
        if max_n_splits is not None and max_n_splits <= 0:
            raise ValueError("Non positive input for max_n_splits")

    def split(self, X, y):
        if not all(y.index == X.index):
            raise ValueError(
                "Indexes of feature data and response data are not the same"
            )

        X, y = indexable(X, y)
        test_days = self.test_days
        gap = self.gap
        train_days = self.train_days
        min_train_days = self.min_train_days
        max_n_splits = self.max_n_splits
        unique_groups = sorted(list(y.reset_index()["date_id"].unique()))
        n_groups = _num_samples(unique_groups)
        if min_train_days + gap + test_days > n_groups:
            raise ValueError(
                "Number of days in data is less than the minimum days need for a fold"
            )

        group_test_starts = list(range(n_groups - test_days, 0, -test_days))
        if max_n_splits is not None:
            group_test_starts = group_test_starts[:max_n_splits]
        group_test_starts = [
            g for g in group_test_starts if g - gap - min_train_days >= 0
        ]

        for group_test_start in group_test_starts:
            group_train_end = group_test_start - gap
            group_train_start = max(group_train_end - train_days, 0)
            train_dates = unique_groups[group_train_start:group_train_end]
            test_dates = unique_groups[group_test_start : group_test_start + test_days]
            test_index_array = X[X['date_id'].isin(test_dates)].index
            train_index_array = X[X['date_id'].isin(train_dates)].index
            X_test, y_test = X.iloc[test_index_array], y.iloc[test_index_array]
            X_train, y_train = X.iloc[train_index_array], y.iloc[train_index_array]
            out = [X_train, X_test, y_train, y_test]
            out = [x.drop(columns='date_id') for x in out]
            yield out


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def get_dataset_by_date_id(start: int, end: int, cols: list[str] = features_cols + [target_col]):
    lf = pl.scan_parquet(input_path)
    filter = ((pl.col('date_id') >= start) & (pl.col('date_id') <= end))
    df = lf.filter(filter).select(cols).collect()
    return reduce_mem_usage(df.to_pandas())

# 
num_dates_per_training = 100

def foo(
    num_dates: int
):
    