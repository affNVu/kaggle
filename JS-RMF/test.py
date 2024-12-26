import polars as pl
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler


from sklearn.model_selection import train_test_split, cross_val_score
from functools import partial


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
    
def objective(trial, x_train, x_test, y_train, y_test):
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dvalid = xgb.DMatrix(x_test, label=y_test)

    param = {
        "verbosity": 0,
        "objective": "reg:squarederror",
        "device": "cuda",
        # use exact for small dataset.
        "tree_method": "gpu_hist",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        # 
        "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=50),
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    bst = xgb.XGBRegressor(**param)
    fmse = make_scorer(mean_squared_error)
    mean_cv = np.mean(cross_val_score(bst, x_train, y_train, cv=5, scoring=fmse))
    return mean_cv

def get_dataset(
    cols: list[str],
    input_paths: str = 'inputs/train.parquet/*/*.parquet',
    fraction: float = 0.1,
    head:int =None,
) -> pd.DataFrame:
    # data handling
    lf = pl.scan_parquet(input_paths)
    head = lf.select(pl.len()).collect()['len'][0] if head is None else head
    df = lf.head(head).select(cols).collect()
    df = df.sample(fraction=fraction).to_pandas()
    df = reduce_mem_usage(df)
    return df


def train_single_model(X: pl.DataFrame, y: pl.DataFrame, model_name: str, n_trials: int = 100, timeout: int = 600, cv_frac: float = .1):
    _, X_cv, _, y_cv = train_test_split(X, y, test_size=cv_frac)
    print('CV Set: ', X_cv.shape)
    X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X_cv, y_cv, test_size=0.3)
    _objective = partial(objective, x_train=X_train_cv, y_train=y_train_cv, x_test=X_test_cv, y_test=y_test_cv)
    # hyper tune
    study = optuna.create_study(direction="minimize", sampler=TPESampler())
    study.optimize(_objective, n_trials=n_trials, timeout=timeout)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    params = {
        "verbosity": 2,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cuda",
        **trial.params,
    }
    print(params)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.05)
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    model.save_model(f'{model_name}.json')
    del X_train
    del X_valid
    return model

input_path = 'inputs/train.parquet/*/*.parquet'
lf = pl.scan_parquet(input_path)
#
columns = lf.columns
features_cols = [x for x in columns if 'feature' in x]
responder_cols = [x for x in columns if 'responder' in x]
target_col = 'responder_6'

models = []
num_models = 7
# for i in range(num_models):
#     model_name = f'xgb_{i}'
#     #
#     df = get_dataset(cols=features_cols+[target_col],head=int(1e7),fraction=0.95)
#     X = df[features_cols]
#     y = df[target_col]
#     print(X.shape, y.shape)
#     model = train_single_model(X, y, model_name, n_trials=30, cv_frac=0.1)
#     models.append(model)


# load saved model
num_models = 7
models = [xgb.XGBRegressor() for i in range(num_models)]
for i, model in enumerate(models):
    model.load_model(f'xgb_{i}.json')
    model.set_params(device='cuda')

# batch predicting

n = 47127338
batch_size = int(1e6)
num_batch = n // batch_size
lf = pl.scan_parquet(input_path)
preds = []
y_trues = []
for i in range(num_batch+1):
    print(f'Batch: {i}/{num_batch}')
    rows = list(range(i*batch_size, min((i+1)*batch_size, n), 1))
    df = lf.select(pl.all().gather(rows)).collect()
    df = reduce_mem_usage(df.to_pandas())
    _X_train = df[features_cols]
    y_train = df[target_col]
    X_train = [model.predict(_X_train) for model in models]
    y_trues.append(y_train)
    X_train = np.vstack(X_train).T
    preds.append(X_train)
    # booster = train_single_model(X_train, y_train, 'booster', n_trials=20)
X_train_booster = np.row_stack(preds)
pl.DataFrame(X_train_booster).write_parquet('X_train_booster.parquet')
_y_trues = [x.to_numpy().reshape(-1,1) for x in y_trues]
y_train_booster = np.row_stack(_y_trues)
pl.DataFrame(y_train_booster).write_parquet('y_train_booster.parquet')
