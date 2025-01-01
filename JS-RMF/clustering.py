import polars as pl
import faiss
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor

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

def get_dataset(
    cols: list[str],
    input_paths: str = 'inputs/train.parquet/*/*.parquet',
    fraction: float = 0.1,
    head:int =None,
):
    # data handling
    lf = pl.scan_parquet(input_paths)
    head = lf.select(pl.len()).collect()['len'][0] if head is None else head
    df = lf.head(head).select(cols).collect()
    df = df.sample(fraction=fraction).to_pandas()
    df = reduce_mem_usage(df)
    return df
    

input_path = 'inputs/train.parquet/*/*.parquet'
lf = pl.scan_parquet(input_path)
#
columns = lf.collect_schema().names()
features_cols = [x for x in columns if 'feature' in x]
responder_cols = [x for x in columns if 'responder' in x]
target_col = 'responder_6'

df = get_dataset(cols=features_cols+[target_col],fraction=0.1)

_X = df[features_cols]
y = df[target_col].to_numpy()
print(_X.shape)

preproc = SimpleImputer()
X = preproc.fit_transform(_X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# knn 
neigh = KNeighborsRegressor(n_neighbors=10)
neigh.fit(X_train, y_train)
y_pred_neigh = neigh.predict(X_test).astype(float)
print(f'kNN: {r2_score(y_true=y_test, y_pred=y_pred_neigh)}')

# # xgb
# tree = xgb.XGBRegressor()
# tree.fit(X_train, y_train)
# y_pred_tree = tree.predict(X_test)
# print(f'xgb: {r2_score(y_true=y_test, y_pred=y_pred_tree)}')

# mlp
mlp = MLPRegressor()
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
print(f'mlp: {r2_score(y_true=y_test, y_pred=y_pred_mlp)}')
#

class KnnApprox:
    def __init__(self, ncentroids:int, niters: int = 20, nredos: int = 5):
        self.ncentroids = ncentroids
        self.niters = niters
        self.kmeans = None
        self.nredo = nredos
        self.preprocs = None

    def fit(self, X, y):
        verbose = True
        d = X.shape[1]
        kmeans = faiss.Kmeans(d, self.ncentroids, niter=self.niters, nredo=self.nredo, verbose=verbose)
        kmeans.train(X)
        self.kmeans = kmeans
        # find centroid of inputs
        _, I = kmeans.index.search(X, 1)
        Xs = {k:[] for k in range(self.ncentroids)}
        ys = {k:[] for k in range(self.ncentroids)}
        models = {} 
        preprocs = {}
        for i, x in enumerate(I.flatten()):
            Xs[x].append(X[i])
            ys[x].append(y[i])
        # 
        for i in range(self.ncentroids):
            X_local = np.row_stack(Xs[i])
            y_local = np.row_stack(ys[i])
            local = Ridge(alpha=0.1)
            preproc = StandardScaler()
            X_local = preproc.fit_transform(X_local)
            local.fit(X_local,y_local)
            models[i] = local
            preprocs[i] = preproc
        self.models = models
        self.preprocs = preprocs

    def predict(self, X):
        # look for nearest centroid
        _, I = self.kmeans.index.search(X, 1)
        y_preds = []
        for i, x in enumerate(I.flatten()):
            _input = X[i].reshape(1,-1)
            inp = self.preprocs[x].transform(_input)
            model = self.models[x]
            y_pred = model.predict(inp)
            y_preds.append(y_pred)
        return np.row_stack(y_preds)


    def score(self, X, y):
        y_preds = self.predict(X)
        return r2_score(y_pred=y_preds,y_true=y)

knn = KnnApprox(ncentroids=10000,niters=50)
knn.fit(X_train, y_train)
y_test = y_test.reshape(-1,1)
knn.score(X_test, y_test.reshape(-1,1))
