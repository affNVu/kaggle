# import xgboost as xgb
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
import polars as pl
import torch.optim as optim
import polars as pl
import numpy as np
from sklearn import preprocessing, linear_model, model_selection, pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import math

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


def make_prediction(preproc, dae, reg, X):
    with torch.no_grad():
        X_trf = preproc.transform(X)
        X_trf = torch.from_numpy(X_trf)
        X_embeded = dae.encode(X_trf)
        y_pred = reg(X_embeded)
    return y_pred


def score_predictions(preproc, dae, reg, X, y):
    y_pred = make_prediction(preproc=preproc, dae=dae, reg=reg, X=X)
    y_pred = y_pred.numpy()
    y = y.numpy()
    return r2_score(y_pred=y_pred, y_true=y)


def find_highest_gradient(model):
    max_grad = None
    for param in model.parameters():
        if param.grad is not None:
            grad_values = param.grad.data.flatten()
            max_grad_param = grad_values.max()
            if max_grad is None or max_grad_param > max_grad:
                max_grad = max_grad_param
    return max_grad

def calc_loss_batch(input_batch, target_batch, model, device, loss_fn):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    y_pred = model(input_batch)
    loss = loss_fn(y_pred, target_batch)
    return loss

def calc_loss_loader(data_loader, model, device, loss_fn, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device, loss_fn=loss_fn)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter, loss_fn):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter, loss_fn=loss_fn)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter, loss_fn=loss_fn)
    model.train()
    return train_loss, val_loss


def train_model(model, train_loader, test_loader, optimizer, device,
    n_epochs, eval_freq, eval_iter, loss_fn,
    warmup_steps, early_stopper, initial_lr=3e-05, min_lr=1e-6):
    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    global_step = -1

    # Retrieve the maximum learning rate from the optimizer
    peak_lr = optimizer.param_groups[0]["lr"]

    # Calculate the total number of iterations in the training process
    total_training_steps = len(train_loader) * n_epochs

    # Calculate the learning rate increment during the warmup phase
    lr_increment = (peak_lr - initial_lr) / warmup_steps


    for epoch in range(n_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            # Adjust the learning rate based on the current phase (warmup or cosine annealing)
            if global_step < warmup_steps:
                # Linear warmup
                lr = initial_lr + global_step * lr_increment  
            else:
                # Cosine annealing after warmup
                progress = ((global_step - warmup_steps) / 
                            (total_training_steps - warmup_steps))
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            # Apply the calculated learning rate to the optimizer
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(optimizer.param_groups[0]["lr"])

                # Calculate and backpropagate the loss

            loss = calc_loss_batch(input_batch, target_batch, model=model, loss_fn=loss_fn, device='cpu')
            loss.backward()

            # Apply gradient clipping after the warmup phase to avoid exploding gradients

            if global_step > warmup_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
                
            optimizer.step()
            # Periodically evaluate the model on the training and validation sets

            # if global_step % eval_freq == 0:
            #     train_loss, val_loss = evaluate_model(
            #         model, train_loader, test_loader,
            #         'cpu', 1, loss_fn
            #     )
            #     train_losses.append(train_loss)
            #     val_losses.append(val_loss)
            #     # Print the current losses
                

        train_loss, val_loss = evaluate_model(
            model=model, train_loader=train_loader, val_loader=test_loader, device='cpu', loss_fn=loss_fn, eval_iter=None)
        print(f"Ep {epoch+1} (Iter {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, "
                        f"Val loss {val_loss:.3f}"
                )
        if early_stopper.early_stop(val_loss):            
            print('Plateu, Breaking') 
            return
        
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class DAE(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        ).double()
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, d),
        ).double()

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        encoded_x = self.encode(x)
        decoded_x = self.decode(encoded_x)
        return decoded_x

class Regressor(nn.Module):
    def __init__(self, d:int):
        super().__init__()
        self.reg = nn.Sequential(
            nn.Linear(d, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.reg(x)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def get_df_by_date_id(start, end):
    lf = pl.scan_parquet('inputs/train.parquet/*/*.parquet')
    filter_ = ((pl.col('date_id') > start) & (pl.col('date_id') < end))
    df = lf.filter(filter_).collect()
    return df


def get_dataset(partition_size: int = 200):
    lf = pl.scan_parquet('inputs/train.parquet/*/*.parquet')
    date_ids = sorted(lf.select('date_id').unique().collect()['date_id'].to_list())
    n  = len(date_ids)
    ranges = list(range(0, n, partition_size))
    for i in range(0, len(ranges)-1):
        start = ranges[i]
        end = ranges[i+1]
        df = get_df_by_date_id(start,end).to_pandas()
        df = reduce_mem_usage(df)
        yield df


def main(partition_size=10):
    cols = pl.scan_parquet('inputs/train.parquet/*/*.parquet').collect_schema().names()
    feature_cols = [x for x in cols if 'feature' in x]
    target_col = 'responder_6'
    dfs = get_dataset(partition_size=partition_size)
    for i, df in enumerate(dfs):
        print(f'Iteration {i}')
        X = df[feature_cols]
        y = df[target_col]
       
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.01)
         # firs iter
        if i == 0:
            preproc = pipeline.Pipeline(
                steps=[
                    ('impute', SimpleImputer(strategy='mean', fill_value=0)),
                    # ('min_max', preprocessing.MinMaxScaler()),
                    ('norm', preprocessing.StandardScaler()),
                ]
            )
            preproc.fit(X)
            X_train_trf = preproc.transform(X_train)
            X_test_trf = preproc.transform(X_test)
            reg = nn.Sequential(
                nn.Linear(X_train_trf.shape[1], 100),
                nn.ReLU(),
                nn.BatchNorm1d(100),
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.BatchNorm1d(50),
                nn.Linear(50, 25),
                nn.ReLU(),
                nn.BatchNorm1d(25),
                nn.Linear(25, 1),
                # nn.ReLU(),
                # nn.Linear(32, 16),
                # nn.ReLU(),
                # nn.Linear(16, 1),
            ).double()
            #
            optimizer = torch.optim.AdamW(reg.parameters(), weight_decay=0.1)
        else:
            X_train_trf = preproc.transform(X_train)
            X_test_trf = preproc.transform(X_test)

        y_train_trf = torch.from_numpy(y_train.to_numpy().reshape(-1, 1)).double()
        y_test_trf = torch.from_numpy(y_test.to_numpy().reshape(-1, 1)).double()
        reg_batch_size = 128
        reg_train_loader = DataLoader(
            list(zip(X_train_trf, y_train_trf)),
            batch_size=reg_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0
        )
        reg_test_loader = DataLoader(
            list(zip(X_test_trf, y_test_trf)),
            batch_size=reg_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0
        )
        n_epochs = 1
        total_steps = len(reg_train_loader) * n_epochs
        warmup_steps = int(0.1 * total_steps) # 20% warmup
        # print(X_trf.shape, y_train_trf.shape)
        early_stopper = EarlyStopper(patience=3, min_delta=10)
        # training the model:
        train_model(
            model=reg,
            train_loader=reg_train_loader,
            test_loader=reg_test_loader,
            optimizer=optimizer,
            device='cpu',
            n_epochs=n_epochs,
            eval_freq=500, 
            eval_iter=1,
            warmup_steps=warmup_steps, 
            loss_fn=nn.MSELoss(),
            early_stopper=early_stopper,
            initial_lr=1e-9, min_lr=1e-9
        )
        break

    torch.save(reg, 'reg.pth')
 


class JSModel:
    def __init__(self, batch_size: int = 128, n_epochs: int = 100, device:str='cpu'):
        self.a = 0
        self.preproc = pipeline.Pipeline(
            steps=[
                ('impute', SimpleImputer(strategy='mean', fill_value=0)),
                # ('min_max', preprocessing.MinMaxScaler()),
                # ('norm', preprocessing.StandardScaler()),
            ]
        )
        self.reg = None
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.optimizer = None
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.01)
        X_train_trf = self.preproc.fit_transform(X_train)
        X_test_trf = self.preproc.transform(X_test)
        reg = nn.Sequential(
                nn.Linear(X_train_trf.shape[1], 100),
                nn.ReLU(),
                nn.BatchNorm1d(100),
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.BatchNorm1d(50),
                nn.Linear(50, 25),
                nn.ReLU(),
                nn.BatchNorm1d(25),
                nn.Linear(25, 1),
                # nn.ReLU(),
                # nn.Linear(32, 16),
                # nn.ReLU(),
                # nn.Linear(16, 1),
            ).double()
        optimizer = torch.optim.AdamW(reg.parameters(), weight_decay=0.1)
        self.optimizer = optimizer
        #
        y_train_trf = torch.from_numpy(y_train.to_numpy().reshape(-1, 1)).double()
        y_test_trf = torch.from_numpy(y_test.to_numpy().reshape(-1, 1)).double()
        reg_batch_size = self.batch_size
        reg_train_loader = DataLoader(
            list(zip(X_train_trf, y_train_trf)),
            batch_size=reg_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0
        )
        reg_test_loader = DataLoader(
            list(zip(X_test_trf, y_test_trf)),
            batch_size=reg_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0
        )
        n_epochs = self.n_epochs
        total_steps = len(reg_train_loader) * n_epochs
        warmup_steps = int(0.1 * total_steps) # 20% warmup
        early_stopper = EarlyStopper(patience=3, min_delta=10)
        train_model(
            model=reg,
            train_loader=reg_train_loader,
            test_loader=reg_test_loader,
            optimizer=optimizer,
            device=self.device,
            n_epochs=n_epochs,
            eval_freq=500, 
            eval_iter=1,
            warmup_steps=warmup_steps, 
            loss_fn=nn.MSELoss(),
            early_stopper=early_stopper,
            initial_lr=1e-9, min_lr=1e-9
        )   
        self.reg = reg
        torch.save(reg, 'reg.pth')


    def continue_fit(self, X, y):
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.01)
        #
        X_train_trf = self.preproc.transform(X_train)
        X_test_trf = self.preproc.transform(X_test)
        y_train_trf = torch.from_numpy(y_train.to_numpy().reshape(-1, 1)).double()
        y_test_trf = torch.from_numpy(y_test.to_numpy().reshape(-1, 1)).double()
        #
        reg_batch_size = self.batch_size
        reg_train_loader = DataLoader(
            list(zip(X_train_trf, y_train_trf)),
            batch_size=reg_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0
        )
        reg_test_loader = DataLoader(
            list(zip(X_test_trf, y_test_trf)),
            batch_size=reg_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0
        )
        n_epochs = self.n_epochs
        early_stopper = EarlyStopper(patience=3, min_delta=10)
        model = self.reg
        train_model(
            model=model,
            train_loader=reg_train_loader,
            test_loader=reg_test_loader,
            optimizer=self.optimizer,
            device=self.device,
            n_epochs=n_epochs,
            eval_freq=500, 
            eval_iter=1,
            warmup_steps=1, 
            loss_fn=nn.MSELoss(),
            early_stopper=early_stopper,
            initial_lr=1e-9, min_lr=1e-9
        )   
        self.reg = model
        torch.save(model, 'reg.pth')

    def predict(self, X):
        X_trf = self.preproc.transform(X)
        y_pred = self.reg(torch.from_numpy(X_trf)).detach().numpy()
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y_true=y, y_pred=y_pred)

df = get_df_by_date_id(0, 200)
cols = df.columns
feature_cols = [x for x in cols if 'feature' in x]
target_col = 'responder_6'
X = df[feature_cols]
y = df[target_col]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

model = JSModel(n_epochs=1)
model.fit(X_train, y_train)
model.score(X_test, y_test)


# preproc = pipeline.Pipeline(
#     steps=[
#         ('impute', SimpleImputer(strategy='mean', fill_value=0)),
#         # ('min_max', preprocessing.MinMaxScaler()),
#         # ('norm', preprocessing.StandardScaler()),
#     ]
# )
# # X_train_trf = preproc.fit_transform(X_train)
# # X_test_trf = preproc.transform(X_test)
# # noise = np.random.normal(0,1,X_train_trf.shape)
# # X_train_trf_corrupted = X_train_trf + noise

# # # base models

# # pred_head = 10 ** 5
# # # xgb_reg = xgb.XGBRegressor()
# # # xgb_reg.fit(X_train_trf, y_train)
# # # y_pred_xgb = xgb_reg.predict(X_test_trf[:pred_head])
# # # r2_xgb = r2_score(y_true=y_test[:pred_head],y_pred=y_pred_xgb)
# # # print(f'Base Linee XGB: {r2_xgb}')

# # # ridge
# # ridge = Ridge().fit(X_train_trf, y_train)
# # y_pred_ridge = ridge.predict(X_test_trf)
# # r2_ridge = r2_score(y_true=y_test,y_pred=y_pred_ridge)
# # print(f'Base Linee Ridge: {r2_ridge}')

# # # # mlp
# # # mlp = MLPRegressor([100, 50, 25, 1]).fit(X_train_trf, y_train)
# # # y_pred_mlp = mlp.predict(X_test_trf)
# # # r2_mlp = r2_score(y_true=y_test,y_pred=y_pred_mlp)
# # # print(f'Base Linee MLP: {r2_mlp}')

# # # training DAE
# # # Create dataloader
# # # batch_size=2**10
# # n_epochs = 100
# # shuffle=True
# # drop_last=True
# # num_workers=0
# # # train_ratio = 0.90
# # # split_idx = int(train_ratio * len(X_train_trf))

# # # train_loader = DataLoader(
# # #     list(zip(X_train_trf_corrupted[:split_idx], X_train_trf[:split_idx])),
# # #     batch_size=batch_size,
# # #     shuffle=shuffle,
# # #     drop_last=drop_last,
# # #     num_workers=num_workers
# # # )
# # # test_loader = DataLoader(
# # #     list(zip(X_train_trf_corrupted[split_idx:], X_train_trf[split_idx:])),
# # #     batch_size=batch_size,
# # #     shuffle=shuffle,
# # #     drop_last=drop_last,
# # #     num_workers=num_workers
# # # )

# # # n_epochs = 1000
# # # dae = DAE(d=X_train_trf.shape[1])
# # # optimizer = torch.optim.AdamW(dae.parameters(), weight_decay=0.1)
# # # total_steps = len(train_loader) * n_epochs
# # # warmup_steps = int(0.2 * total_steps) # 20% warmup
# # # loss_fn = nn.MSELoss()
# # # train_model(
# # #     model=dae,
# # #     train_loader=train_loader,
# # #     test_loader=test_loader,
# # #     optimizer=optimizer,
# # #     device='cpu',
# # #     n_epochs=n_epochs,
# # #     eval_freq=500, 
# # #     eval_iter=1,
# # #     warmup_steps=20, 
# # #     loss_fn=nn.MSELoss(),
# # #     initial_lr=1e-5, min_lr=1e-5
# # # )

# # # # regression
# # # # data for regression
# # # dae.eval()
# # # X_train_dae = torch.from_numpy(X_train_trf).double()
# # # X_trf = dae.encode(X_train_dae).detach()
# # y_train_trf = torch.from_numpy(y_train.to_numpy().reshape(-1, 1)).double()
# # y_test_trf = torch.from_numpy(y_test.to_numpy().reshape(-1, 1)).double()
# # reg_batch_size = 128
# # reg_train_loader = DataLoader(
# #     list(zip(X_train_trf, y_train_trf)),
# #     batch_size=reg_batch_size,
# #     shuffle=shuffle,
# #     drop_last=drop_last,
# #     num_workers=num_workers
# # )
# # reg_test_loader = DataLoader(
# #     list(zip(X_test_trf, y_test_trf)),
# #     batch_size=reg_batch_size,
# #     shuffle=shuffle,
# #     drop_last=drop_last,
# #     num_workers=num_workers
# # )

# # total_steps = len(reg_train_loader) * n_epochs
# # warmup_steps = int(0.2 * total_steps) # 20% warmup
# # # print(X_trf.shape, y_train_trf.shape)
# # early_stopper = EarlyStopper(patience=3, min_delta=10)

# # reg = nn.Sequential(
# #     nn.Linear(X_train_trf.shape[1], 100),
# #     nn.ReLU(),
# #     nn.BatchNorm1d(100),
# #     nn.Linear(100, 50),
# #     nn.ReLU(),
# #     nn.BatchNorm1d(50),
# #     nn.Linear(50, 25),
# #     nn.ReLU(),
# #     nn.BatchNorm1d(25),
# #     nn.Linear(25, 1),
# #     # nn.ReLU(),
# #     # nn.Linear(32, 16),
# #     # nn.ReLU(),
# #     # nn.Linear(16, 1),
# # ).double()
# # print(y_train.shape)
# # loss=nn.MSELoss() # loss function
# # optimizer = torch.optim.AdamW(reg.parameters(), weight_decay=0.1)

# # # training the model:
# # train_model(
# #     model=reg,
# #     train_loader=reg_train_loader,
# #     test_loader=reg_test_loader,
# #     optimizer=optimizer,
# #     device='cpu',
# #     n_epochs=100,
# #     eval_freq=500, 
# #     eval_iter=1,
# #     warmup_steps=3, 
# #     loss_fn=nn.MSELoss(),
# #     early_stopper=early_stopper,
# #     initial_lr=1e-9, min_lr=1e-9
# # )

# # torch.save(reg, 'reg.pth')


# # # test_loader = DataLoader(
# # #     list(zip(torch.from_numpy(X_test.to_numpy()), torch.from_numpy(y_test.to_numpy()))),
# # #     batch_size=batch_size,
# # #     shuffle=shuffle,
# # #     drop_last=drop_last,
# # #     num_workers=num_workers
# # # )
# # # # 
# # # score_predictions(preproc,dae,reg,X_test,y_test)

# # # import xgboost as xgb

# # # tree = xgb.XGBRegressor()
# # # tree.fit(X_train_trf, y_train_trf)


# # class MLP(nn.Module):
# #     def __init__(self, in_dim, out_dim=1):
# #         super(MLP, self).__init__()
# #         self.layers = nn.Sequential(
# #             nn.Linear(in_dim, 10000),
# #             nn.ReLU(),
# #             nn.Linear(10000, 1),
# #             # nn.ReLU(),
# #             # nn.Linear(32, 16),
# #             # nn.ReLU(),
# #             # nn.Linear(16, 1)
# #         ).double()
        
# #     def forward(self, x):
# #         return self.layers(x)
# # num_epochs = 1000
# # batch_size = 256
# # loss_func = nn.MSELoss()
# # mlp_torch = MLP(in_dim=X_train_trf.shape[1])
# # optimizer = torch.optim.Adam(mlp_torch.parameters(), lr=0.01, weight_decay=0.01)
# # trainloader = DataLoader(list(zip(X_train_trf, y_train)), batch_size=batch_size)
# # testloader = DataLoader(list(zip(X_test_trf[:10000], y_test[:10000])), batch_size=batch_size)

# # ### training step
# # in_sample_r2_ = []
# # for epoch in range(num_epochs): 
# #     in_sample_r2_temp = []
# #     running_loss = []
# #     for id_batch, (X_batch, y_batch) in enumerate(trainloader):
# #         optimizer.zero_grad()
# #         y_pred = mlp_torch(X_batch)
# #         loss = loss_func(y_pred, y_batch.unsqueeze(1))
# #         loss.backward()
# #         optimizer.step()
# #         running_loss.append(loss.item())
# #         # store in-sample R-squared
# #         in_sample_r2_temp.append(r2_score(y_batch.detach().numpy(), y_pred.T[0].detach().numpy()))
# #     in_sample_r2_.append(np.mean(in_sample_r2_temp))
    
# #     if epoch % 5 == 1:
# #         print(f"Epoch {epoch}: {np.mean(running_loss)}, averaged in-sample R-squared is: {np.mean(in_sample_r2_temp)}")
