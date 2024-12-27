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
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
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


class DAE(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        ).double()
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, d),
            nn.ReLU(),
            nn.Identity(d)
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

# class JSDataset(IterableDataset):
#     def __init__(self, path: str, start: float, end: float, batch_size: int):
#         super().__init__()
#         self.path = path
#         self.batch_size = batch_size
#         lf = pl.scan_parquet(self.path)
#         # get cols
#         cols = lf.columns
#         self.feature_cols = [x for x in cols if 'feature' in x]
#         self.target_col = 'responder_6'
#         # resolve indicies
#         n = lf.select(pl.len()).collect().item()
#         self.start_idx = int(n * start)
#         self.end_idx = int(n * end)
#         n = self.end_idx - self.start_idx + 1
#         self.num_batch = n // batch_size
#         self.n = n
#         self.current_idx = 0
#         print(f'n: {self.n}, start_idx: {self.start_idx}, end_idx: {self.end_idx}, num_batch: {self.num_batch}')

#     def __iter__(self):
#         lf = pl.scan_parquet(self.path)
#         rows = list(range(self.batch_size * self.current_idx, self.batch_size * (self.current_idx +1)))
#         df = lf.select(pl.all().gather(rows)).collect(streaming=True)
#         _X = df[self.feature_cols].to_numpy()
#         noise = np.random.normal(0,1,X.shape)
#         X = _X + noise
#         y = _X
#         # move idx
#         self.current_idx = (self.current_idx + 1) % self.num_batch
#         yield X, y

lf = pl.scan_parquet('inputs/train.parquet/*/*.parquet').head(int(1e6))
cols = lf.columns
feature_cols = [x for x in cols if 'feature' in x]
target_col = 'responder_6'
df = reduce_mem_usage(lf.collect(streaming=True).to_pandas())
X = df[feature_cols]
y = df[target_col]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

preproc = pipeline.Pipeline(
    steps=[
        ('impute', SimpleImputer(strategy='mean', fill_value=0)),
        ('min_max', preprocessing.MinMaxScaler()),
        ('norm', preprocessing.StandardScaler()),
    ]
)
X_train_trf = preproc.fit_transform(X_train)
noise = np.random.normal(0,1,X_train_trf.shape)
X_train_trf_corrupted = X_train_trf + noise



# Create dataloader
batch_size=2**10
shuffle=True
drop_last=True
num_workers=0
train_ratio = 0.90
split_idx = int(train_ratio * len(X_train_trf))

train_loader = DataLoader(
    list(zip(X_train_trf_corrupted[:split_idx], X_train_trf[:split_idx])),
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers
)
test_loader = DataLoader(
    list(zip(X_train_trf_corrupted[split_idx:], X_train_trf[split_idx:])),
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers
)

initial_lr = 0.0001
peak_lr = 0.01
n_epochs = 100
total_training_steps = len(train_loader) * n_epochs

warmup_steps = int(0.2 * total_training_steps) # 20% warmup
print(warmup_steps)
loss_fn = nn.MSELoss()
dae = DAE(d=X_train_trf.shape[1])
learning_rate = 1e-4
weight_decay = 1e-2
optimizer = torch.optim.AdamW(dae.parameters(), weight_decay=0.1)
lr_increment = (peak_lr - initial_lr) / warmup_steps
global_step = -1
dae.train()
track_lrs = []
min_lr = 0.1 * initial_lr
loss_fn = nn.MSELoss()
eval_freq = 10
train_losses = []
val_losses = []
for epoch in range(n_epochs):
    dae.train()
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

        loss = calc_loss_batch(input_batch, target_batch, model=dae, loss_fn=loss_fn, device='cpu')
        loss.backward()

        # Apply gradient clipping after the warmup phase to avoid exploding gradients

        if global_step > warmup_steps:
            torch.nn.utils.clip_grad_norm_(dae.parameters(), max_norm=1.0)  
            
        optimizer.step()
        # Periodically evaluate the model on the training and validation sets
        if global_step % eval_freq == 0:
            train_loss, val_loss = evaluate_model(
                dae, train_loader, test_loader,
                'cpu', 1, loss_fn
            )
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            # Print the current losses
            print(f"Ep {epoch+1} (Iter {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
            )


# # regression
# dae.eval()
# X_train_dae = torch.from_numpy(X_train_trf).float()
# X_trf = dae.encode(X_train_dae).detach()
# print(X_trf.shape)
# reg = nn.Sequential(
#     nn.Linear(X_trf.shape[1], 24),
#     nn.ReLU(),
#     nn.Linear(24, 12),
#     nn.ReLU(),
#     nn.Linear(12, 6),
#     nn.ReLU(),
#     nn.Linear(6, 1)
# )
# y_train_trf = torch.from_numpy(y_train.to_numpy().reshape(-1, 1)).float().squeeze()
# print(y_train.shape)
# loss=nn.MSELoss() # loss function
# optimizers=optim.Adam(params=reg.parameters(),lr=0.01)


# # training the model:
# num_of_epochs=100
# for i in range(num_of_epochs):
#   # give the input data to the architecure
#   y_train_prediction=reg(X_trf)  # model initilizing
#   loss_value=loss(y_train_prediction.squeeze(),y_train_trf)   # find the loss function:
#   optimizers.zero_grad() # make gradients zero for every iteration so next iteration it will be clear
#   loss_value.backward()  # back propagation
#   optimizers.step()  # update weights in NN

#   # print the loss in training part:
#   if i % 10 == 0:
#     print(f'[epoch:{i}]: The loss value for training part={loss_value}')
