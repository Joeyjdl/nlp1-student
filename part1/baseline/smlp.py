from scipy.sparse import csr_matrix
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import time
import scipy
import scipy.sparse
import json
import numpy as np

class SparseMLP(nn.Module):
    """One-hidden-layer MLP that accepts sparse input tensors.

    Inputs: feature vectors of size `input_dim` (e.g. bag-of-words).
    Process: sparse matmul into a dense hidden layer, ReLU + dropout, linear (num_classes) and log-softmax outputs
        log-softmax, rather than softmax, for numerical stability
    Output: logprobs over `num_classes`.
    """

    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.2):
        """
        input_dim: size of the input feature space (e.g. vocabulary size)
        hidden_dim: width of the hidden layer
        num_classes: number of output labels
        dropout: fraction of hidden units zeroed during training (regularisation)
        """
        super().__init__()

        # First-layer weights stay dense; inputs arriving here are sparse.
        self.weight = nn.Parameter(
            torch.empty(input_dim, hidden_dim)
        )
        self.bias = nn.Parameter(
            torch.zeros(hidden_dim)
        )

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Everything after the first layer is dense.
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),            
            nn.LogSoftmax(dim=-1) # for numerical stability, it is common to use log softmax, rather than softmax
        )

        nn.init.xavier_uniform_(self.weight) # why xavier?

    def forward(self, x):
        """Map a sparse batch [batch, input_dim] to class log-probabilities [batch, num_classes]."""
        # First matmul keeps x sparse; result is dense from here on.
        x = torch.sparse.mm(x, self.weight)
        x = x + self.bias

        x = self.activation(x)
        x = self.dropout(x)
        # logprobs
        return self.output(x)

class SparseLinear(nn.Module):
    """Log-linear softmax regressor that accepts sparse input tensors.

    Inputs: feature vectors of size `input_dim` (e.g. bag-of-words).
    Process: sparse matmul straight into num_classes linear predictors and log-softmax outputs
        log-softmax, rather than softmax, for numerical stability
    Output: logprobs over `num_classes`.
    """

    def __init__(self, input_dim, num_classes):
        """
        input_dim: size of the input feature space (e.g. vocabulary size)
        num_classes: number of output labels
        """
        super().__init__()

        # Weights start at zero, like sklearn's LogisticRegression
        self.weight = nn.Parameter(
            torch.zeros(input_dim, num_classes)
        )
        self.bias = nn.Parameter(
            torch.zeros(num_classes)
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """Map a sparse batch [batch, input_dim] to class log-probabilities [batch, num_classes]."""
        x = torch.sparse.mm(x, self.weight)
        x = x + self.bias
        # logprobs
        return self.log_softmax(x)

MODEL_TYPES = {
    'mlp': SparseMLP,
    'linear': SparseLinear,
}

def scipy_to_torch_sparse(x):
    """Convert a SciPy sparse matrix to a coalesced PyTorch sparse COO tensor.

    Input: SciPy sparse matrix.
    Output: torch sparse tensor with the same shape and non-zero values.
    """
    x = x.tocoo()
    indices = torch.tensor([x.row, x.col], dtype=torch.long)
    values = torch.tensor(x.data, dtype=torch.float32)

    return torch.sparse_coo_tensor(indices, values, size=x.shape).coalesce()

class SparseDataset(Dataset):
    """Wraps a SciPy sparse feature matrix and label list for use with DataLoader.

    Inputs: X (sparse matrix of examples), y (integer labels).
    Output per item: one sparse row and its label tensor.
    """

    def __init__(self, X, y):
        self.X = X
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def sparse_collate(batch):
    """Collate individual sparse rows into one PyTorch sparse batch.

    Input: list of (sparse row, label) pairs from SparseDataset.
    Process: stack SciPy rows, convert to a torch sparse tensor, stack labels.
    Output: (X_batch sparse tensor, y_batch long tensor).
    """
    X_rows, y_rows = zip(*batch)

    # Stack SciPy sparse rows, then convert the whole batch at once.
    X_batch = scipy.sparse.vstack(X_rows)
    X_batch = scipy_to_torch_sparse(X_batch)

    y_batch = torch.stack(y_rows)

    return X_batch, y_batch

def vectorize(texts, encoder):
    """Encode texts with an encoder that returns dense vectors and stack them into a SciPy CSR matrix.

    texts: list of inputs for the encoder
    encoder: an instance of Encoder returning numpy vectors
    """
    return csr_matrix([encoder(x) for x in texts])


def train_one_epoch(model, dataloader, optimizer, criterion, device="cpu", l2_penalty=0.0):
    """Run one training epoch: forward, loss, backward, optimizer step.

    Inputs: model, train dataloader, optimizer, loss function, device, 
        l2_penalty: coefficient of an L2 penalty on the first-layer weights (model.weight) added to the loss.
    Output: dict with average train loss and accuracy for the epoch.
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    model.train()  # enable dropout / training behaviour

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for X_batch, y_batch in dataloader:
        # move the batch to the device
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()  # clear gradients from the previous step

        logprobs = model(X_batch) # forward pass
        loss = criterion(logprobs, y_batch) # compute the loss

        # gradients use the penalised objective, the reported loss is the data loss only
        objective = loss + l2_penalty * (model.weight ** 2).sum()
        objective.backward()   # compute gradients
        optimizer.step()  # update parameters

        # accumulate weighted loss and correct predictions for epoch averages
        total_loss += loss.item() * y_batch.size(0)
        predictions = logprobs.argmax(dim=1) # get the predicted class
        total_correct += (predictions == y_batch).sum().item() # count the number of correct predictions
        total_examples += y_batch.size(0) # count the number of examples

    return {
        "loss": total_loss / total_examples, # average loss
        "accuracy": total_correct / total_examples, # average accuracy
    }

def evaluate(model, dataloader, criterion, device="cpu"):
    """Evaluate the model without updating weights.

    Inputs: model, eval dataloader, loss function, device.
    Output: (metrics dict with loss/accuracy, numpy array of predicted labels).
    """
    if isinstance(device, str):
        device = torch.device(device)
        
    model.eval()  # disable dropout / training behaviour

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    all_preds = []
    with torch.no_grad():  # no gradients needed for evaluation

        for X_batch, y_batch in dataloader:
            # move the batch to the device
            X_batch = X_batch.to(device) 
            y_batch = y_batch.to(device)

            logprobs = model(X_batch) # forward pass

            loss = criterion(logprobs, y_batch) # compute the loss

            # accumulate metrics; also keep predictions for later use
            total_loss += loss.item() * y_batch.size(0)
            predictions = logprobs.argmax(dim=1) # get the predicted class
            all_preds.append(predictions)

            total_correct += (predictions == y_batch).sum().item() # count the number of correct predictions
            total_examples += y_batch.size(0)

    # concatenate batch predictions into one array
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples
    }, torch.cat([t.detach().cpu() for t in all_preds]).numpy()

def train_neural_network_model(training_set, validation_set, num_classes, encoder, vectorizer, save_to, model_type='mlp',  
                                hidden_dim=128, dropout=0.1, batch_size=64, learning_rate=1e-3, 
                                weight_decay=1e-4, num_epochs=20, C=None, device=torch.device("cpu")):
    """Encode inputs, train a SparseMLP or SparseLinear model, and checkpoint the best validation model.

    Inputs: train/val (input, label) pairs, num_classes, encoder (input -> vector), 
            vectorizer (function (inputs, encoder) -> SciPy CSR matrix), path prefix to save weights,
            model_type ('mlp' or 'linear'; hidden_dim and dropout are only used by 'mlp'),
            C: if given, add sklearn's L2 penalty 1/(2*C*N) * ||W||^2 on the first-layer weights (N training examples),
            plus standard training hyperparameters.
    Process: vectorise documents, build sparse dataloaders, train for `num_epochs`, save best-by-val-loss.
    Output: per-epoch metrics list, best epoch index, and validation predictions from the best model.
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model_type={model_type}, choose from {tuple(MODEL_TYPES)}")
    if isinstance(device, str):
        device = torch.device(device)
        
    time_start = time.time()
    # Convert the training and validation sets to vectors
    print("Transforming documents...")
    X_train_sparse = vectorizer([text for text, label in training_set], encoder)
    training_labels = [int(label) for text, label in training_set]
    X_val_sparse = vectorizer([text for text, label in validation_set], encoder)
    validation_labels = [int(label) for text, label in validation_set]

    time_document_to_vector = time.time()
    print(f"Document transformation done! Time taken: {time_document_to_vector - time_start:.3f} seconds")

    train_dataset = SparseDataset(X_train_sparse, training_labels) # create the training dataset
    val_dataset = SparseDataset(X_val_sparse, validation_labels) # create the validation dataset

    # create the data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=sparse_collate,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sparse_collate,
    )

    time_dataloader = time.time()
    print(f"Dataloader done! Time taken: {time_dataloader - time_document_to_vector:.3f} seconds")

    # save architecture-defining arguments
    if model_type == 'mlp':
        args = {
            'input_dim': len(encoder), 
            'hidden_dim': hidden_dim, 
            'num_classes': num_classes, 
            'dropout': dropout,
        }
    else:
        args = {
            'input_dim': len(encoder), 
            'num_classes': num_classes, 
        }
    with open(f"{save_to}.json", "w", encoding="utf-8") as f:
        json.dump({'model_type': model_type, 'args': args}, f, indent=2, ensure_ascii=False)
        
    # Create the model based on the input dimension, number of classification classes (and for the MLP the hidden dimension and dropout rate)
    model = MODEL_TYPES[model_type](**args).to(device)
    criterion = nn.NLLLoss() # create the loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # create the optimizer
    # sklearn minimises C * sum(losses) + 0.5 * ||W||^2, which has the same minimiser as mean(losses) + ||W||^2 / (2 * C * N)
    l2_penalty = 0.0 if C is None else 1.0 / (2 * C * len(training_set))

    metrics = []
    print("Training...")
    predictions = None
    for epoch in tqdm(range(num_epochs)):
        # Train the model for one epoch
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, l2_penalty)
        # Evaluate the model on the validation set
        val_metrics, preds = evaluate(model, val_loader, criterion, device)
        # Store the metrics for this epoch
        metrics.append({"train_metrics": train_metrics, "val_metrics": val_metrics})

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train loss: {train_metrics['loss']:.4f} | "
            f"train acc: {train_metrics['accuracy']:.4f} | "
            f"val loss: {val_metrics['loss']:.4f} | "
            f"val acc: {val_metrics['accuracy']:.4f}"
            )
        
        # Saving the best model according to validation loss
        if epoch != 0:
            if val_metrics["loss"] <= best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(model.state_dict(), f"{save_to}.pt")
                best_epoch = epoch
                predictions = preds
        else:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), f"{save_to}.pt")
            best_epoch = epoch
            predictions = preds
    
    time_training = time.time()
    print(f"Training done! Time taken: {time_training - time_dataloader:.3f} seconds")

    return metrics, best_epoch, predictions


def load(path, device="cpu"):
    """
    Reconstruct the model architecture (using {path}.json) and load trained weights (using {path}.pt) and return it.
    """

    with open(f"{path}.json", "r", encoding="utf-8") as f:
        saved = json.load(f)
        
    if isinstance(device, str):
        device = torch.device(device)

    with open(f"{path}.pt", "rb") as f:
        state_dict = torch.load(f, map_location=device)
        
    model = MODEL_TYPES[saved['model_type']](**saved['args'])
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model


def predict_probs(x_sparse, model, batch_size=256):
    """
    Return a numpy array with shape [num_inputs, num_classes] with the pmfs of the inputs as predicted by the model.
    
    x_sparse: SciPy CSR matrix with one encoded input per row (see vectorize)
    model: trained model
    batch_size: how many at a time
    """
    
    device = model.weight.device
        
    labels = [0 for _ in range(x_sparse.shape[0])]

    dl = DataLoader(
        SparseDataset(x_sparse, labels),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sparse_collate,
    )
    
    model.eval()  # disable dropout / training behaviour
    
    all_probs = []
    with torch.no_grad():  # no gradients needed for evaluation

        for x_batch, y_batch in dl:
            # move the batch to the device
            x_batch = x_batch.to(device) 
            logprobs = model(x_batch) # forward pass
            all_probs.extend(logprobs.exp().detach().cpu().numpy())
            
    return np.array(all_probs)
