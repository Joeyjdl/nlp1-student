import datasets 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm


VOCAB_SIZE = 2048
MAX_LEN = 256
BS = 16
PAD_ID = 2047


def collate_fn(batch):
    def pad(sequences):
        out = torch.full(
            (len(sequences), MAX_LEN),
            PAD_ID,
            dtype=torch.long,
        )
        for i, ids in enumerate(sequences):
            out[i, :len(ids)] = torch.tensor(ids)
        return out

    x1 = pad([item["ids1"] for item in batch])
    x2 = pad([item["ids2"] for item in batch])
    y = torch.tensor(
        [item["label"] for item in batch],
        dtype=torch.long,
    )

    return x1, x2, y


class DualLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        pad_id=PAD_ID,
        embed_dim=128,
        lstm_hidden_dim=128,
        lstm_layers=2,
        linear_hidden_dim=128,
        dropout=.2,
    ):
        super().__init__()
        self.pad_id = pad_id

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)

        self.encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden_dim,
            batch_first=True,
            num_layers=lstm_layers,
            bidirectional=True,
            dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * 4 * lstm_hidden_dim, linear_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(linear_hidden_dim, 2),
        )

    def _encode(self, seq):
        lengths = (seq != self.pad_id).sum(dim=1).cpu()

        emb = self.embedding(seq)
        emb = self.embedding_dropout(emb)

        # Avoid diminishing gradient through padding
        packed = nn.utils.rnn.pack_padded_sequence(
            emb,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        _, (h_n, _) = self.encoder(packed)

        return torch.cat([h_n[-2], h_n[-1]], dim=-1) # Bidirectional

    def forward(self, seq_1, seq_2, inf=False):
        h_n_1 = self._encode(seq_1)
        h_n_2 = self._encode(seq_2)

        logits = self.classifier(
            torch.cat([
                h_n_1,
                h_n_2,
                torch.abs(h_n_1 - h_n_2),
                h_n_1 * h_n_2
            ], dim=-1)
        )

        return torch.softmax(logits, dim=-1) if inf == True else logits


def main():
	tokenizer = Tokenizer.from_file("tokenizer.json")

	def tokenize_batch(batch):
		return {
			"ids1": [
				tokenizer.encode(s).ids
				for s in batch["sentence1"]
			],
			"ids2": [
				tokenizer.encode(s).ids
				for s in batch["sentence2"]
			],
		}

	ds_labeled = datasets.load_dataset("google-research-datasets/paws", "labeled_final")
	encoded_labeled = ds_labeled.map(
		tokenize_batch,
		batched=True,
		remove_columns=["id", "sentence1", "sentence2"],
	)
	test_loader = DataLoader(
		encoded_labeled["test"],
		batch_size=BS,
		collate_fn=collate_fn,
	)

	model = DualLSTMClassifier()
	ckpt = torch.load("model_ft_labeled.pth", map_location="cpu", weights_only=True)
	model.load_state_dict(ckpt["wts"])

	total_correct = 0
    
	for sen_1, sen_2, labels in tqdm(test_loader):
		with torch.set_grad_enabled(False):
			logits = model(sen_1, sen_2)

			total_correct += (logits.argmax(dim=1) == labels).sum().item()

	accuracy = total_correct / len(encoded_labeled["test"])
	print(total_correct)
	print(f"Accuracy on test set: {accuracy:.2%}")


if __name__ == "__main__":
    main()
