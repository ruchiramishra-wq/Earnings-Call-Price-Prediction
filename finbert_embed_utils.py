import os, glob, hashlib, shutil
import torch
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer



def setup_finbert():
    """
    Sets up the FinBERT model and tokenizer for embedding financial text.

    Returns:
        model: The FinBERT model.
        tokenizer: The FinBERT tokenizer.
        device: torch.device
    """
    model_name = "yiyanghkust/finbert-tone"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    encoder.eval()

    for param in encoder.parameters():  # freeze FinBERT weights
        param.requires_grad = False

    return encoder, tokenizer, device


def chunks(text, tokenizer, max_tokens=512, overlap=50):
    """
    Splits the input text into chunks of tokens with specified maximum length (512) and overlap.

    Args:
        text (str): The input text to be chunked.
        tokenizer: The tokenizer to convert text to token IDs.
        max_tokens (int): Maximum number of tokens per chunk.
        overlap (int): Number of overlapping tokens between consecutive chunks.

    Returns:
        List of chunks, where each chunk is a list of token IDs.        
    """
    tokens = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False
    )["input_ids"]

    out = []
    start = 0
    while start < len(tokens):
        out.append(tokens[start:start + max_tokens])
        start += max_tokens - overlap
    return out


def chunk_to_vector(chunk_id_list, encoder, tokenizer, device, batch_size=16):
    """
    Takes in a list of chunks (each chunk is a list of token IDs), uses FinBERT to compute
    CLS vector for each chunk.

    Args:
        chunk_id_list: List of chunks, where each chunk is a list of token IDs.
        encoder: FinBERT model.
        tokenizer: FinBERT tokenizer.
        device: cpu or gpu device.
        batch_size: Number of chunks to process in a batch.

    Returns:
        torch.Tensor of shape (num_chunks, hidden_dim)
    """
    vecs = []

    with torch.no_grad():
        # process in batches and prepare inputs by padding/truncating
        for i in range(0, len(chunk_id_list), batch_size):
            batch = chunk_id_list[i:i + batch_size]

            inputs = [
                tokenizer.prepare_for_model(
                    ch,
                    add_special_tokens=True,
                    max_length=512,
                    truncation=True,
                    return_attention_mask=True
                )
                for ch in batch
            ]

            enc = tokenizer.pad(
                inputs,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )

            # ensure batch dimension
            if enc["input_ids"].dim() == 1:
                enc["input_ids"] = enc["input_ids"].unsqueeze(0)
            if enc["attention_mask"].dim() == 1:
                enc["attention_mask"] = enc["attention_mask"].unsqueeze(0)
            if "token_type_ids" in enc and enc["token_type_ids"].dim() == 1:
                enc["token_type_ids"] = enc["token_type_ids"].unsqueeze(0)

            enc = {k: v.to(device) for k, v in enc.items()}

            out = encoder(**enc).last_hidden_state   # (B,512,768)
            vec = out[:, 0, :]                       # (B,768) CLS embedding
            vecs.append(vec)

    vec = torch.cat(vecs, dim=0)
    return vec  # (C,768)


def transcript_id(text):
    """
    Generates a unique identifier for a given transcript using its MD5 hash.
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def build_cache(data, cache_dir, encoder, tokenizer, device, overlap=0):
    """
    Takes in a dataset of transcripts, labels, and financial features, computes FinBERT
    embeddings, and caches to disk.

    Args:
        data: List of tuples (transcript, label, fin_features).
        cache_dir: Directory to store cached embeddings.
        encoder: FinBERT model.
        tokenizer: FinBERT tokenizer.
        device: cpu or gpu device.
        overlap: Number of overlapping tokens between consecutive chunks.

    """
    os.makedirs(cache_dir, exist_ok=True)

    for i, (transcript, y, fin_features) in enumerate(data):
        cid = transcript_id(transcript)
        path = os.path.join(cache_dir, f"{cid}.pt")
        if os.path.exists(path):
            continue

        chunk_id_list = chunks(transcript, tokenizer, overlap=overlap)
        Z = chunk_to_vector(chunk_id_list, encoder, tokenizer, device, batch_size=8)

        f = torch.tensor(fin_features, dtype=torch.float16)
        torch.save(
            {"Z": Z.to(torch.float16), "fin_features": f, "y": int(y)},
            path
        )

        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(data)}] cached | files={len(glob.glob(cache_dir+'/*.pt'))}")

    print(f"Cached {len(data)} transcripts → {cache_dir}")


class CachedDataset(Dataset):
    """
    Dataset for loading cached FinBERT embeddings and labels.
    """
    def __init__(self, cache_dir):
        self.paths = glob.glob(os.path.join(cache_dir, "*.pt"))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        obj = torch.load(self.paths[idx], map_location="cpu")
        return obj["Z"].float(), torch.tensor(obj["y"], dtype=torch.float32)


class CachedZFinDataset(Dataset):
    """
    Dataset for loading cached FinBERT embeddings, financial features, and labels.
    """
    def __init__(self, cache_dir):
        self.paths = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        obj = torch.load(self.paths[idx], map_location="cpu")
        Z = obj["Z"].float()  # (C,768)
        y = torch.tensor(obj["y"], dtype=torch.float32)
        fin = obj["fin_features"].float().view(-1)
        return Z, fin, y
    

def zip_cache(cache_dir, zip_path):
    """
    Zips the entire cache directory into a single .zip file.
    """
    assert os.path.exists(cache_dir), f"{cache_dir} does not exist"
    shutil.make_archive(
        base_name=zip_path.replace(".zip", ""),
        format="zip",
        root_dir=cache_dir
    )
    print(f"Created zip file: {zip_path}")

    