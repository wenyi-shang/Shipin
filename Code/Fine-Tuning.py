# Fine-tuning GuWenBERT for Chinese Classical Poetry with Masked Language Modeling
import os
import gc
import json
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import DataLoader, IterableDataset


# Set seeds for reproducibility 
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class PoemIterableDataset(IterableDataset):
    """
    Custom dataset class for handling poem text data.
    Inherits from IterableDataset for memory-efficient streaming of large datasets.
    """
    def __init__(self, texts):
        self.texts = texts
    def __iter__(self):
        yield from self.texts

def get_punctuation_ids(tokenizer):
    """
    Extract token IDs for punctuation marks to exclude them from masking.
    """
    ascii_p = list(r"""!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~""")
    cjk_p = ["，","。","、","；","？","！","：","“","”","‘","’","（","）"]
    chars = ascii_p + cjk_p
    return set(tokenizer.convert_tokens_to_ids([c for c in chars if c in tokenizer.vocab]))

def make_collate_fn(tokenizer, punctuation_ids, mlm_prob):
    """
    Create a collate function for DataLoader that implements masked language modeling.
    
    Args:
        tokenizer: HuggingFace tokenizer
        punctuation_ids: Set of punctuation token IDs to exclude from masking
        mlm_prob: Probability of masking each token 
    
    Returns:
        function: Collate function for DataLoader
    """

    mask_id = tokenizer.mask_token_id
    vocab_size = tokenizer.vocab_size

    def collate_fn(batch_texts):
        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_special_tokens_mask=True,
            return_tensors="pt"
        )
        orig_input_ids = enc["input_ids"].clone()
        input_ids = enc["input_ids"]

        special_mask = enc["special_tokens_mask"].bool()
        punct_ids = torch.tensor(list(punctuation_ids))
        punct_mask = torch.isin(input_ids, punct_ids.to(input_ids.device))
        no_mask = special_mask | punct_mask

        labels = orig_input_ids.clone()
        rand = torch.rand(labels.shape)
        mask_pos = (rand < mlm_prob) & ~no_mask
        labels[~mask_pos] = -100

        prob_matrix = torch.rand(labels.shape)
        mask_replace = (prob_matrix < 0.8) & mask_pos
        input_ids[mask_replace] = mask_id
        random_replace = (prob_matrix >= 0.8) & (prob_matrix < 0.9) & mask_pos
        random_tokens = torch.randint(vocab_size, labels.shape, device=input_ids.device)
        input_ids[random_replace] = random_tokens[random_replace]

        return {
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "orig_input_ids": orig_input_ids
        }

    return collate_fn

def split_dataset_2way(poems_list, train_ratio=0.8, val_ratio=0.2, seed=42):
    """
    Split dataset into training and validation sets.

    """
    assert abs((train_ratio + val_ratio) - 1.0) < 1e-6
    np.random.seed(seed)
    poems_array = np.array(poems_list)
    np.random.shuffle(poems_array)
    n_train = int(len(poems_list) * train_ratio)
    return list(poems_array[:n_train]), list(poems_array[n_train:])


def freeze_layers(model, num_layers_to_freeze=6):
    """
    Freeze the bottom N layers of the BERT model to reduce trainable parameters.
    """
    for name, param in model.named_parameters():
        if any(name.startswith(f"bert.encoder.layer.{i}.") for i in range(num_layers_to_freeze)):
            param.requires_grad = False
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Freeze {num_layers_to_freeze} and the parameters are: {trainable_params}")


def run_grid_search(train_ds, val_ds, LRs, BATCH_SIZES, EPOCHS, mlm_probs, tokenizer, save_dir):
    """
    Perform grid search over hyperparameters to find the best configuration.
    
    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        LRs: List of learning rates to try
        BATCH_SIZES: List of batch sizes to try
        EPOCHS: List of epoch counts to try
        mlm_probs: Masking probability for MLM
        tokenizer: HuggingFace tokenizer
        save_dir: Directory to save models and results
    
    Returns:
        tuple: (best_config, best_val_acc, best_loss_curve, best_val_acc_curve)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    punctuation_ids = get_punctuation_ids(tokenizer)

    # Track global best results across all hyperparameter combinations
    best_val_acc = 0.0
    best_config = None
    best_model_state_dict = None
    best_loss_curve = []
    best_val_acc_curve = []

    # Early stopping parameters
    patience = 8
    min_delta = 0.0001

    # Grid search over all hyperparameter combinations
    for lr in LRs:
        for bs in BATCH_SIZES:
            for ep in EPOCHS:
                for mlm_prob in mlm_probs:
                    print(f" Running: lr={lr}, bs={bs}, ep={ep}, mlm_prob={mlm_prob} on {device}")

                    try:
                        model = AutoModelForMaskedLM.from_pretrained("ethanyt/guwenbert-base", use_safetensors=True).to(device).train()
                        freeze_layers(model, num_layers_to_freeze=6)
                    except RuntimeError:
                        print("Switching to CPU due to OOM")
                        device = torch.device("cpu")
                        torch.cuda.empty_cache()
                        gc.collect()
                        model = AutoModelForMaskedLM.from_pretrained("ethanyt/guwenbert-base", use_safetensors=True).to(device).train()

                    # Setup optimizer and scheduler
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ep)

                    # Setup data loaders
                    collate_fn = make_collate_fn(tokenizer, punctuation_ids, mlm_prob)
                    train_loader = DataLoader(train_ds, batch_size=bs, collate_fn=collate_fn)
                    val_loader   = DataLoader(val_ds, batch_size=bs, collate_fn=collate_fn)
                    
                    # Training tracking variables
                    loss_curve = []
                    val_acc_curve = []
                    wait = 0
                    best_hyperparam_val_acc = 0.0
                    best_epoch_state_dict = None

                    for epoch in range(ep):
                        model.train()
                        epoch_loss, epoch_steps = 0.0, 0
                        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{ep}"):
                            # Forward pass
                            outputs = model(
                                batch["input_ids"].to(device),
                                attention_mask=batch["attention_mask"].to(device),
                                labels=batch["labels"].to(device)
                            )
                            loss = outputs.loss

                            # Backward pass
                            loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()

                            # Track loss
                            epoch_loss += loss.item()
                            epoch_steps += 1

                         # Update learning rate
                        scheduler.step()
                        current_lr = scheduler.get_last_lr()[0]
                        epoch_avg_loss = epoch_loss / epoch_steps
                        loss_curve.append(epoch_avg_loss)

                        # Validation
                        model.eval()
                        correct_val, total_val = 0, 0
                        for batch_acc in val_loader:
                            input_ids = batch_acc["input_ids"].to(device)
                            attention_mask = batch_acc["attention_mask"].to(device)
                            labels = batch_acc["labels"].to(device)

                    # Get predictions without gradients
                            with torch.no_grad():
                                logits = model(input_ids, attention_mask=attention_mask).logits
                            preds = logits.argmax(dim=-1)

                    # Calculate accuracy for masked tokens only
                            for i in range(input_ids.size(0)):
                                mask_pos = (labels[i] != -100).nonzero(as_tuple=True)[0]
                                for pos in mask_pos:
                                    if labels[i,pos].item() == preds[i,pos].item():
                                        correct_val += 1
                                    total_val += 1

                        val_acc = correct_val / total_val if total_val else 0.0
                        val_acc_curve.append(val_acc)

                        print(f" Epoch {epoch+1} avg_loss={epoch_avg_loss:.4f}, val_MLM_acc={val_acc:.4f}, lr={current_lr:.6e}")

                        # Early stopping 
                        if val_acc > best_hyperparam_val_acc + min_delta:
                            best_hyperparam_val_acc = val_acc
                            wait = 0

                        # Save best model state for this hyperparameter combination
                            best_epoch_state_dict = {
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),
                                'loss': epoch_avg_loss,
                            }
                        else:
                            wait += 1
                            if wait >= patience:
                                print(f"Early stopping at epoch {epoch+1} (no val_MLM_acc improvement)")
                                break

                # Save best of this hyperparam 
                    hyperparam_dir = os.path.join(save_dir, f"fine-tuned-best_lr{lr}_bs{bs}_mlm{mlm_prob}")
                    os.makedirs(hyperparam_dir, exist_ok=True)
                    model.save_pretrained(hyperparam_dir)
                    tokenizer.save_pretrained(hyperparam_dir)

                    # Load best epoch state and save in HuggingFace format
                    model.load_state_dict(best_epoch_state_dict['model'])
                    model.save_pretrained(hyperparam_dir)
                    tokenizer.save_pretrained(hyperparam_dir)

                    plt.figure(figsize=(8,5))
                    ax1 = plt.gca()
                    ax2 = ax1.twinx()
                    ax1.plot(loss_curve, color='blue', marker='o', label='Loss')
                    ax1.set_xlabel("Epoch")
                    ax1.set_ylabel("Loss", color='blue')
                    ax1.tick_params(axis='y', labelcolor='blue')

                    ax2.plot(val_acc_curve, color='red', marker='x', label='Val MLM Acc')
                    ax2.set_ylabel("Val MLM Accuracy", color='red')
                    ax2.tick_params(axis='y', labelcolor='red')

                    plt.title(f"Convergence: lr={lr}, bs={bs}")
                    plt.grid(True)
                    plt.savefig(os.path.join(save_dir, f"loss_valacc_curve_lr{lr}_bs{bs}.png"))
                    plt.close()

                    #  Global best tracking
                    if best_hyperparam_val_acc > best_val_acc:
                        best_val_acc = best_hyperparam_val_acc
                        best_config = (lr, bs, ep)
                        best_model_state_dict = best_epoch_state_dict
                        best_loss_curve = loss_curve.copy()
                        best_val_acc_curve = val_acc_curve.copy()

                    del model, train_loader, val_loader
                    torch.cuda.empty_cache()
                    gc.collect()

    # Save global best 
    if best_model_state_dict is None:
        raise ValueError("No model was saved during grid search. Try lowering min_delta or patience.")

    final_save_dir = os.path.join(save_dir, "guwenbert-best")
    os.makedirs(final_save_dir, exist_ok=True)

    # Load and save the globally best model
    model = AutoModelForMaskedLM.from_pretrained("ethanyt/guwenbert-base")
    model.load_state_dict(best_model_state_dict['model'])
    model.save_pretrained(final_save_dir)
    tokenizer.save_pretrained(final_save_dir)

    # Also plot final best
    plt.figure(figsize=(8,5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(best_loss_curve, color='blue', marker='o', label='Loss')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2.plot(best_val_acc_curve, color='red', marker='x', label='Val MLM Acc')
    ax2.set_ylabel("Val MLM Accuracy", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title(f"Final best: val_acc={best_val_acc:.4f}")
    plt.grid(True)
    plt.savefig(os.path.join(final_save_dir, "final_best_loss_valacc.png"))
    plt.close()

    print(f"\nSaved global best model with val_MLM_acc={best_val_acc:.4f} at {final_save_dir}")

    return best_config, best_val_acc, best_loss_curve, best_val_acc_curve




def evaluate_and_plot(model_path, tokenizer_path, test_ds, loss_curve, batch_size=8, epoch_size=None):
    """
    Evaluate the trained model on test dataset and calculate MLM accuracy.
    
    Args:
        model_path: Path to saved model
        tokenizer_path: Path to saved tokenizer
        test_ds: Test dataset
        loss_curve: Training loss curve (for plotting)
        batch_size: Batch size for evaluation
        epoch_size: Number of epochs
    """
       
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path, use_safetensors=True).to(device).eval()
   
    punctuation_ids = get_punctuation_ids(tokenizer)
    collate_fn = make_collate_fn(tokenizer, punctuation_ids)
    loader = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn)

    # Evaluate model
    total, correct = 0, 0
    for batch in tqdm(loader, desc="Testing"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Get predictions
        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask).logits
        preds = logits.argmax(dim=-1)

        for i in range(input_ids.size(0)):
            mask_pos = (labels[i] != -100).nonzero(as_tuple=True)[0]
            for pos in mask_pos:
                if labels[i,pos].item() == preds[i,pos].item():
                    correct += 1
                total += 1
    acc = correct / total if total else 0.0
    print(f"\n MLM accuracy on test set: {acc:.4f}")

    del model, tokenizer, loader
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":

    with open('Data/Xianqin Han Wei Jin Nanbeichao shi/train_set.json', 'r', encoding='utf-8') as f:
        all_poems = json.load(f)
    short_all_poems = [p["poem"] for poet in all_poems.values() for p in poet if 0 < len(p["poem"].strip()) <= 510]

    short_train_poems, short_val_poems = split_dataset_2way(short_all_poems, 0.8, 0.2)

    train_ds = PoemIterableDataset(short_train_poems)
    val_ds = PoemIterableDataset(short_val_poems)

    with open('Data/Xianqin Han Wei Jin Nanbeichao shi/test_set.json', 'r', encoding='utf-8') as f:
        test_poems = json.load(f)
    short_test_poems = [p["poem"] for poet in test_poems.values() for p in poet if 0 < len(p["poem"].strip()) <= 510]
    test_ds = PoemIterableDataset(short_test_poems)

    tokenizer = AutoTokenizer.from_pretrained("ethanyt/guwenbert-base")

    LRs = [1e-5, 3e-5, 5e-5]
    BATCH_SIZES = [32, 64]
    EPOCHS = [200]
    mlm_probs = [0.1, 0.15]

    best_config, best_val_acc, best_loss_curve, best_val_acc_curve = run_grid_search(
        train_ds, val_ds, LRs, BATCH_SIZES, EPOCHS, mlm_probs, tokenizer, save_dir="./Models1.6"
    )
    print(f"\n Best config: lr={best_config[0]}, bs={best_config[1]}, ep={best_config[2]} with val_acc={best_val_acc:.4f}")

    evaluate_and_plot(
        model_path="./Models1.6/guwenbert-best",
        tokenizer_path="./Models1.6/guwenbert-best",
        test_ds=test_ds,
        loss_curve=best_loss_curve,
        batch_size=16,
        epoch_size=len(best_loss_curve)//(best_config[2]) if best_config else None
    )
