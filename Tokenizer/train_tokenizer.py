from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents
from tokenizers.normalizers import Lowercase
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Digits, Whitespace
from tokenizers.pre_tokenizers import Punctuation
from tokenizers.trainers import BpeTrainer
from datasets import load_dataset

# Load dataset
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

# Iterator (train split only + remove empty lines)
def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset["train"]), batch_size):
        batch = dataset["train"][i:i+batch_size]["text"]
        batch = [x for x in batch if x.strip()]  # remove empty lines
        yield batch

# Tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Trainer (add vocab size!)
trainer = BpeTrainer(
    vocab_size=30000,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

# Normalization
tokenizer.normalizer = normalizers.Sequence([
    NFD(),
    Lowercase(),
    StripAccents()
])

# Pre-tokenization
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    Whitespace(),
    Punctuation(),
    Digits(individual_digits=True)
])

# Train
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

# Save
tokenizer.save("data/tokenizer-wiki.json")
