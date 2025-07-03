from helper import one_hot

chars = sorted(list(set("hello")))
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for ch, i in char2idx.items()}
data = [char2idx[ch] for ch in "hello"]

print("Characters:", chars)
print("Character to Index Mapping:", char2idx)
print("Data:", data)

vocab_size = len(chars)
seq_len = vocab_size + 1
i = 0

inputs = [one_hot(data[j], vocab_size) for j in range(i, i + seq_len)]
print("One-hot inputs:")
for inp in inputs:
    print(inp)