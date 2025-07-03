import numpy as np


def one_hot(index, vocab_size):
    vec = np.zeros((1, vocab_size))
    vec[0, index] = 1
    return vec

def train(rnn, data, epochs=100, lr=0.01, seq_len=5, vocab_size=27):
    rnn.add_output_layer(rnn.hidden_size, vocab_size)

    for epoch in range(epochs):
        total_loss = 0

        for i in range(len(data) - seq_len):
            # Prepare input sequence and target
            inputs = [one_hot(data[j], vocab_size) for j in range(i, i + seq_len)]
            targets = data[i + 1:i + seq_len + 1]  # next character indices

            # Forward pass
            hidden_states = rnn.forward(inputs)
            outputs = [rnn.output(h) for h in hidden_states]

            # Compute loss
            loss = sum(rnn.cross_entropy(y_pred, y_true_idx)
                       for y_pred, y_true_idx in zip(outputs, targets))
            total_loss += loss

            # Backward pass (manual BPTT)
            dW_xh = np.zeros_like(rnn.W_xh)
            dW_hh = np.zeros_like(rnn.W_hh)
            db_h  = np.zeros_like(rnn.b_h)
            dW_hy = np.zeros_like(rnn.W_hy)
            db_y  = np.zeros_like(rnn.b_y)
            dh_next = np.zeros((1, rnn.hidden_size))

            for t in reversed(range(seq_len)):
                y_pred = outputs[t]
                y_true_idx = targets[t]

                # dL/dy
                dy = y_pred.copy()
                dy[0, y_true_idx] -= 1  # softmax + cross-entropy gradient

                # output layer grads
                dW_hy += hidden_states[t].T @ dy
                db_y += dy

                # backprop into h
                dh = dy @ rnn.W_hy.T + dh_next
                dh_raw = (1 - hidden_states[t] ** 2) * dh  # tanh'

                db_h += dh_raw
                dW_xh += inputs[t].T @ dh_raw
                if t > 0:
                    dW_hh += hidden_states[t - 1].T @ dh_raw
                dh_next = dh_raw @ rnn.W_hh.T

            # Clip gradients to prevent exploding gradients
            for dparam in [dW_xh, dW_hh, db_h, dW_hy, db_y]:
                np.clip(dparam, -5, 5, out=dparam)

            # Update weights
            rnn.W_xh -= lr * dW_xh
            rnn.W_hh -= lr * dW_hh
            rnn.b_h  -= lr * db_h
            rnn.W_hy -= lr * dW_hy
            rnn.b_y  -= lr * db_y

        print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")


def generate(rnn, start_seq, char2idx, idx2char, length=10):
    vocab_size = len(char2idx)
    input_seq = [char2idx[ch] for ch in start_seq]
    inputs = [one_hot(i, vocab_size) for i in input_seq]

    hidden_states = rnn.forward(inputs)
    h = hidden_states[-1]

    result = start_seq
    current_idx = input_seq[-1]

    for _ in range(length):
        x = one_hot(current_idx, vocab_size)
        h = np.tanh(x @ rnn.W_xh + h @ rnn.W_hh + rnn.b_h)  # reuse forward logic
        y = rnn.output(h)
        current_idx = np.random.choice(vocab_size, p=y.ravel())
        result += idx2char[current_idx]

    return result