import numpy as np

from A11M.rnn.sample import RNN

# Define variable
X = [
    np.array([1, 0.8]),
    np.array([0.5, -0.7]),
    np.array([0.3, 1.2])
]

rnn = RNN(input_size=2, hidden_size=2)

states = rnn.forward(X)

print("Step-by-step hidden states:")
for t, h_t in enumerate(states):
    print(f"t={t} | h={h_t}")