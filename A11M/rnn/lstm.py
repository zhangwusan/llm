from .base import RnnBase
import numpy as np

class LSTM(RnnBase):

    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__(input_size, hidden_size)
        self.output_size = output_size
        self.reset_parameters()

    def forward(self, inputs: list[np.ndarray], h0: np.ndarray = None) -> list[np.ndarray]:
        H = self.hidden_size
        if h0 is None:
            h_t = np.zeros((1, H))
        else:
            h_t = h0
        c_t = np.zeros((1, H))  # initial cell state

        self.last_hidden_states = []
        for x_t in inputs:
            # Forget gate
            f_t = self.sigmoid(x_t @ self.W_f + h_t @ self.U_f + self.b_f)
            # Input gate
            i_t = self.sigmoid(x_t @ self.W_i + h_t @ self.U_i + self.b_i)
            # Cell candidate
            g_t = self.tanh(x_t @ self.W_g + h_t @ self.U_g + self.b_g)
            # Update cell state
            c_t = f_t * c_t + i_t * g_t
            # Output gate
            o_t = self.sigmoid(x_t @ self.W_o + h_t @ self.U_o + self.b_o)
            # Update hidden state
            h_t = o_t * self.tanh(c_t)

            self.last_hidden_states.append(h_t)
        return self.last_hidden_states
    
    def predict(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        hs = self.forward(inputs)
        ys = [h @ self.W_out + self.b_out for h in hs]
        return ys
    
    def compute_loss(self, outputs: list[np.ndarray], targets: list[np.ndarray]) -> float:
        return 0.5 * sum(np.sum((y - t) ** 2) for y, t in zip(outputs, targets)) / len(outputs)

    def train_step(self, inputs, targets, lr=0.01):
        preds = self.predict(inputs)
        loss = self.compute_loss([preds[-1]], [targets[-1]])
        self.update_output_weights(inputs, targets, lr)
        return loss
    
    def update_output_weights(self, inputs: list[np.ndarray], targets: list[np.ndarray], lr=0.01):
        hs = self.forward(inputs)
        ys = [h @ self.W_out + self.b_out for h in hs]

        grad_W_out = np.zeros_like(self.W_out)
        grad_b_out = np.zeros_like(self.b_out)

        # Only consider the last output for gradient and loss
        h_last = hs[-1]   # shape (1, hidden_size)
        y_last = ys[-1]   # shape (1, output_size)

        batch_size = len(inputs)
        
        # Ensure target shape matches output shape
        t = targets[-1].reshape(y_last.shape) if targets[-1].shape != y_last.shape else targets[-1]

        dy = y_last - t

        grad_W_out = h_last.T @ dy
        grad_b_out = dy

        # Average gradients over batch (batch size is 1 here because one input sequence at a time)
        grad_W_out /= batch_size
        grad_b_out /= batch_size

        self.W_out -= lr * grad_W_out
        self.b_out -= lr * grad_b_out

        loss = 0.5 * np.sum((y_last - t)**2)
        return loss
    
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)
    
    def reset_parameters(self):
        H, I, O = self.hidden_size, self.input_size, self.output_size

        # Input weights (W_x) and hidden weights (W_h) for all gates (f, i, g, o) standard normal distribution
        # Forget Gate
        self.W_f = np.random.randn(I, H) * 0.1 # to scale 
        self.U_f = np.random.randn(H, H) * 0.1
        self.b_f = np.zeros((1, H)) # one dimension

        # Input Gate
        self.W_i = np.random.randn(I, H) * 0.1
        self.U_i = np.random.randn(H, H) * 0.1
        self.b_i = np.zeros((1, H))

        self.W_g = np.random.randn(I, H) * 0.1
        self.U_g = np.random.randn(H, H) * 0.1
        self.b_g = np.zeros((1, H))

        # Output Gate
        self.W_o = np.random.randn(I, H) * 0.1
        self.U_o = np.random.randn(H, H) * 0.1
        self.b_o = np.zeros((1, H))

        # Output layer
        self.W_out = np.random.randn(H, O) * 0.1
        self.b_out = np.zeros((1, O))
