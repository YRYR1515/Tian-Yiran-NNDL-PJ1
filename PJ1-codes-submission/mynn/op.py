from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
    
        scale = np.sqrt(2.0 / in_dim)
        self.W = initialize_method(size=(in_dim, out_dim)) * scale
        self.b = np.zeros((1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return X @ self.params['W'] + self.params['b']

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.params['W'].T
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size)) * scale
        self.b = np.zeros((1, out_channels, 1, 1))

        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

        self.input = None
        self.input_padded = None

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [out, in, k, k]
        """
        self.input = X
        n, c, h, w = X.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        if p > 0:
            X_pad = np.pad(X, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
        else:
            X_pad = X
        self.input_padded = X_pad

        h_out = (h + 2 * p - k) // s + 1
        w_out = (w + 2 * p - k) // s + 1

        out = np.zeros((n, self.out_channels, h_out, w_out))
        W = self.params['W']
        b = self.params['b']

        for i in range(h_out):
            hs = i * s
            he = hs + k
            for j in range(w_out):
                ws = j * s
                we = ws + k
                x_slice = X_pad[:, :, hs:he, ws:we]  # [n, in, k, k]
                for oc in range(self.out_channels):
                    out[:, oc, i, j] = np.sum(x_slice * W[oc][None, :, :, :], axis=(1, 2, 3))

        out += b
        return out

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        X = self.input
        X_pad = self.input_padded
        W = self.params['W']

        n, c, h, w = X.shape
        _, _, h_out, w_out = grads.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        dW = np.zeros_like(W)
        db = np.sum(grads, axis=(0, 2, 3), keepdims=True).reshape(1, self.out_channels, 1, 1)
        dX_pad = np.zeros_like(X_pad)

        for i in range(h_out):
            hs = i * s
            he = hs + k
            for j in range(w_out):
                ws = j * s
                we = ws + k
                x_slice = X_pad[:, :, hs:he, ws:we]  # [n, in, k, k]

                for oc in range(self.out_channels):
                    grad_ij = grads[:, oc, i, j]  # [n]
                    dW[oc] += np.sum(x_slice * grad_ij[:, None, None, None], axis=0)
                    dX_pad[:, :, hs:he, ws:we] += W[oc][None, :, :, :] * grad_ij[:, None, None, None]

        if p > 0:
            dX = dX_pad[:, :, p:-p, p:-p]
        else:
            dX = dX_pad

        self.grads['W'] = dW
        self.grads['b'] = db
        return dX

    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model=None, max_classes=10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.predicts = None
        self.labels = None
        self.grads = None
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.labels = labels
        batch_size = predicts.shape[0]

        if self.has_softmax:
            self.predicts = softmax(predicts)
        else:
            self.predicts = predicts

        probs = np.clip(self.predicts, 1e-12, 1.0)
        correct_probs = probs[np.arange(batch_size), labels]
        loss = -np.sum(np.log(correct_probs)) / batch_size
        return loss

    def backward(self):
        batch_size = self.predicts.shape[0]
        # Gradient of softmax + cross-entropy combined: softmax(x) - one_hot(y)
        self.grads = self.predicts.copy()
        self.grads[np.arange(batch_size), self.labels] -= 1
        self.grads /= batch_size
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition