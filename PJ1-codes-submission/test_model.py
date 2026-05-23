import mynn as nn
import numpy as np
from struct import unpack
import gzip

USE_CNN = False
USE_MOMENTUM = False

if USE_CNN:
    model = nn.models.Model_CNN()
    base_dir = r'./codes/best_models/cnn'
else:
    model = nn.models.Model_MLP()
    base_dir = r'./codes/best_models/mlp'

if USE_MOMENTUM:
    model_path = base_dir + r'_momentum/best_model.pickle'
else:
    model_path = base_dir + r'/best_model.pickle'

model.load_model(model_path)
print(f'Loaded model from: {model_path}')

test_images_path = r'./codes/dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = r'./codes/dataset/MNIST/t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)

with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()

if USE_CNN:
    test_imgs = test_imgs.reshape(-1, 1, 28, 28)

logits = model(test_imgs)
print(f'Test accuracy: {nn.metric.accuracy(logits, test_labs)}')
