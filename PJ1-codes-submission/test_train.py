# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

np.random.seed(309)

USE_CNN = True 
USE_MOMENTUM = True  

train_images_path = r'.\codes\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\codes\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

if USE_CNN:
    train_imgs = train_imgs.reshape(-1, 1, 28, 28)
    valid_imgs = valid_imgs.reshape(-1, 1, 28, 28)
    model = nn.models.Model_CNN()
    init_lr = 0.06
    base_dir = r'./codes/best_models/cnn'
else:
    model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', None)
    init_lr = 0.06
    base_dir = r'./codes/best_models/mlp'

if USE_MOMENTUM:
    optimizer = nn.optimizer.MomentGD(init_lr=init_lr, model=model, mu=0.9)
    save_dir = base_dir + '_momentum'
else:
    optimizer = nn.optimizer.SGD(init_lr=init_lr, model=model)
    save_dir = base_dir
scheduler = nn.lr_scheduler.StepLR(optimizer=optimizer, step_size=1500, gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=train_labs.max()+1)

runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, batch_size=32, scheduler=scheduler)

runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=100, save_dir=save_dir)

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

plt.show()