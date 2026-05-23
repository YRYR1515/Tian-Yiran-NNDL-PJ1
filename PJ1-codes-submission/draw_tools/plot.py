# plot the score and loss
import matplotlib.pyplot as plt
import numpy as np

colors_set = {'Kraftime' : ('#E3E37D', '#968A62')}

def plot(runner, axes, set=colors_set['Kraftime']):
    train_color = set[0]
    dev_color = set[1]

    n_train = len(runner.train_scores)
    n_dev = len(runner.dev_scores)
    train_x = np.arange(n_train)
    # dev 点稀疏，把它均匀铺到和 train 同一段 iteration 区间上
    dev_x = np.linspace(0, max(n_train - 1, 0), n_dev) if n_dev > 0 else np.array([])

    # 绘制训练损失变化曲线
    axes[0].plot(train_x, runner.train_loss, color=train_color, label="Train loss")
    # 绘制评价损失变化曲线
    axes[0].plot(dev_x, runner.dev_loss, color=dev_color, linestyle="--", label="Dev loss")
    # 绘制坐标轴和图例
    axes[0].set_ylabel("loss")
    axes[0].set_xlabel("iteration")
    axes[0].set_title("")
    axes[0].legend(loc='upper right')
    # 绘制训练准确率变化曲线
    axes[1].plot(train_x, runner.train_scores, color=train_color, label="Train accuracy")
    # 绘制评价准确率变化曲线
    axes[1].plot(dev_x, runner.dev_scores, color=dev_color, linestyle="--", label="Dev accuracy")
    # 绘制坐标轴和图例
    axes[1].set_ylabel("score")
    axes[1].set_xlabel("iteration")
    axes[1].legend(loc='lower right')