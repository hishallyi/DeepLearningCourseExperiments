import math
import torch
from torch.utils import data
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as mse_fn, mean_absolute_error as mae_fn
import numpy as np
import time

# 设置中文支持
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def mape_fn(y, pred):
    mask = y != 0
    y = y[mask]
    pred = pred[mask]
    mape = np.abs((y - pred) / y)
    mape = np.mean(mape) * 100
    return mape


# 计算评价指标mse：均方误差；mae：平均绝对误差；mape：平均绝对百分误差
def eval(y, pred):
    y = y.cpu().numpy()
    pred = pred.cpu().numpy()
    mse = mse_fn(y, pred)
    rmse = math.sqrt(mse)
    mae = mae_fn(y, pred)
    mape = mape_fn(y, pred)
    return [rmse, mae, mape]


# 测试函数（用于分类）
def test(net, output_model, data_iter, loss_fn, denormalize_fn, device='cpu'):
    # 初始化评价指标
    rmse, mae, mape = 0, 0, 0
    batch_count = 0
    total_loss = 0.0

    # 切换为评估模型
    net.eval()

    if output_model is not None:
        output_model.eval()
    
    # 按批次测试
    for X, Y in data_iter:
        X = X.to(device).float()
        Y = Y.to(device).float()
        # 前向传播
        output, hidden = net(X)
        if output_model is not None:
            y_hat = output_model(output.squeeze()).squeeze()
        else:
            y_hat = output.squeeze()
        # 计算损失
        loss = loss_fn(y_hat, Y)

        # 逆标准化
        Y = denormalize_fn(Y)
        y_hat = denormalize_fn(y_hat)
        # 计算损失值表评价指标并累加
        a, b, c = eval(Y.detach(), y_hat.detach())
        rmse += a
        mae += b
        mape += c
        total_loss += loss.detach().cpu().numpy().tolist()
        batch_count += 1
    return [rmse / batch_count, mae / batch_count, mape / batch_count], total_loss / batch_count


# 模型训练函数（早停）
def train(net, train_iter, val_iter, test_iter, loss_fn, denormalize_fn, optimizer, num_epoch,
          early_stop=10, device='cpu', output_model=None, is_print=True, is_print_batch=False):
    train_loss_lst = []
    val_loss_lst = []
    train_score_lst = []
    val_score_lst = []
    epoch_time = []

    best_epoch = 0
    best_val_rmse = 9999
    early_stop_flag = 0

    # 按epoch训练
    for epoch in range(num_epoch):
        # 切换训练模式
        net.train()
        if output_model is not None:
            output_model.train()
        epoch_loss = 0
        batch_count = 0
        batch_time = []
        rmse, mae, mape = 0, 0, 0

        # 按批次训练
        for X, Y in train_iter:
            batch_s = time.time()
            X = X.to(device).float()
            Y = Y.to(device).float()

            # 前向传播
            output, hidden = net(X)

            if output_model is not None:
                y_hat = output_model(output.squeeze()).squeeze()
            else:
                y_hat = output.squeeze()

            # 求损失
            loss = loss_fn(y_hat, Y)

            # 反向传播
            loss.backward()

            # 参数更新
            optimizer.step()

            # 梯度清零
            optimizer.zero_grad()

            Y = denormalize_fn(Y)    # 去标准化
            y_hat = denormalize_fn(y_hat)
            # 计算该批次的评价指标并累加
            a, b, c = eval(Y.detach(), y_hat.detach())
            rmse += a
            mae += b
            mape += c
            epoch_loss += loss.detach().cpu().numpy().tolist()
            batch_count += 1
            # sample_num += X.shape[0]

            batch_time.append(time.time() - batch_s)
            if is_print and is_print_batch:
                print('epoch-batch: %d-%d, train loss %.4f, time use %.3fs' %
                      (epoch + 1, batch_count, epoch_loss, batch_time[-1]))

        # 记录损失和评价指标
        train_loss = epoch_loss / batch_count
        train_loss_lst.append(train_loss)
        train_score_lst.append([rmse/batch_count, mae/batch_count, mape/batch_count])

        # 验证集
        val_score, val_loss = test(net, output_model, val_iter, loss_fn, denormalize_fn, device)
        val_score_lst.append(val_score)
        val_loss_lst.append(val_loss)

        epoch_time.append(np.array(batch_time).sum())

        # 打印本轮训练结果
        if is_print:
            print('*** epoch%d, train loss %.4f, train rmse %.4f, val loss %.4f, val rmse %.6f, time use %.3fs' %
                  (epoch + 1, train_loss, train_score_lst[-1][0], val_loss, val_score[0], epoch_time[-1]))

        # 早停(验证集上的指标rmse总数超过10次就停止训练)
        if val_score[0] < best_val_rmse:
            best_val_rmse = val_score[0]
            best_epoch = epoch
            early_stop_flag = 0
        else:
            early_stop_flag += 1
            best_epoch = epoch    # 早停那轮epoch也需要保存
            if early_stop_flag == early_stop:
                print(f'\nThe model has not been improved for {early_stop} rounds. Stop early!')
                break

    # 输出最终训练结果
    print(f'\n{"*" * 40}\nFinal result:')
    print(f'Get best validation rmse {np.array(val_score_lst)[:, 0].min() :.4f} '
          f'at epoch {best_epoch+1}')
    print(f'Total time {np.array(epoch_time).sum():.2f}s')
    print()

    # 计算测试集效果
    test_score, test_loss = test(net, output_model, test_iter, loss_fn, denormalize_fn, device)
    print('Test result:')
    print(f'Test RMSE: {test_score[0]}    Test MAE: {test_score[1]}    Test MAPE: {test_score[2]}')
    return train_loss_lst, val_loss_lst, train_score_lst, val_score_lst, best_epoch


# 绘制训练损失和验证损失
def visualize(train_data, test_data, x_label='epoch', y_label='loss'):
    x = np.arange(0, len(train_data)).astype(dtype=np.int32)
    plt.plot(x, train_data, label=f"train_{y_label}", linewidth=1.5)
    plt.plot(x, test_data, label=f"val_{y_label}", linewidth=1.5)

    plt.title("模型训练损失和验证损失变化图")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


# 绘制评价指标变化图
def plot_metric(train_score, val_score):
    train_score = np.array(train_score)
    val_score = np.array(val_score)

    plt.figure(figsize=(10, 6), dpi=300)
    plt.suptitle('三种指标变化图')

    plt.subplot(2, 2, 1)
    plt.plot(train_score[:, 0], label='train', c='#d28ad4')
    plt.plot(val_score[:, 0], label='val', c='#33ff33')
    plt.legend()
    plt.ylabel('RMSE')

    plt.subplot(2, 2, 2)
    plt.plot(train_score[:, 1], label='train', c='#e765eb')
    plt.plot(val_score[:, 1], label='val', c='#3333ff')
    plt.legend()
    plt.ylabel('MAE')

    plt.subplot(2, 2, 3)
    plt.plot(train_score[:, 2], label='train', c='#660099')
    plt.plot(val_score[:,2], label='val', c='#ff0000')
    plt.legend()
    plt.ylabel('MAPE')

    plt.show()


# 根据列表X、Y的长度一致，如果不一致用0填充较短的列表直至一致
def fill_zero(X, Y):
    if len(X) > len(Y):
        for i in range(len(X) - len(Y)):
            Y.append(0)
    elif len(X) < len(Y):
        for i in range(len(Y) - len(X)):
            X.append(0)
    return Y


# 绘制训练时间对比图
def train_time_comparison(times, labels, title):
    plt.figure(dpi=100)
    bars = plt.bar(labels, times, color=['blue', 'green', 'red'])

    plt.title(title)
    plt.xlabel('Models')
    plt.ylabel('Training Time (seconds)')

    # 在每个柱子上方显示时间数据
    for bar, time in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, str(time), ha='center', color='black', fontsize=12)

    plt.tight_layout()
    plt.show()


# 绘制损失、指标对比图
def losses_comparison(path_list, labels):

    train_losses = []
    val_losses = []

    train_rmse_list = []
    train_mae_list = []
    train_mape_list = []

    val_rmse_list = []
    val_mae_list = []
    val_mape_list = []

    for i in range(len(path_list)):
        path = path_list[i]
        # 加载模型参数
        checkpoint = torch.load(path)

        train_loss_lst, val_loss_lst, \
            train_score_lst, val_score_lst = checkpoint['other_info']['train_losses'], checkpoint['other_info']['val_losses'], checkpoint['other_info']['train_score'], checkpoint['other_info']['val_score']
        
        train_losses.append(train_loss_lst)
        val_losses.append(val_loss_lst)

        train_rmse_list.append(row[0] for row in train_score_lst)
        train_mae_list.append(row[1] for row in train_score_lst)
        train_mape_list.append(row[2] for row in train_score_lst)

        val_rmse_list.append(row[0] for row in val_score_lst)
        val_mae_list.append(row[1] for row in val_score_lst)
        val_mape_list.append(row[2] for row in val_score_lst)
    
    # 绘制损失对比图
    plt.figure(dpi=300)
    plt.suptitle('损失变化对比图')

    x = np.arange(0, len(train_losses[-1])).astype(dtype=np.int32)

    plt.subplot(2, 1, 1)
    for i in range(len(train_losses)):
        plt.plot(x, fill_zero(x, train_losses[i]), label=labels[i], linewidth=1.5)
    plt.legend()
    plt.ylabel('Train Loss')

    plt.subplot(2, 1, 2)
    for i in range(len(val_losses)):
        plt.plot(x, fill_zero(x, val_losses[i]), label=labels[i], linewidth=1.5)
    plt.legend()
    plt.ylabel('Val Loss')
    plt.xlabel('Epoch')

    # 绘制衡量指标对比图
    plt.figure(figsize=(10, 6), dpi=300)
    plt.suptitle('指标对比图')

    x = list(range(50))

    plt.subplot(3, 2, 1)
    for i in range(len(train_rmse_list)):
        train_rmse_list[i] = list(train_rmse_list[i])
        plt.plot(x, fill_zero(x, train_rmse_list[i]), label=labels[i], linewidth=1.5)
    plt.legend()
    plt.ylabel('Train RMSE')

    plt.subplot(3, 2, 2)
    for i in range(len(val_rmse_list)):
        val_rmse_list[i] = list(val_rmse_list[i])
        plt.plot(x, fill_zero(x, val_rmse_list[i]), label=labels[i], linewidth=1.5)
    plt.legend()
    plt.ylabel('Val RMSE')

    plt.subplot(3, 2, 3)
    for i in range(len(train_mae_list)):
        train_mae_list[i] = list(train_mae_list[i])
        plt.plot(x, fill_zero(x, train_mae_list[i]), label=labels[i], linewidth=1.5)
    plt.legend()
    plt.ylabel('Train MAE')

    plt.subplot(3, 2, 4)
    for i in range(len(val_mae_list)):
        val_mae_list[i] = list(val_mae_list[i])
        plt.plot(x, fill_zero(x, val_mae_list[i]), label=labels[i], linewidth=1.5)
    plt.legend()
    plt.ylabel('Val MAE')

    plt.subplot(3, 2, 5)
    for i in range(len(train_mape_list)):
        train_mape_list[i] = list(train_mape_list[i])
        plt.plot(x, fill_zero(x, train_mape_list[i]), label=labels[i], linewidth=1.5)
    plt.legend()
    plt.ylabel('Train MAPE')
    plt.xlabel('Epoch')

    plt.subplot(3, 2, 6)
    for i in range(len(val_mape_list)):
        val_mape_list[i] = list(val_mape_list[i])
        plt.plot(x, fill_zero(x, val_mape_list[i]), label=labels[i], linewidth=1.5)
    plt.legend()
    plt.ylabel('Val MAPE')
    plt.xlabel('Epoch')