# 导入使用到的外部库
import os, math, time, random, csv
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data


# 导入使用到的内部python函数
from _1_dataloaders import train_Dataset, val_Dataset, test_Dataset
from _0_metrics_for_binary_classification import metrics_for_binary_classification
from Task_radMBNet import GCNGenerator


#----------------------------------------------------------------------------
                        # hyper parameter definition
#----------------------------------------------------------------------------


root = r'./data'
csv_path = root + '/' + '0_CSVs'
parameters_path = root + '/' + '1_Parameters'
log_path = root + '/' + '2_logs'
result_path = root + '/' + '3_Results'
Result_file_name = 'Result_binary_classification.csv'


batch_size = 4# 训练时, 一次取多少数据进行训练
epochs = 100# 训练轮次
epoch_divider = 1 # 训练时, 经多少个epoch即验证
lr = 3e-4 # 学习率
weight_decay_factor = 1e-4 # 权重衰减率
Time = 10 # 10折交叉验证
type_1_label = 0 # 类别1的标签
type_2_label = 1 # 类别2的标签
patient = 30 # 早停法的耐心值
seed = 123 # 随机种子数
GPU_use = 0 # 选择使用哪一块GPU
device = torch.device('cuda:{}'.format(str(GPU_use)) if torch.cuda.is_available() else 'cpu') # 选择使用哪块GPU或使用CPU


#----------------------------------------------------------------------------


# 固定随机种子, 使得结果得以复现
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


# 创建参数和结果文件夹
if not os.path.exists(parameters_path):
    os.makedirs(parameters_path)
if not os.path.exists(result_path):
    os.makedirs(result_path)


# 评估函数
def evaluate_fuc(model, loader):
    preds, labels, scores = [], [], []
    for (a, b, c, d, e, f,g) in loader:
        node_feature = a.to(device)
        mat_pcc = b.to(device)
        mat_topo = c.to(device)
        label = f.float().to(device)
        with torch.no_grad():
            logits = model(node_feature, mat_pcc, mat_topo,device)
            label= label.float()
            threshold =0.3186 #yuzhi
            pred =logits.ge(threshold).float()
            # pred = logits.argmax(dim=1)
            for p in range(len(label)):
                preds.append(pred[p])
                labels.append(label[p])
                y_score = logits[p]#eryuan
                # y_score = logits[p][1]
                scores.append(y_score)
    return preds, labels, scores


# 计算模型参数
def compute_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


#----------------------------------------------------------------------------


def main():
    # 计算程序总运行时间
    tik = time.time()

    test__acc, test__sen, test__spe, test__auc = [], [], [], []
    test__precision, test__balanced_acc, test__recall, test__F1_score = [], [], [], []
    test__FNR, test__FPR, test__TPR, test__TNR = [], [], [], []

    # GPU config
    # 显示 GPU 是否可用
    GPU_available = torch.cuda.is_available()
    print('If the GPU is available?' + str(GPU_available))
    # 显示 GPU 数量
    GPU_num = torch.cuda.device_count()
    print('Amount of GPU: ' + str(GPU_num))
    # 显示 每块GPU名称
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('GPU #' + str(GPU_idx) + ': ' + GPU_name)
    # 显示 当前使用的GPU
    current_GPU_name = torch.cuda.get_device_name(GPU_use)
    print('Currently using GPU: {}, {}'.format(GPU_use, current_GPU_name))


# ----------------------------------------------------------------------------
    for fold in range(1, Time + 1):
        print('\nCurrent fold:', fold)

        # 日志文件
        loss_log_name = 'loss_log_epoch_'+str(epochs) + '_initial_lr_'+str(lr) + '_fold_'+str(fold)
        print('loss_log_name: ', loss_log_name)
        loss_f = open(os.path.join(log_path, loss_log_name + '.txt'), 'w')

        # load train data
        train_dataset = train_Dataset(csv_path, fold)
        print('Number of training images:', len(train_dataset))
        train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

        # load validation data
        val_dataset = val_Dataset(csv_path, fold)
        print('Number of valuation images:', len(val_dataset))
        val_loader = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

        steps_everyepoch = len(train_dataset)/batch_size
        step_all = steps_everyepoch * epochs

        # 创建分类网络，设置损失函数，优化器和学习率迭代衰减
        model = GCNGenerator(90,90,90).to(device)
        # criterion = nn.CrossEntropyLoss().to(device)

        criterion = nn.BCELoss().to(device)#defen score
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay_factor, amsgrad=True)
        lrScheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(step_all))

        # 显示模型参数量
        print('Amount of model parameters:', compute_parameters(model))


        # ----------------------------------------------------------------------------
        # train start
        best_acc, best_epoch = 0, 0
        for epoch in range(1, epochs+1):
            model.train()
            for batch_idx,(a,b,c,d,e,f) in enumerate(train_loader):
                optimizer.zero_grad()
                # 将图像和label放在cuda上执行
                node_feature = a.to(device)
                mat_pcc = b.to(device)
                label = f.float().to(device)


                pred_label = model(node_feature,mat_pcc,device)

                loss = criterion(pred_label, label)

                # 梯度清零与反向传播

                loss.backward()
                optimizer.step()
                lrScheduler.step()

                # 检测训练过程
                print('Time:[{}/{}]---Epoch:[{}/{}]---Batch_index:[{}/{}]---'.format(fold, Time, epoch, epochs, batch_idx + 1, math.ceil(steps_everyepoch)), end="")
                print('Loss:{:.6f}'.format(loss.item()), flush=True, end="")
                print('---Current_lr:', lrScheduler.get_lr())


            # ----------------------------------------------------------------------------
            # validation start
            # 每隔epoch_divider个epoch就验证一次，以将最好的模型保存下来
            if epoch % epoch_divider == 0:
                model.eval()
                val_pred, val_labels, val_scores = evaluate_fuc(model, val_loader)

                val_pred = torch.tensor(val_pred)
                print("val_red:", val_pred)
                val_labels = torch.tensor(val_labels)
                print("val_labels:", val_labels)
                val_scores = torch.tensor(val_scores)

                # 计算验证集的准确率, 通过val的 最高acc 来保存最好模型参数
                val_acc,_,_,_\
                    ,_,_,_,_\
                    ,_,_,_,_ = metrics_for_binary_classification(
                    type_1_label, type_2_label, val_pred, val_labels, val_scores)

                if val_acc > best_acc:
                    best_epoch = epoch
                    best_acc = val_acc

                    # 保存最好的模型和优化器
                    save_model_name = os.path.join(parameters_path, str(fold) + '_model_best.pth')
                    torch.save(model.state_dict(), save_model_name)
                    save_optimizer_name = os.path.join(parameters_path, str(fold) + '_optimizer_best.pth')
                    torch.save(optimizer.state_dict(), save_optimizer_name)
                # ----------------------------------------------------------------------------
                # 早停法, 即得到最高acc的轮次后, 再训练多少轮次, 如果还没提升最高acc, 就停止训练, 节省时间
                if epoch - best_epoch > patient:
                    if val_acc < best_acc:
                        break
        # 打印最好acc时的验证准确率和训练轮次
        print('\nvaluation best acc:{:.2f}%'.format(100 * best_acc), 'best epoch:', best_epoch)


        # ----------------------------------------------------------------------------
        # test start
        # load test data
        test_dataset = test_Dataset(csv_path, fold)
        print('Number of test images:', len(test_dataset))
        test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

        print('\nloading checkpoint...')
        model.load_state_dict(torch.load(save_model_name))
        optimizer.load_state_dict(torch.load(save_optimizer_name))

        model.eval()
        test_pred, test_labels, test_scores = evaluate_fuc(model, test_loader)

        test_pred = torch.tensor(test_pred)
        test_labels = torch.tensor(test_labels)
        test_scores = torch.tensor(test_scores)

        test_acc, test_sen, test_spe, test_auc, \
        test_precision, test_balanced_accuracy, test_recall, test_F1_score, \
        test_FNR, test_FPR, test_TPR, test_TNR = metrics_for_binary_classification(
            type_1_label, type_2_label, test_pred, test_labels, test_scores)


# ----------------------------------------------------------------------------
# write a csv to record the metrics
        test__acc.append(test_acc)
        test__sen.append(test_sen)
        test__spe.append(test_spe)
        test__auc.append(test_auc)

        test__precision.append(test_precision)
        test__balanced_acc.append(test_balanced_accuracy)
        test__recall.append(test_recall)
        test__F1_score.append(test_F1_score)

        test__FNR.append(test_FNR)
        test__FPR.append(test_FPR)
        test__TPR.append(test_TPR)
        test__TNR .append(test_TNR)


        lines = [[a,b,c,d, e,f,g,h, u,v,w,z] for a,b,c,d, e,f,g,h, u,v,w,z in
                 zip(test__acc, test__sen, test__spe, test__auc,
                     test__precision, test__balanced_acc, test__recall, test__F1_score,
                     test__FNR, test__FPR, test__TPR, test__TNR)]

        with open(os.path.join(result_path, Result_file_name), mode='w', newline='') as ff:
            writer = csv.writer(ff)
            writer.writerow(['\t', 'Acc\t', 'Sen\t', 'Spe\t', 'Auc\t',
                             'precision\t', 'balanced_acc\t', 'recall\t', 'F1_score\t',
                             'FNR\t', 'FPR\t', 'TPR\t', 'TNR\t'])
            for idxc, (line) in enumerate(lines):
                writer.writerow([idxc+1,
                                 '{:.4f}\t'.format(line[0]), '{:.4f}\t'.format(line[1]),
                                 '{:.4f}\t'.format(line[2]), '{:.4f}\t'.format(line[3]),
                                 '{:.4f}\t'.format(line[4]), '{:.4f}\t'.format(line[5]),
                                 '{:.4f}\t'.format(line[6]), '{:.4f}\t'.format(line[7]),
                                 '{:.4f}\t'.format(line[8]), '{:.4f}\t'.format(line[9]),
                                 '{:.4f}\t'.format(line[10]), '{:.4f}\t'.format(line[11])])
            writer.writerow(['Avg\t',
                             '{:.4f}\t'.format(np.average(test__acc)), '{:.4f}\t'.format(np.average(test__sen)),
                             '{:.4f}\t'.format(np.average(test__spe)), '{:.4f}\t'.format(np.average(test__auc)),
                             '{:.4f}\t'.format(np.average(test__precision)), '{:.4f}\t'.format(np.average(test__balanced_acc)),
                             '{:.4f}\t'.format(np.average(test__recall)), '{:.4f}\t'.format(np.average(test__F1_score)),
                             '{:.4f}\t'.format(np.average(test__FNR)), '{:.4f}\t'.format(np.average(test__FPR)),
                             '{:.4f}\t'.format(np.average(test__TPR)), '{:.4f}\t'.format(np.average(test__TNR))])
            writer.writerow(['Std\t',
                             '{:.4f}\t'.format(np.std(test__acc)), '{:.4f}\t'.format(np.std(test__sen)),
                             '{:.4f}\t'.format(np.std(test__spe)), '{:.4f}\t'.format(np.std(test__auc)),
                             '{:.4f}\t'.format(np.std(test__precision)), '{:.4f}\t'.format(np.std(test__balanced_acc)),
                             '{:.4f}\t'.format(np.std(test__recall)), '{:.4f}\t'.format(np.std(test__F1_score)),
                             '{:.4f}\t'.format(np.std(test__FNR)), '{:.4f}\t'.format(np.std(test__FPR)),
                             '{:.4f}\t'.format(np.std(test__TPR)), '{:.4f}\t'.format(np.std(test__TNR))])

        # 关闭日志文件
        loss_f.close()

    # 计算时间
    tok = time.time()
    s = tok - tik
    m = s // 60
    h = m // 60
    rest_m = m - h * 60
    rest_s = s - m * 60
    print('Total time for the code: {}h,{}m, {:.2f}s - all {:.2f}s'.format(h, rest_m, rest_s, s))

if __name__ == '__main__':
    main()