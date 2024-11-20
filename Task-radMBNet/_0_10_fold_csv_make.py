import os, csv
import numpy as np


# 10折交叉验证的csv准备, 为每一折准备 train val test, 保证每个被试参与测试


#----------------------------------------------------------------------------

# root = r'C:\Users\yy854\Desktop\winter\3_CNNs for image classification_python\3D\data'
# roots = r'./data/out_datas/EDSD'
roots = r'./data'
root = r'./data'
csv_path = os.path.join(root + '/' + '0_CSVs')

# name是与 类别文件夹名称 一致的
type_1_name = 'AD'
type_1_label = 0

type_2_name = 'NC'
type_2_label = 1


ROIs = ['Baselineimage.nii.gz']
firstROIS =['firsttime.nii.gz']
basemat =['baseline_feas.mat']
firstmat =['first_feas.mat']

#----------------------------------------------------------------------------
# 创建文件夹
if not os.path.exists(csv_path):
    os.mkdir(csv_path)
for i in range(1,11):
    if not os.path.exists(csv_path + '/' + str(i)):
        os.mkdir(csv_path + '/' + str(i))
#----------------------------------------------------------------------------


# for type 1
type_1_list = []
for i in os.listdir(os.path.join(roots + '/' + type_1_name)):
    type_1_list_for_one_sub = []
    for ii in os.listdir(os.path.join(roots + '/' + type_1_name + '/' + i)):
        for iii in ROIs:
            if  ii == 'feas1.mat':
                type_1_list_for_one_sub.append(os.path.join(roots + '/' + type_1_name + '/' + i + '/' + ii))
    for ii in os.listdir(os.path.join(roots + '/' + type_1_name + '/' + i)):
        for iii in ROIs:
            if ii == 'baseline_feas1.mat':
                type_1_list_for_one_sub.append(os.path.join(roots + '/' + type_1_name + '/' + i + '/' + ii))
    for ii in os.listdir(os.path.join(roots + '/' + type_1_name + '/' + i)):
        for iii in ROIs:
            if ii == 'gwbaseline.nii':
                type_1_list_for_one_sub.append(os.path.join(roots + '/' + type_1_name + '/' + i + '/' + ii))
    for ii in os.listdir(os.path.join(roots + '/' + type_1_name + '/' + i)):
        for iii in ROIs:
            if ii == 'linear1.mat':
                type_1_list_for_one_sub.append(os.path.join(roots + '/' + type_1_name + '/' + i + '/' + ii))

    type_1_list.append(type_1_list_for_one_sub)


all_path_with_label_for_type_1 = [] # 在string后面添加其对应label
for j in type_1_list:
    j.append(str(type_1_label))
    all_path_with_label_for_type_1.append(j)
type_1_amount = len(all_path_with_label_for_type_1)
all_path_with_label_for_type_1 = np.array(all_path_with_label_for_type_1)


# for type 2
type_2_list = []
for i in os.listdir(os.path.join(roots + '/' + type_2_name)):
    type_2_list_for_one_sub = []
    for ii in os.listdir(os.path.join(roots + '/' + type_2_name + '/' + i)):
        for iii in ROIs:
            if  ii == 'feas1.mat':
                type_2_list_for_one_sub.append(os.path.join(roots + '/' + type_2_name + '/' + i + '/' + ii))
    for ii in os.listdir(os.path.join(roots + '/' + type_2_name + '/' + i)):
        for iii in ROIs:
            if ii == 'baseline_feas1.mat':
                type_2_list_for_one_sub.append(os.path.join(roots + '/' + type_2_name + '/' + i + '/' + ii))
    for ii in os.listdir(os.path.join(roots + '/' + type_2_name + '/' + i)):
        for iii in ROIs:
            if ii == 'gwbaseline.nii':
                type_2_list_for_one_sub.append(os.path.join(roots + '/' + type_2_name + '/' + i + '/' + ii))
    for ii in os.listdir(os.path.join(roots + '/' + type_2_name + '/' + i)):
        for iii in ROIs:
            if ii == 'linear1.mat':
                type_2_list_for_one_sub.append(os.path.join(roots + '/' + type_2_name + '/' + i + '/' + ii))

    type_2_list.append(type_2_list_for_one_sub)


all_path_with_label_for_type_2 = [] # 在string后面添加其对应label
for j in type_2_list:
    j.append(str(type_2_label))
    all_path_with_label_for_type_2.append(j)
type_2_amount = len(all_path_with_label_for_type_2)
all_path_with_label_for_type_2 = np.array(all_path_with_label_for_type_2)


# for type 3
# type_3_list = []
# for i in os.listdir(os.path.join(root + '/' + type_3_name)):
#     type_3_list_for_one_sub = []
#     for ii in os.listdir(os.path.join(root + '/' + type_3_name+ '/' + i)):
#         for iii in ROIs:
#             if ii == 'baseline_feas.mat'or ii == 'first_feas.mat' or ii=='second_feas.mat':
#                 type_3_list_for_one_sub.append(os.path.join(root + '/' + type_3_name + '/' + i + '/' + ii))
#     type_3_list.append(type_3_list_for_one_sub)


# all_path_with_label_for_type_3 = [] # 在string后面添加其对应label
# for j in type_3_list:
#     j.append(str(type_3_label))
#     all_path_with_label_for_type_3.append(j)
# type_3_amount = len(all_path_with_label_for_type_3)
# all_path_with_label_for_type_3 = np.array(all_path_with_label_for_type_3)


# ----------------------------------------------------------------------------


# 从全部的ndarray中删除目标ndarray
def remove_array(All_array, delete_array):
    length_all = All_array.shape[0]
    length_delete_array = delete_array.shape[0]

    All_array_list = All_array.tolist()
    delete_array_list = delete_array.tolist()
    index_list = []
    for i in range(length_all):
        s = All_array_list[i]
        for ii in range(length_delete_array):
            ss = delete_array_list[ii]
            if s == ss:
                index_list.append(i)

    rest_array = np.delete(All_array, index_list, axis=0)
    return rest_array


# ----------------------------------------------------------------------------


# 10 折交叉验证
for i in range(10):

# for type 1
    type_1_test = all_path_with_label_for_type_1[
                  int(0.1 * i * type_1_amount):int(0.1 * (i+1) * type_1_amount),:]
    if i < 9:
        type_1_val = all_path_with_label_for_type_1[
                      int(0.1 * (i+1) * type_1_amount):int(0.1 * (i+2) * type_1_amount),:]
    else:
        type_1_val = all_path_with_label_for_type_1[:int(0.1  * type_1_amount), :]
    type_1_train_val = remove_array(all_path_with_label_for_type_1, type_1_test)
    type_1_train = remove_array(type_1_train_val, type_1_val)


# for type 2
    type_2_test = all_path_with_label_for_type_2[
                  int(0.1 * i * type_2_amount):int(0.1 * (i+1) * type_2_amount),:]
    if i < 9:
        type_2_val = all_path_with_label_for_type_2[
                     int(0.1 * (i + 1) * type_2_amount):int(0.1 * (i + 2) * type_2_amount), :]
    else:
        type_2_val = all_path_with_label_for_type_2[:int(0.1 * type_2_amount), :]
    type_2_train_val = remove_array(all_path_with_label_for_type_2, type_2_test)
    type_2_train = remove_array(type_2_train_val, type_2_val)


# for type 3
#     type_3_test = all_path_with_label_for_type_3[
#                   int(0.1 * i * type_3_amount):int(0.1 * (i+1) * type_3_amount),:]
#     if i < 9:
#         type_3_val = all_path_with_label_for_type_3[
#                      int(0.1 * (i + 1) * type_3_amount):int(0.1 * (i + 2) * type_3_amount), :]
#     else:
#         type_3_val = all_path_with_label_for_type_3[:int(0.1 * type_3_amount), :]
#     type_3_train_val = remove_array(all_path_with_label_for_type_3, type_3_test)
#     type_3_train = remove_array(type_3_train_val, type_3_val)


# ----------------------------------------------------------------------------


# 拼接数据并打乱引入随机性
# for 2 types
    train = np.concatenate((type_1_train, type_2_train), axis=0)
    np.random.shuffle(train)

    val = np.concatenate((type_1_val, type_2_val), axis=0)
    np.random.shuffle(val)

    test = np.concatenate((type_1_test, type_2_test), axis=0)
    np.random.shuffle(test)

# for 3 types
#     train = np.concatenate((type_1_train, type_2_train, type_3_train),axis=0)
#     np.random.shuffle(train)
#
#     val = np.concatenate((type_1_val, type_2_val, type_3_val),axis=0)
#     np.random.shuffle(val)
#
#     test = np.concatenate((type_1_test, type_2_test, type_3_test),axis=0)
#     np.random.shuffle(test)


# ----------------------------------------------------------------------------


# train, write into csv
    csv_path_for_one_fold = os.path.join(csv_path + '/' + str(i + 1))
    file_name_train = str(i + 1) + '_train.csv'
    ROIs_with_single_label_amount = train.shape[1]

    with open(os.path.join(csv_path_for_one_fold, file_name_train), mode='w', newline='') as f:
        writer = csv.writer(f)
        for line in train:
            a = [line[i] for i in range(ROIs_with_single_label_amount)]
            writer.writerow(a)


# val, write into csv
    file_name_val = str(i + 1) + '_val.csv'
    with open(os.path.join(csv_path_for_one_fold, file_name_val), mode='w', newline='') as f:
        writer = csv.writer(f)
        for line in val:
            a = [line[i] for i in range(ROIs_with_single_label_amount)]
            writer.writerow(a)


# test, write into csv
    file_name_test = str(i + 1) + '_test.csv'
    with open(os.path.join(csv_path_for_one_fold, file_name_test), mode='w', newline='') as f:
        writer = csv.writer(f)
        for line in test:
            a = [line[i] for i in range(ROIs_with_single_label_amount)]
            writer.writerow(a)

    print(i+1, 'fold over')
print('\nOver file generation for 10 fold cross validation, pls check it out.')
