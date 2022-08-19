import numpy as np
import torch
import random
def stack_uneven(arrays, fill_value=-9999.0):
        sizes = [a.shape for a in arrays]
        max_sizes = np.max(list(zip(*sizes)), -1)
        # The resultant array has stacked on the first dimension
        result = np.full((len(arrays),) + tuple(max_sizes), fill_value)
        for i, a in enumerate(arrays):
          # The shape of this array `a`, turned into slices
          slices = tuple(slice(0,s) for s in sizes[i])
          # Overwrite a block slice of `result` with this array `a`
          result[i][slices] = a
        return result
    
def data_iter(data, batch_size=10, device="cpu"):
    num_examples=len(data)
    idx=np.arange(num_examples)
    np.random.shuffle(idx)
    for i in range(0, num_examples, batch_size):
        d=[data[i][:, 0:-1] for i in idx[i:min(i+10, num_examples)]]
        l=[data[i][:, -1] for i in idx[i:min(i+10, num_examples)]]
        length=np.array([e.shape[1] for e in d])
        sorted_index=np.argsort(-length)
        sorted_length=length[sorted_index]
        yield torch.tensor(stack_uneven(d)[sorted_index]).float().to(device), torch.tensor(np.stack(l))[:,0].float().to(device), torch.tensor(sorted_length).to(device)

def normalize_and_split(data_list, train_ratio=0.8, val_ratio_of_val_test=0.8):
    # normalize elevation within individual sites
    for i, a in enumerate(data_list):
        min_h=a[3][0].min()
        max_h=a[3][0].max()
        normalized=(a[3][0]-min_h)/(max_h-min_h)
        data_list[i][3]=np.repeat(normalized[None, :], a.shape[1], axis=0)
    
    # shuffle the dataset before spliting into train, val, and test
    random.shuffle(data_list)
    
    # split dataset
    num_train=int(len(data_list)*train_ratio)
    num_val=int((len(data_list)-num_train) * val_ratio_of_val_test)
    num_test=len(data_list)-num_train-num_val
    
    train_data=data_list[0:num_train]
    val_data=data_list[num_train:num_train+num_val]
    test_data=data_list[num_train+num_val:]
    
    # normalize u10
    max_u10=max([a[4].max() for a in train_data])
    min_u10=min([a[4].min() for a in train_data])
    for i, a in enumerate(train_data):
        train_data[i][4]=(train_data[i][4]-min_u10)/(max_u10-min_u10)
    for i, a in enumerate(val_data):
        val_data[i][4]=(val_data[i][4]-min_u10)/(max_u10-min_u10)
    for i, a in enumerate(test_data):
        test_data[i][4]=(test_data[i][4]-min_u10)/(max_u10-min_u10)
        
    # normalize v10
    max_v10=max([a[5].max() for a in train_data])
    min_v10=min([a[5].min() for a in train_data])
    for i, a in enumerate(train_data):
        train_data[i][5]=(train_data[i][5]-min_v10)/(max_v10-min_v10)
    for i, a in enumerate(val_data):
        val_data[i][5]=(val_data[i][5]-min_v10)/(max_v10-min_v10)
    for i, a in enumerate(test_data):
        test_data[i][5]=(test_data[i][5]-min_v10)/(max_v10-min_v10)
        
    # normalize t2m
    max_temp=max([a[6].max() for a in train_data])
    min_temp=min([a[6].min() for a in train_data])
    for i, a in enumerate(train_data):
        train_data[i][6]=(train_data[i][6]-min_temp)/(max_temp-min_temp)
    for i, a in enumerate(val_data):
        val_data[i][6]=(val_data[i][6]-min_temp)/(max_temp-min_temp)
    for i, a in enumerate(test_data):
        test_data[i][6]=(test_data[i][6]-min_temp)/(max_temp-min_temp)
    
    return num_train, num_val, num_test, train_data, val_data, test_data
        