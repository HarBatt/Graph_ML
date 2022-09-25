import torch
import numpy as np
import pandas as pd
from shared.component_logger import component_logger as logger
from torch.utils.data import DataLoader, Subset
import random

def fill_nan_values(data_frame, statistical_flag=False, fill_nan="bfill"):
    if statistical_flag:
        data_frame = data_frame.replace([np.inf, 0, -np.inf], np.nan)
    else:
        data_frame = data_frame.replace([np.inf, -np.inf], np.nan)

    data_frame = data_frame.apply(pd.to_numeric, errors='coerce')
    if fill_nan == "fill_mean":
        data_frame = data_frame.replace(np.nan, data_frame.mean())
    if fill_nan == "fill_zero":
        data_frame = data_frame.replace(np.nan, 0)
    if fill_nan == "bfill":
        data_frame = data_frame.fillna(method='bfill', inplace=False)
    if fill_nan == "ffill":
        data_frame = data_frame.fillna(method='ffill', inplace=False)
    if fill_nan == "interpolate":
        data_frame = data_frame.interpolate(method='linear',
                                            inplace=False)
    return data_frame


def downsample(data_frame, down_len, labels):
    np_data = np.array(data_frame)
    orig_len, col_num = np_data.shape
    down_time_len = orig_len // down_len # integer division to get the number of downsampled time steps
    np_data = np_data.transpose()
    d_data = np_data[:, :down_time_len*down_len].reshape(col_num, -1, down_len) # reshape the data into a 3D array
    d_data = np.median(d_data, axis=2).reshape(col_num, -1) # take the median of the downsampled data to reduce the size
    d_data = d_data.transpose()
    d_data = d_data.tolist()
    if labels is not None:
        np_labels = np.array(labels)
        d_labels = np_labels[:down_time_len*down_len].reshape(-1, down_len)
        d_labels = np.round(np.max(d_labels, axis=1))
        d_labels = d_labels.tolist()
    
    else:
        d_labels = None
            
    return d_data, d_labels

def MinMaxScaler(data):
    """Min Max normalizer.

    Args:
    - data: original data

    Returns:
    - norm_data: normalized data
    """
    minimum, maximum = np.min(data, 0), np.max(data, 0)
    numerator = data - minimum
    denominator = maximum - minimum
    norm_data = numerator / (denominator + 1e-7)
    return norm_data, (minimum, maximum)


def real_data_loading(train_path, test_path):
    """Preprocess labelled data"""
    train = pd.read_csv(train_path, index_col=0)
    test = pd.read_csv(test_path, index_col=0)
    train = train.iloc[:, 1:]
    test = test.iloc[:, 1:]
    downsample_length = 10

    # Fill NaN values
    train = fill_nan_values(train, fill_nan="fill_mean")
    test = fill_nan_values(test, fill_nan="fill_mean")
    train = fill_nan_values(train, fill_nan="fill_zero")
    test = fill_nan_values(test, fill_nan="fill_zero")

    # trim column names
    train = train.rename(columns=lambda x: x.strip())
    test = test.rename(columns=lambda x: x.strip())

    # collect attack labels
    train_labels = train.attack
    test_labels = test.attack

    train = train.drop(columns=['attack'])
    test = test.drop(columns=['attack'])

    logger.log("Shape of train data: {}".format(train.shape))
    logger.log("Shape of test data: {}".format(test.shape))
    test.columns = train.columns

    # Normalize data
    x_train, _ = MinMaxScaler(train)
    x_test, _ = MinMaxScaler(test)

    # Downsample data for large datasets
    d_train_x, d_train_labels = downsample(x_train, downsample_length, train_labels)
    train_df = pd.DataFrame(d_train_x, columns = train.columns)
    train_df['attack'] = d_train_labels
    train_df = train_df.iloc[2160:]

    d_test_x, d_test_labels = downsample(x_test, downsample_length, test_labels)
    test_df = pd.DataFrame(d_test_x, columns = test.columns)
    test_df['attack'] = d_test_labels

    return train_df, test_df, list(train.columns)

def get_fc_graph_struc(feature_list):
    struc_map = dict()
    for feature in feature_list:
        if feature not in struc_map:
            struc_map[feature] = []

        for other_feature in feature_list:
            if other_feature!=feature:
                struc_map[feature].append(other_feature)
    
    return struc_map

def build_loc_net(struc, all_features, feature_map):
    index_feature_map = feature_map
    edge_indexes = [[],[]]
    for node_name, node_list in struc.items():
        if node_name not in all_features:
            continue

        if node_name not in index_feature_map:
            index_feature_map.append(node_name)
        
        p_index = index_feature_map.index(node_name)
        for child in node_list:
            if child not in all_features:
                continue

            if child not in index_feature_map:
                logger.log("error: {} not in index_feature_map".format(child))

            c_index = index_feature_map.index(child)
            edge_indexes[0].append(c_index)
            edge_indexes[1].append(p_index)
    return edge_indexes


def construct_data(data, feature_map, label = False, labels=0):
    res = []
    for feature in feature_map:
        if feature in data.columns:
            res.append(data.loc[:, feature].values.tolist())
        else:
            logger.log("Error: {} not in data".format(feature))
    
    if label:
        # append labels as last
        sample_n = len(res[0])
        if type(labels) == int:
            res.append([labels]*sample_n)
        elif len(labels) == sample_n:
            res.append(labels)

    return res

def adj_matrix_to_edge_index(adj_matrix):
    """
        converts an adjacency matrix to edge index
        Parameters
        ----------
            adj_matrix : Adjacency matrix of the graph
                
        Return
        ----------
            edge_index: edge index of the graph
    """
    edge_index = []
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[i])):
            pair = [adj_matrix[i][i], adj_matrix[i][j]]
            edge_index.append(pair)
    
    edge_index = torch.tensor(edge_index).t().contiguous()

    return edge_index


def get_tensor_loaders(train_dataset, batch, val_ratio=0.1):
    """
    Given a dataset, return a train and validation dataloader

    Parameters:
    -----------
    train_dataset: torch.utils.data.Dataset
        The dataset to be split into train and validation
    batch: int
        The batch size
    val_ratio: float
        The ratio of validation dataset to the train dataset
    """
    dataset_len = int(len(train_dataset))
    train_use_len = int(dataset_len * (1 - val_ratio))
    val_use_len = int(dataset_len * val_ratio)
    val_start_index = random.randrange(train_use_len)
    indices = torch.arange(dataset_len)

    train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
    train_subset = Subset(train_dataset, train_sub_indices)

    val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
    val_subset = Subset(train_dataset, val_sub_indices)

    train_dataloader = DataLoader(train_subset, batch_size=batch, shuffle=True)
    val_dataloader = DataLoader(val_subset, batch_size=batch, shuffle=False)

    return train_dataloader, val_dataloader




