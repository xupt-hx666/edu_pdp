import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from args import args_parser

args = args_parser()


def generate_education_data():
    # 生成教育场景的合成数据（20个特征，5个类别）
    X, y = make_classification(
        n_samples=5000,
        n_features=args.input_dim,
        n_classes=args.num_classes,
        class_sep=2.0,  # 类别分离程度
        n_informative=8,  # 对分类有效的特征数量
        weights=[0.3, 0.3, 0.1, 0.2, 0.1],  # 每一类数据占的权重
        n_clusters_per_class=1,  # 每个分类中数据集中在一簇(即在每一类中数据集中)
        random_state=42,  # 随机数状态，若设置成None则每次生成的数据集都不一样，42保证每次生成的数据集一致便于后续的模型评估
        shuffle=True
    )
    return X, y


"""存在约束条件:class*cluster<=2**informative"""


def split_non_iid_data(X, y, num_clients):
    # 模拟Non-IID分布：每个客户端只包含2个类别
    client_data = []
    for i in range(num_clients):
        class_1 = i % args.num_classes
        class_2 = (i + 1) % args.num_classes
        indices = np.where((y == class_1) | (y == class_2))[0]
        np.random.shuffle(indices)
        client_data.append((X[indices], y[indices]))
    return client_data


def load_data(client_id):
    # 生成数据并划分Non-IID分布
    X, y = generate_education_data()
    client_data = split_non_iid_data(X, y, args.K)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        client_data[client_id][0],
        client_data[client_id][1],
        test_size=0.2,  # 测试集占比20%
        random_state=42
    )

    # 转换为TensorDataset
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)  # 分类任务使用LongTensor
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.B, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.B, shuffle=False)

    return train_loader, test_loader
