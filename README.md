个性化联邦学习模型及同态加密实现

模型目标：模型经训练后能够分析学生的学习偏好，从而将学生分为有不同学习偏好的类别，以期实现个性化学习方案提供。


FedPer模型的核心是在深度学习模型中设置了基础层和个性化层。基础层负责学习通用的特征表示，而个性化层则针对每个客户端的特定数据分布进行优化。在训练过程中，基础层的参数在所有客户端之间共享，并通过FedAvg算法进行更新；而个性化层的参数则在每个客户端本地进行训练，以此实现个性化学习方案的目标。

参考github开源”https://github.com/ki-ljl/FedPer”
环境配置：https://blog.csdn.net/weixin_47590992/article/details/141819466?spm=1001.2014.3001.5506


工作
一：基于fedper模型的改进
1.模拟教育数据集
由于缺少相关教育数据集，且考虑到不同数据集在数据处理时的差异性，我从机器学习库里引入make_classification（）生成数据集函数生成模拟教育数据集，便于进行模型训练。

parser.add_argument('--input_dim', type=int, default=20, help='输入特征维度')
parser.add_argument('--num_classes', type=int, default=5, help='分类类别数')

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
生成5000个数据，每个数据样本规定有20个样本特征，总的数据分类划分为5。而对于模拟的教育数据集来说，类别初步划分为喜爱文学的，喜爱理学的，喜爱艺术的...(可根据具体场景自主定义)。样本特征可以是学生年龄，性别，家庭环境，教育背景，学习时长，学习频率，课堂参与度，教育资源使用情况，是否接受小组合作学习...等，其中规定对于划分类别有用的特征有8个，可能存在一些无用特征，如出勤率，身高等。
X为二维数组(矩阵)，每一行代表一个样本，每一列代表每一个样本的其一特征。
y为一维数组，数据标签（即五个分类）。

2.进行客户端数据分配，模拟Non-IID分布（非独立同分布）
Non-IID分布特点：
A.数据可能在类别分布上存在显著差异。例如，某些客户端可能主要拥有某一类别的数据，而其他客户端则拥有不同的类别分布.
B.数据的分布可能随时间变化，不同客户端的数据可能在不同的时间点上有不同的分布
这个对于模拟个性化教育场景是非常有必要的，因为个性化教育场景与上述特点有类似之处，例如，不同的客户端上的学生学习偏好数据可能差异较大，考虑到地理环境跟地区发展程度，沿海地区的学生的学习偏好可能与西北内陆地区的学生的学习偏好可能有较为明显的差异，这可能是由教育资源导致的。而且学生的学习偏好可能随着时间改变而改变。
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

3.实现效果检验量化。通过计算不同客户端模型中personal_layers（个性化层）参数的欧氏距离，衡量各客户端模型的个性化程度。差异度越大，表明各客户端模型在本地数据上的适应越独特；差异度越小，则说明模型个性化程度较低。此外有效验证了联邦学习策略在联邦学习框架（如 FedPer）中，通常希望：基础层（共享层）保持一致性（通过聚合更新）。个性化层（本地层）保持差异性（通过本地训练保留特有模式）。通过量化差异度，验证算法是否达到了预期效果。
def validate_personalization(models):
    """验证个性化效果：比较不同客户端的个性化层差异"""
    personal_params = []
    for model in models:
        params = [p.data.cpu().numpy()
                  for p in model.personal_layers.parameters()]
        personal_params.append(params)

    # 计算参数差异度
    diff = np.mean([np.linalg.norm(personal_params[i][0] - personal_params[j][0])
                    for i in range(len(models)) for j in range(i + 1, len(models))])
    print(f"个性化层平均差异度：{diff:.4f}")

运行结果展示：
=== Final Test ===
====================================
Client 0 Test | Loss: 0.2217 | Acc: 90.12%
Client 1 Test | Loss: 0.0227 | Acc: 98.75%
Client 2 Test | Loss: 0.0118 | Acc: 98.75%
Client 3 Test | Loss: 0.0140 | Acc: 98.77%
Client 4 Test | Loss: 0.0414 | Acc: 97.53%
个性层差异
=======================
个性化层平均差异度：11.2030



未来改进建议：
在更加具体化的教育场景中，大的分类中需要将类簇数设置为多个，比如当大类划分为喜爱文学的，喜爱理学的，喜爱艺术的，喜爱棋类的，这是初步的学习偏好分类。在每一个大类中又会存在喜爱的教学方式的不同，存在偏好时间学习，偏好小组合作学习，偏好线上上下教师授课学习。这样我们的个性化教育推荐不仅能将不同学生进行学习内容的分类，还可以根据不同学生学习方式的不同进行学习方式的推荐。
这需要对data.process部分代码进行修改。

