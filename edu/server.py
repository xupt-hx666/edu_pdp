import torch
import numpy as np
from model import EducationModel
from args import args_parser
from client import train, test, validate_personalization
import torch.nn as nn

args = args_parser()


class FedPer:
    def __init__(self):
        self.args = args
        # 全局模型仅包含基础层
        self.global_base = nn.Sequential(
            nn.Linear(args.input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2)).to(args.device)

        # 初始化客户端模型（包含完整结构）
        self.client_models = []
        for _ in range(args.K):
            model = EducationModel(name=f"client_{_}").to(args.device)
            # 用全局基础层初始化
            model.base_layers.load_state_dict(self.global_base.state_dict())
            self.client_models.append(model)

    def aggregate(self, selected_clients):
        # 仅聚合基础层参数
        base_weights = [self.client_models[idx].base_layers.state_dict()
                        for idx in selected_clients]

        # 联邦平均
        averaged_weights = {}
        for key in base_weights[0].keys():
            averaged_weights[key] = torch.stack(
                [w[key] for w in base_weights], 0).mean(0)

        # 更新全局基础层
        self.global_base.load_state_dict(averaged_weights)

        # 分发新基础层到所有客户端
        for model in self.client_models:
            model.base_layers.load_state_dict(self.global_base.state_dict())

    def server_round(self, round_idx):
        num_selected = max(int(args.C * args.K), 1)
        # selected_clients = np.random.choice(range(args.K), num_selected, replace=False)
        selected_clients = list(range(num_selected))

        # 更新客户端基础层参数（而不是整个模型）
        for idx in selected_clients:
            self.client_models[idx].base_layers.load_state_dict(self.global_base.state_dict())

        # 客户端本地训练
        for idx in selected_clients:
            self.client_models[idx] = train(args, self.client_models[idx], idx)

        # 聚合模型
        self.aggregate(selected_clients)

    def run(self):
        for r in range(args.r):
            print(f"\n=== Round {r + 1}/{args.r} ===")
            self.server_round(r)

        # 最终测试
        print("\n=== Final Test ===")
        print("====================================")
        for idx in range(args.K):
            test(args, self.client_models[idx], idx)

        print("个性层差异")
        print("=======================")
        validate_personalization(self.client_models)  # 调用验证函数
