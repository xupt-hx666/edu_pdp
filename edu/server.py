from model import EducationModel
from args import args_parser
from client import train, test, validate_personalization
import torch.nn as nn
from crypto import PaillierEncryptor

args = args_parser()


class FedPer:
    def __init__(self):
        self.args = args
        self.encryptor = PaillierEncryptor()
        # 全局模型仅包含基础层
        self.global_base = nn.Sequential(
            nn.Linear(args.input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        ).to(args.device)

        # 初始化客户端模型（包含完整结构）
        self.client_models = []
        for _ in range(args.K):
            model = EducationModel(name=f"client_{_}").to(args.device)
            # 用全局基础层初始化
            model.base_layers.load_state_dict(self.global_base.state_dict())
            self.client_models.append(model)

    def aggregate(self, encrypted_weights_list):
        """聚合加密参数"""
        averaged_weights = {}
        for key in encrypted_weights_list[0].keys():
            # 提取加密数据和形状
            encrypted_arrays = [w[key]["encrypted"] for w in encrypted_weights_list]
            shapes = [w[key]["shape"] for w in encrypted_weights_list]
            shape = encrypted_weights_list[0][key]["shape"]
            if not all(shape == shapes[0] for shape in shapes):
                raise ValueError("参数形状不一致")

            # 同态加法聚合
            summed = []
            for i in range(len(encrypted_arrays[0])):
                total = encrypted_arrays[0][i]
                for arr in encrypted_arrays[1:]:
                    total += arr[i]
                summed.append(total)

            # 解密并还原
            decrypted_tensor = self.encryptor.decrypt_tensor({
                "shape": shape,
                "encrypted": summed
            }).to(args.device)
            averaged_weights[key] = decrypted_tensor

        return averaged_weights

    def server_round(self, round_idx):
        num_selected = max(int(args.C * args.K), 1)
        # selected_clients = np.random.choice(range(args.K), num_selected, replace=False)
        selected_clients = list(range(num_selected))

        # 更新客户端基础层参数（而不是整个模型）
        for idx in selected_clients:
            self.client_models[idx].base_layers.load_state_dict(self.global_base.state_dict())

        # 客户端本地训练
        encrypted_weights_list = []
        for idx in selected_clients:
            model = self.client_models[idx]
            model.base_layers.load_state_dict(self.global_base.state_dict())
            encrypted_weights = train(args, model, idx, self.encryptor)
            encrypted_weights_list.append(encrypted_weights)

        # 聚合模型
        averaged_weights = self.aggregate(encrypted_weights_list)
        self.global_base.load_state_dict(averaged_weights)

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
