from phe import paillier
import numpy as np
from tqdm import tqdm
import torch


class PaillierEncryptor:
    def __init__(self, key_size=1024):
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=key_size)

    def encrypt_tensor(self, tensor, progress=True):
        """加密张量并返回加密后的数据和原始形状"""
        scaled = (tensor.cpu().numpy() * 1e4 + 1e4).astype(int)  # 处理负值并放大
        # 检查参数范围
        if np.any(scaled < 0) or np.any(scaled >= self.public_key.n):
            raise ValueError("加密值超出Paillier支持范围")
        encrypted = []
        with tqdm(total=scaled.size, desc="Encrypting", disable=not progress) as pbar:
            for x in scaled.flatten():
                encrypted.append(self.public_key.encrypt(int(x)))
                pbar.update(1)
        return {
            "shape": tensor.shape,
            "encrypted": encrypted
        }

    def decrypt_tensor(self, encrypted_data, progress=True):
        """解密数据并还原为PyTorch张量"""
        decrypted = []
        with tqdm(total=len(encrypted_data["encrypted"]), desc="Decrypting", disable=not progress) as pbar:
            for x in encrypted_data["encrypted"]:
                decrypted.append(self.private_key.decrypt(x))
                pbar.update(1)
        decrypted_np = (np.array(decrypted).reshape(encrypted_data["shape"]) - 1e4) / 1e4  # 还原缩放和偏移
        return torch.from_numpy(decrypted_np).float()
