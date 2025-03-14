import torch
from torch import nn, optim
from data_process import load_data
import numpy as np


def train(args, model, client_id):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.base_layers.parameters(), 'lr': args.lr * 0.1},
        {'params': model.personal_layers.parameters(), 'lr': args.lr}
    ], weight_decay=args.weight_decay)

    train_loader, _ = load_data(client_id)

    for epoch in range(args.E):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(
            f"Client {client_id} Epoch {epoch + 1}/{args.E} | Loss: {epoch_loss / len(train_loader):.4f} | Acc: {accuracy:.2f}%")

    return model


def test(args, model, client_id):
    model.eval()
    _, test_loader = load_data(client_id)
    criterion = nn.CrossEntropyLoss()

    total = 0
    correct = 0
    test_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Client {client_id} Test | Loss: {test_loss / len(test_loader):.4f} | Acc: {accuracy:.2f}%")


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
