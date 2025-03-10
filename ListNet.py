import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

# -----------------------------
# NAS-Bench-101 搜索空间设置
# -----------------------------
NUM_NODES = 7                # NAS-Bench-101 中 cell 的节点数（包括输入和输出）
OPS = ['conv3x3', 'conv1x1', 'maxpool3x3']  # 可选操作
NUM_INTERMEDIATE = NUM_NODES - 2  # 中间节点数

class Architecture:
    """
    用于表示 NAS-Bench-101 中的 cell 架构。
    matrix 为上三角二值矩阵（节点间连接情况），
    ops 为中间节点的操作列表。
    """
    def __init__(self, matrix, ops):
        self.matrix = matrix  # numpy 数组，大小 (NUM_NODES, NUM_NODES)，上三角有效
        self.ops = ops        # 长度为 NUM_INTERMEDIATE 的操作列表

    def __str__(self):
        return f"Matrix: {self.matrix.tolist()}, Ops: {self.ops}"

    def copy(self):
        return Architecture(self.matrix.copy(), self.ops.copy())

def random_architecture():
    """
    随机生成一个 NAS-Bench-101 架构。
    """
    matrix = np.zeros((NUM_NODES, NUM_NODES), dtype=int)
    for i in range(NUM_NODES):
        for j in range(i+1, NUM_NODES):
            matrix[i, j] = random.choice([0, 1])
    ops = [random.choice(OPS) for _ in range(NUM_INTERMEDIATE)]
    return Architecture(matrix, ops)

def mutate(arch, mutation_rate=0.1):
    """
    对给定架构进行变异：对矩阵中的每个连接以及每个操作以一定概率进行随机翻转/替换。
    """
    new_arch = arch.copy()
    # 变异矩阵中上三角部分
    for i in range(NUM_NODES):
        for j in range(i+1, NUM_NODES):
            if random.random() < mutation_rate:
                new_arch.matrix[i, j] = 1 - new_arch.matrix[i, j]
    # 变异操作
    for idx in range(len(new_arch.ops)):
        if random.random() < mutation_rate:
            new_arch.ops[idx] = random.choice(OPS)
    return new_arch

def crossover(parent1, parent2):
    """
    对两个父代架构进行交叉操作，分别对操作列表和矩阵进行 one-point crossover，
    返回生成的子代（这里只返回一个子代）。
    """
    child = parent1.copy()
    # 操作列表交叉
    point = random.randint(1, len(parent1.ops) - 1)
    child.ops = parent1.ops[:point] + parent2.ops[point:]
    # 矩阵交叉：将上三角部分拉平后交叉，再恢复成矩阵
    indices = [(i, j) for i in range(NUM_NODES) for j in range(i+1, NUM_NODES)]
    point_matrix = random.randint(1, len(indices) - 1)
    flat1 = [parent1.matrix[i, j] for (i, j) in indices]
    flat2 = [parent2.matrix[i, j] for (i, j) in indices]
    new_flat = flat1[:point_matrix] + flat2[point_matrix:]
    new_matrix = np.zeros((NUM_NODES, NUM_NODES), dtype=int)
    for idx, (i, j) in enumerate(indices):
        new_matrix[i, j] = new_flat[idx]
    child.matrix = new_matrix
    return child

def evaluate_architecture(arch):
    """
    模拟 NAS-Bench-101 的评价函数，计算架构的真实得分。
    此处将矩阵中连接数和操作类型作为评价依据，
    实际使用时请替换为真实评价接口。
    """
    matrix_score = np.sum(arch.matrix) / ((NUM_NODES * (NUM_NODES - 1)) / 2)  # 归一化后的连接数得分（0~1）
    # 计算操作得分：conv3x3 得分较高，其他稍低
    ops_score = sum([1.0 if op == 'conv3x3' else 0.5 for op in arch.ops]) / len(arch.ops)
    score = 0.7 * matrix_score + 0.3 * ops_score
    return score

def architecture_to_feature(arch):
    """
    将架构转换为特征向量：
      - 拉平上三角矩阵（仅取 i<j 部分）
      - 对中间节点操作进行 one-hot 编码
    """
    matrix_features = arch.matrix[np.triu_indices(NUM_NODES, k=1)]
    ops_features = []
    for op in arch.ops:
        one_hot = [1 if op == candidate else 0 for candidate in OPS]
        ops_features.extend(one_hot)
    features = np.concatenate([matrix_features, np.array(ops_features)])
    return features.astype(np.float32)

# -----------------------------
# ListNet 性能预测器
# -----------------------------
class ListNetPredictor(nn.Module):
    """
    使用简单的全连接网络作为 ListNet 预测器，
    输入为架构特征，输出为单个得分。
    """
    def __init__(self, input_dim):
        super(ListNetPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.model(x)

def listnet_loss(preds, targets):
    """
    ListNet 的 listwise loss：对预测值和真实值先做 softmax，再计算交叉熵损失。
    preds 和 targets 均为 (N, 1) 的向量。
    """
    preds = preds.view(-1)
    targets = targets.view(-1)
    preds_softmax = torch.softmax(preds, dim=0)
    targets_softmax = torch.softmax(targets, dim=0)
    loss = -torch.sum(targets_softmax * torch.log(preds_softmax + 1e-10))
    return loss

def train_listnet(predictor, train_features, train_scores, num_epochs=100, lr=0.01):
    """
    训练 ListNet 预测器。
    train_features: (N, feature_dim) 的 numpy 数组
    train_scores: (N,) 的 numpy 数组（真实评价得分）
    """
    optimizer = optim.Adam(predictor.parameters(), lr=lr)
    predictor.train()
    features_tensor = torch.tensor(np.array(train_features))
    scores_tensor = torch.tensor(np.array(train_scores)).unsqueeze(1)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        preds = predictor(features_tensor)
        loss = listnet_loss(preds, scores_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    return predictor

# -----------------------------
# 进化搜索算法
# -----------------------------
def evolutionary_search(predictor, population_size=50, generations=20, mutation_rate=0.1):
    """
    利用进化算法进行架构搜索：
      - 初始种群随机生成
      - 每一代使用预测器估计得分，保留排名靠前的一半
      - 通过交叉和变异产生新的候选架构，更新种群
    最终返回预测器认为最优的架构，并给出真实评价得分。
    """
    # 初始化种群
    population = [random_architecture() for _ in range(population_size)]
    for gen in range(generations):
        print(f"\nGeneration {gen+1}")
        # 预测器评估
        features = [architecture_to_feature(arch) for arch in population]
        inputs = torch.tensor(np.array(features))
        predictor.eval()
        with torch.no_grad():
            scores = predictor(inputs).view(-1).numpy()
        # 按预测得分降序排序（得分越高越好）
        sorted_indices = np.argsort(scores)[::-1]
        population = [population[i] for i in sorted_indices]
        best_score = scores[sorted_indices[0]]
        print(f"  Best predicted score: {best_score:.4f}")
        # 产生新种群：保留上半部分个体，并通过交叉、变异生成新个体
        new_population = population[:population_size // 2]
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population[:population_size // 2], 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        population = new_population
    # 返回最终种群中预测器认为最优的架构
    features = [architecture_to_feature(arch) for arch in population]
    inputs = torch.tensor(np.array(features))
    predictor.eval()
    with torch.no_grad():
        scores = predictor(inputs).view(-1).numpy()
    sorted_indices = np.argsort(scores)[::-1]
    best_arch = population[sorted_indices[0]]
    print("\nBest architecture found:")
    print(best_arch)
    # 使用真实评价函数对最优架构评估
    real_score = evaluate_architecture(best_arch)
    print(f"Real evaluation score: {real_score:.4f}")
    return best_arch

# -----------------------------
# 主函数：训练 ListNet 预测器并利用进化算法搜索架构
# -----------------------------
def main():
    # 生成训练数据（模拟 NAS-Bench-101 数据）
    num_train_samples = 200
    architectures = [random_architecture() for _ in range(num_train_samples)]
    train_features = [architecture_to_feature(arch) for arch in architectures]
    train_scores = [evaluate_architecture(arch) for arch in architectures]
    input_dim = len(train_features[0])
    
    # 初始化并训练 ListNet 预测器
    predictor = ListNetPredictor(input_dim)
    print("Training ListNet predictor...")
    predictor = train_listnet(predictor, train_features, train_scores, num_epochs=100, lr=0.01)
    
    # 利用进化算法进行架构搜索
    print("\nStarting evolutionary search...")
    best_arch = evolutionary_search(predictor, population_size=50, generations=20, mutation_rate=0.1)

if __name__ == '__main__':
    main()
