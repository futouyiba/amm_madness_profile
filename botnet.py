import numpy as np
from halutmatmul.halutmatmul import HalutMatmul
import torch
import torch.nn.functional as F

import json
import numpy as np
import csv
from tqdm import tqdm

class MultiLayerPerceptron:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

def read_csv_data(file_path, num_lines=1000000):
    data = []
    labels = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in tqdm(reader, desc="Reading CSV"):
            # print(type(row[0]))
            labels.append(int(float(row[0])))  # 第一列是标签
            features = [float(x) for x in row[1:]]  # 后面的列是特征
            data.append(features)
            if len(data) >= num_lines:
                break
    return data, labels

# 加载模型参数JSON文件
# json_file = '../simulate/model_int.json'
json_file = '../simulate/model.json'


with open(json_file, 'r') as file:
    model_data = json.load(file)

# 解析模型参数
layers_data = model_data['layers']

weights = []
biases = []

for layer_data in layers_data:
    weights.append(np.array(layer_data['weights']))
    biases.append(np.array(layer_data['biases']))

# 创建模型实例
model = MultiLayerPerceptron(weights, biases)

# 加载测试数据集和标签
test_csv_file = '../dataset/fingerprint/train_redeal.csv'  # 输入的 CSV 文件路径
data, labels = read_csv_data(test_csv_file,120000)

test_data, test_labels = data[100000:], labels[100000:]

# train_csv_file = '../dataset/botnet/train_redeal.csv'  # 输入的 CSV 文件路径
# train_data, train_labels = read_csv_data(train_csv_file,200000)

train_data, train_labels = data[:100000], labels[:100000]

def fix_reset(x, num):
    for i in range(len(x)):
        for j in range(len(x[i])):
            if num < 0:
                x[i][j] = x[i][j] >> (-num) << (-num)


def get_layers(data):
    layers = []
    fix_point = [-1, -2, -2, -1]

    layer_input = data
    for i in range(len(model.weights)):
        # fix_reset(layer_input, fix_point[i])
        layers.append(np.array(layer_input))
        layer_output = np.dot(layer_input, model.weights[i].T) + model.biases[i]
        layer_output = np.maximum(layer_output, 0)  # 使用ReLU激活函数
        layer_input = layer_output

    layers.append(np.array(layer_input))

    return layers
train_layers = get_layers(train_data)
test_layers = get_layers(test_data)

hm_1 = HalutMatmul(C=40, K=64)
hm_1.learn_offline(train_layers[0], model.weights[0].T)

hm_2 = HalutMatmul(C=64, K=64)
hm_2.learn_offline(train_layers[1], model.weights[1].T)

hm_3 = HalutMatmul(C=32, K=64)
hm_3.learn_offline(train_layers[2], model.weights[2].T)

test_halt_1 = np.maximum(hm_1.matmul_online(np.array(test_data)) + model.biases[0], 0)
test_halt_2 = np.maximum(hm_2.matmul_online(test_halt_1) + model.biases[1], 0)
test_halt_3 = np.maximum(hm_3.matmul_online(test_halt_2) + model.biases[2], 0)

predicted_labels = np.argmax(test_halt_3, axis=1)
accuracy = np.mean(predicted_labels == test_labels) * 100
print(f"预测准确率: {accuracy:.4f}%")