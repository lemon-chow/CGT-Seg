import numpy as np

# 文件路径
file_path = '/root/Pytorch-UNet/results/training_results.csv.npy'

# 读取.npy文件，设置allow_pickle为True
data = np.load(file_path, allow_pickle=True)

# 打印数据类型
print(data.dtype)

# 检查是否为结构化数组，并打印每列的数据类型
if data.dtype.names:
    for name in data.dtype.names:
        print(f"Column {name} has data type {data.dtype[name]}")
else:
    # 如果不是结构化数组，就打印整个数组的数据类型
    print("Data type of the array:", data.dtype)
