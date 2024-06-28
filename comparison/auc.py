import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
csv_file_path = '/root/LLUNET/comparsion.csv'
data = pd.read_csv(csv_file_path)

# 分别提取 UNet 和 UResnet 的数据
unet_data = data[data['Net'] == 'UNet']
uresnet_data = data[data['Net'] == 'UResnet']

# 绘制 PR-AUC 分数曲线图
plt.figure(figsize=(10, 6))
plt.plot(unet_data['epoch'], unet_data['pr_auc'], marker='o', linestyle='-', color='b', label='UNet')
plt.plot(uresnet_data['epoch'], uresnet_data['pr_auc'], marker='o', linestyle='-', color='g', label='UResnet')
plt.xlabel('Epoch')
plt.ylabel('PR-AUC Score')
plt.title('PR-AUC Score Comparison: UNet vs UResnet')
plt.legend()
plt.grid(True)
plt.savefig('comparison/pr_auc_comparison.png')
plt.show()
