import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import re

def plot_matrix(cm, labels_name, title=None, thresh=0.8, axis_labels=None, save_path=None):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化混淆矩阵
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()
    if title:
        plt.title(title)
    plt.xticks(np.arange(len(axis_labels)), axis_labels, rotation=45)
    plt.yticks(np.arange(len(axis_labels)), axis_labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]*100:.1f}%",
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    if save_path:
        plt.savefig(save_path)
    plt.show()

# 读取 CSV 文件
csv_file_path = '/root/LLUNET/results/training_results.csv'
training_results = pd.read_csv(csv_file_path)

# 获取最后一行的混淆矩阵
confusion_str = training_results.iloc[-2]['confusion']

# 清理字符串，确保没有多余的逗号和空格
confusion_str = re.sub(r'\[\s*', '[', confusion_str)
confusion_str = re.sub(r'\s*\]', ']', confusion_str)
confusion_str = re.sub(r'\s+', ',', confusion_str)
confusion_str = re.sub(r'\[,', '[', confusion_str)

# 尝试解析字符串
try:
    confusion_matrix = np.array(ast.literal_eval(confusion_str))
    confusion_matrix = confusion_matrix[:4, :4]
    labels = ['class0', 'class1', 'class2', 'class3']
    print("Parsing successful, plotting confusion matrix.")
    plot_matrix(confusion_matrix, labels, title='UResnet Confusion Matrix',
                axis_labels=labels, save_path='confusion_matrix_final_epoch.png')
except SyntaxError as e:
    print("Parsing error:", e)
