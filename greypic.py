from PIL import Image, ImageEnhance
import numpy as np


# 使用numpy和PIL处理图像

# 首先，我们将读取图像并使用numpy转换为数组，然后对数组进行处理。
img = Image.open('88_pre_mask.png')
img_array = np.array(img)

# 获取独特的像素值
unique_pixel_values = np.unique(img_array)

unique_pixel_values
print(unique_pixel_values)
# 根据要求设置颜色：
# 1 - 白色，2 - 灰色，3 - 黑色
# 我们假设这里的1, 2, 3是像素值，将使用255表示白色，128表示灰色，0表示黑色
# 首先创建一个新的数组，用白色初始化它
new_img_array = np.ones_like(img_array) * 255

# 现在，将对应的像素值更改为指定的颜色
new_img_array[img_array == 1] = 255  # 白色
new_img_array[img_array == 2] = 128  # 灰色
new_img_array[img_array == 0] = 0    # 黑色

# 创建一个新的PIL图像并保存
new_img = Image.fromarray(new_img_array)
new_img_path = 'enhanced_88_OUT_1.png'
new_img.save(new_img_path)

new_img.show()

new_img_path
