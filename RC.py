import scipy.io as sci
import numpy as np
import matplotlib.pyplot as plt
# 读取.mat文件
matdata = sci.loadmat('newdata.mat')
matlabel = sci.loadmat('newlabel.mat')

# 转换数据为numpy数组
data = np.array(matdata['newdata'])
label = np.array(matlabel['newlabel'])

# newdata_list = []
# newlabel_list = []
# for i in range(0,data.shape[0]):
#     mask = np.random.randint(2, size=(2, 1))
#     tem = mask@data[i,:].reshape(-1, 1).T
#     newdata_list.append(tem)
#     newlabel_list.append(label[:,i].reshape(-1,1))
#     newlabel_list.append(label[:,i].reshape(-1,1))
# newdata = np.vstack(newdata_list)
# newlabel = np.array(newlabel_list)
# data = newdata
# label = newlabel

Vmax = 2.0
Vmin = 1.0
tr_DL=79.26
tr=62.47
beta=0.45
y0=0.18
T=200

T1=40

index = np.arange(len(data))
np.random.shuffle(index)
datanew = np.array(list(np.array(data)[index]))
labelnew = np.array(list(np.array(label.T)[index]))
trainratio = 0.5
trainnum = round(trainratio*data.shape[0])+1
traindata = np.array(datanew[:trainnum])
testdata = np.array(datanew[trainnum:])
trainlabel = (np.array(labelnew[:trainnum])).T
testlabel = (np.array(labelnew[trainnum:])).T
#######################为数据做变换
# 接下来的操作假定inputtem已经是一个NumPy数组
UV = np.max(traindata)
LV = np.min(traindata)
input = ((traindata - LV) / (UV - LV) * (Vmax - Vmin) + Vmin)
# 将处理过的批次数据追加到总列表中
x = np.zeros(input.shape)
x[:, 0] = (input[:, 0] * (1 - np.exp(-(T / tr_DL))) + y0) * np.exp(-(T1 / tr) ** beta)
# 从第二列开始迭代更新x
for i in range(1, input.shape[1]):
    x[:, i] = (input[:, i] * (1 - np.exp(-(T / tr_DL))) + x[:, i - 1]) * np.exp(-(T1 / tr) ** beta)
x = x.T
weight = trainlabel @ x.T @ np.linalg.pinv(x @ x.T)
print(f"train down, x:{x.shape}, trainlabel:{trainlabel.shape}, weight:{weight.shape}")
#############################开始测试
UV = np.max(testdata)
LV = np.min(testdata)
input = ((testdata - LV) / (UV - LV) * (Vmax - Vmin) + Vmin)
# 将处理过的批次数据追加到总列表中
x = np.zeros(input.shape)
x[:, 0] = (input[:, 0] * (1 - np.exp(-(T / tr_DL))) + y0) * np.exp(-(T1 / tr) ** beta)
# 从第二列开始迭代更新x
for i in range(1, input.shape[1]):
    x[:, i] = (input[:, i] * (1 - np.exp(-(T / tr_DL))) + x[:, i - 1]) * np.exp(-(T1 / tr) ** beta)
x = x.T
y = weight @ x
predicted = np.argmax(y, axis=0)
true = np.argmax(testlabel, axis=0)
acc = np.mean(predicted == true)
################################画图
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Josefin Sans'# 设置全局字体# 之后创建的图形将使用设定的字体
plt.rcParams['font.size'] = 10  # 全局字号设置
plt.figure(figsize=(8, 6))
plt.title('Density Distribution\nAccuracy: {:.2%}'.format(acc))
# 以 hexbin 图的形式展示准确率分布
import seaborn as sns
sns.kdeplot(true, fill=True, label='true')
sns.kdeplot(predicted, fill=True, label='predicted')
plt.legend()
plt.xlabel('Type of ecg')
plt.ylabel('Density comparison')
plt.xticks(np.arange(0, 17, 1))  # 设置横坐标刻度的位置和标签
plt.yticks(np.arange(0, 0.3, 0.05))  # 设置纵坐标刻度的位置
plt.grid(True, linestyle='--', color='gray', linewidth=0.5)
plt.show(block=False)