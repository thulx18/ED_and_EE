import matplotlib.pyplot as plt


f1 = [0.0, 0.0399, 0.2061, 0.3476, 0.6685, 0.7772, 0.7877, 0.8810, 0.9248,0.9408,0.9709]
input_values = [i for i in range(0, len(f1))]
plt.plot(input_values, f1, linewidth=3)

# 设置图片标题，并给坐标轴x,y分别加上标签
plt.xlabel('epoch', fontsize=14)
plt.ylabel('f1', fontsize=14)
# 设置刻度标记的大小,axis='both'表示两坐标轴都设置
plt.tick_params(axis='both', labelsize=14)
plt.show()
plt.savefig("./figure/ed_f1.png")