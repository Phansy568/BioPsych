import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.figure()
plt.title("中文测试标题")
plt.xlabel("横轴标签")
plt.ylabel("纵轴标签")
plt.plot([1, 2, 3], [4, 5, 6])
plt.show()