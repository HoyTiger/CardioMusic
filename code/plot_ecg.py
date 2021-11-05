# plt调用gcf函数取得当前绘制的figure并调用savefig函数
import ecg_plot
from models import *
ecg_1 = np.load('1_1.npy')
ecg_2 = np.load('2.npy')
lead_index = list(range(2))
ecg = np.concatenate([ecg_1, ecg_2], axis=0)
ecg_plot.plot_12(ecg.reshape((-1,2560)), sample_rate=256, title = '', lead_index=lead_index, lead_order=list(range(2)), columns=1)

foo_fig = plt.gcf() # 'get current figure'
foo_fig.savefig('foo.eps', format='eps', dpi=1000)
plt.show()