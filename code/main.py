# plt调用gcf函数取得当前绘制的figure并调用savefig函数
import ecg_plot
from models import *
ecg_1 = np.load('1_3.npy')
ecg_2 = np.load('3_1.npy')
ecg_3 = np.load('2.npy')
ecg_4 = np.load('4_1.npy')
lead_index = list(range(4))
ecg = np.concatenate([ecg_1, ecg_2,ecg_3,ecg_4], axis=0)
ecg_plot.plot_12(ecg.reshape((-1,2560)), sample_rate=256, title = '', lead_index=lead_index, lead_order=list(range(4)), columns=1)
# ecg_plot.save_as_png('ecg.png',dpi=1000)
foo_fig = plt.gcf() # 'get current figure'
foo_fig.savefig('ECG.png', dpi=1000)
plt.show()