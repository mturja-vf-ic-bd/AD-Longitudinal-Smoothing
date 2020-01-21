import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl

with open('conf_list_0_0', 'rb') as f:
     conf_list = pkl.load(f)

for i, conf in enumerate(conf_list):
     df_cm = pd.DataFrame(conf)
     # plt.figure(figsize = (7,10))
     sn.set(font_scale=1.4)#for label size
     sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
     plt.savefig('conf_mat' + str(i)+ '.png')
     plt.show()