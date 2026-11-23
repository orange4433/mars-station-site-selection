# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 21:56:10 2024

@author: 19011
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from  matplotlib import patches

plt.rcParams['font.family'] = 'Arial'
m = 150
# 参数配置
sol_range = [113, 781]
dt = 0.01
hours = np.arange(0, 24, dt)
size = 20
input_file_east = 'wind_east.npy'
input_file_north = 'wind_north.npy'

# 加载数据
RMS_data_east = np.load(input_file_east)
RMS_data_north = np.load(input_file_north)

# 计算风速平方
wind_speed_squared = RMS_data_east**2 + RMS_data_north**2  # 改为风速平方

sols = np.arange(sol_range[0], sol_range[1])
wind_speed_squared = wind_speed_squared[sol_range[0]:sol_range[1],:]

# 绘图设置
fig, ax = plt.subplots(figsize=(8, 8))
cmap = plt.get_cmap('plasma')
vmin = 0  # 设置颜色条的最小值为0
vmax = m  # 设置颜色条的最大值为16的平方=256

# 创建热图
mesh = ax.pcolormesh(
    hours, sols, wind_speed_squared,
    cmap=cmap,
    norm=colors.Normalize(vmin=vmin, vmax=vmax),
    shading='auto'
)



dayleng = np.array([0,193,372,515,668])
scaleleng = dayleng/dayleng[-1]
lengthscale = np.diff(dayleng)/dayleng[-1]
c = ['green','red','orange','b',]
season = ['Spring','Summer','Fall','Winter',]
for i in range(len(c)):
    p1 = patches.Rectangle((1.,1-scaleleng[i+1]),                # 矩形左下角坐标
                          width = 0.035,                   # 矩形的宽度
                          height = lengthscale[i],                  # 矩形的高度
                          alpha = 0.3,                    # 矩阵的透明度
                          facecolor = c[i],            # 是否填充矩阵(设置为绿色)
                          transform=ax.transAxes)    # 保持矩形显示在图像最上方
    fig.add_artist(p1) 
    plt.text(1.005,1-scaleleng[i+1]+lengthscale[i]/3,season[i],transform=ax.transAxes,rotation=90,fontsize=14)

ls_values = ['0°', '90°', '180°', '270°']  # Ls values
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())

# 设置Ls刻度位置（在季节条带之间）
ls_ticks = [(dayleng[i] + 113) for i in range(len(dayleng)-1)]
ax2.set_yticks(ls_ticks)
ax2.set_yticklabels(ls_values, fontsize=14)
ax2.invert_yaxis()
plt.text(0.98, 1.04, 'Ls', transform=ax.transAxes, 
         fontsize=16, verticalalignment='top', rotation=0)
# 调整Ls标签位置，避免与季节标签重叠
ax2.yaxis.set_label_coords(1.08, 0.5)

SunRiseSetTime = np.load('./SunRiseSetTime.npy')
sols = SunRiseSetTime[:668,0]
SunRise = SunRiseSetTime[:668,1]
SunSet = SunRiseSetTime[:668,2]
plt.plot(SunRise,sols,color='orange',linestyle='--')
plt.plot(SunSet,sols,color='orange',linestyle='--')
plt.text(0.25,1.01,"Sunrise",transform=ax.transAxes,fontsize=12)
plt.text(0.75,1.01,"Sunset",transform=ax.transAxes,fontsize=12)

# 坐标轴设置
ax.set_xlim(0, 24)
ax.set_ylim(sol_range[0], sol_range[1])
ax.invert_yaxis()
ax.set_xlabel('Local Mean Solar Time (hour)', fontfamily='Times New Roman', fontsize=16)
ax.set_ylabel('Sol (After InSight landing)', fontfamily='Times New Roman', fontsize=16)
ax.set_xticks(np.arange(0, 25, 2))
ax.set_xticklabels([f'{int(x)}' for x in np.arange(0, 25, 2)], fontfamily='Times New Roman',fontsize=size-2)

# 手动设置 y 轴刻度位置和标签
y_ticks = np.arange(sol_range[0]+87, sol_range[1] + 1, 100)  # 每 100 个 Sol 设置一个刻度
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticks.astype(int), fontfamily='Times New Roman', fontsize=size-2)

# 添加颜色条
norm = colors.Normalize(vmin=0, vmax=m)  # 修改为0-256
cmap=plt.get_cmap('plasma')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# divider = make_axes_locatable(ax)
sm.set_array(wind_speed_squared)
cbar = fig.colorbar(sm, cax=ax.inset_axes((0., -0.12, 0.4, 0.02)),orientation='horizontal',extend='both')
plt.text(0.10,-0.18,'Wind Speed Squared (m²/s²)',transform=ax.transAxes,fontsize=12)  # 修改单位

plt.text(0.02, 0.98, 'Wind Speed Squared', transform=ax.transAxes,  # 修改标题
         fontsize=16, fontweight='bold', 
         verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='lightgray', edgecolor='black'))

'''
cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label('Wind Speed Squared (m²/s²)', fontfamily='Times New Roman', fontsize=size)  # 修改单位
cbar.ax.tick_params(labelsize=size-2)
'''
plt.tight_layout()
plt.show()