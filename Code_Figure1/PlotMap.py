#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 18:11:13 2025

@author: planet_science
"""
import numpy as np
from matplotlib import pyplot as plt 
import marsseis
from obspy import UTCDateTime
from  matplotlib import patches
from matplotlib.pyplot import MultipleLocator
import matplotlib.colors as colors
import pandas as pd

plt.rcParams['font.family'] = 'Arial'

RMSmap = np.load('./RMSmap_BHZ.npy')
sols = np.arange(113,1297)  
dt = 0.01
hours = np.linspace(0,24,int(24/dt)+1)

fig, ax = plt.subplots(figsize=(8,8))

plt.xlim(0,24)
x_major_locator = MultipleLocator(2)
plt.xlabel("Local Mean Solar Time (hour)",fontfamily='Times New Roman',fontsize=16)
plt.ylabel("Sol (After InSight landing)",fontfamily='Times New Roman',fontsize=16)
plt.ylim(113,781)
plt.text(10,284,'Solar conjunction',fontsize=14)
plt.hlines(y=182, xmin=0, xmax=24,color='k',linestyle='--',linewidth=1)
ax.set_xticks(np.arange(0, 25, 2))
ax.set_xticklabels([f'{int(x)}' for x in np.arange(0, 25, 2)], fontfamily='Times New Roman',fontsize=18)
y_ticks = np.arange(200, 781, 100)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticks.astype(int), fontfamily='Times New Roman', fontsize=18)
ax.invert_yaxis()
ax.xaxis.set_major_locator(x_major_locator)

# Add "Seismic Noise" label in top-left corner
# Enlarge the upper-left label and add a background box.


dayleng = np.array([0,193,372,515,668])
scaleleng = dayleng/dayleng[-1]
lengthscale = np.diff(dayleng)/dayleng[-1]
c = ['green','red','orange','b',]
season = ['Spring','Summer','Fall','Winter',]

for i in range(len(c)):
    p1 = patches.Rectangle((1.,1-scaleleng[i+1]),
                          width = 0.035,
                          height = lengthscale[i],
                          alpha = 0.3,
                          facecolor = c[i],
                          transform=ax.transAxes)
    fig.add_artist(p1) 
    # Add season and Ls labels together
    plt.text(1.005, 1-scaleleng[i+1]+lengthscale[i]/3, 
             f"{season[i]}", 
             transform=ax.transAxes, rotation=90, fontsize=14)
ls_values = ['0°', '90°', '180°', '270°']  # Ls values
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())

# Place Ls ticks between the seasonal bands.
ls_ticks = [(dayleng[i] + 113) for i in range(len(dayleng)-1)]
ax2.set_yticks(ls_ticks)
ax2.set_yticklabels(ls_values, fontsize=14)

plt.text(0.98, 1.04, 'Ls', transform=ax.transAxes, 
         fontsize=16, verticalalignment='top', rotation=0)
# Position the Ls label to avoid overlap with the season labels.
ax2.yaxis.set_label_coords(1.08, 0.5)

RMSmap[RMSmap==0] = np.nan
plt.pcolormesh(hours,sols,RMSmap,vmin=-9.5,vmax=-7.5,cmap='plasma')
norm = colors.Normalize(vmin=-9.5, vmax=-7.5)
cmap=plt.get_cmap('plasma')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(RMSmap)
cbar = fig.colorbar(sm, cax=ax.inset_axes((0., -0.12, 0.4, 0.02)),orientation='horizontal',extend='both')

SunRiseSetTime = np.load('./SunRiseSetTime.npy')
sols = SunRiseSetTime[:,0]
SunRise = SunRiseSetTime[:,1]
SunSet = SunRiseSetTime[:,2]
plt.plot(SunRise,sols,color='orange',linestyle='--')
plt.plot(SunSet,sols,color='orange',linestyle='--')
plt.text(0.25,1.01,"Sunrise",transform=ax.transAxes,fontsize=12)
plt.text(0.75,1.01,"Sunset",transform=ax.transAxes,fontsize=12)
plt.text(0.15,-0.18,'$log_{10}$(RMS)',transform=ax.transAxes,fontsize=12)

cat = pd.read_excel('Catalogue.xlsx')
evedate = list(cat['Date(UTC)'].dropna())
quality = list(cat['Quality'].dropna())
evetype = list(cat['Type'].dropna())
eve = list(cat['Name'].dropna())
eve1 = list(set(eve))

HFsols = []
HFhours = []
LFsols = []
LFhours = []
for i in range(len(eve1)):
    idx = eve.index(eve1[i])
    if evetype[idx] == '2.4 HZ' or evetype[idx] == 'HIGH FREQUENCY':
        tt = UTCDateTime(evedate[idx])
        t1 = marsseis.lmst()
        t1.utc2lmst(tt)
        HFsols.append(int(t1.sol))
        hours = t1.hour+t1.min/60
        HFhours.append(hours)

    if evetype[idx] == 'LOW FREQUENCY' or evetype[idx] == 'BROADBAND':
        tt = UTCDateTime(evedate[idx])
        t1 = marsseis.lmst()
        t1.utc2lmst(tt)
        LFsols.append(int(t1.sol))
        hours = t1.hour+t1.min/60
        LFhours.append(hours)
        

HFsols = np.array(HFsols)
HFhours = np.array(HFhours)
plt.scatter(HFhours,HFsols,marker='v',color='c',s=40,edgecolors='k',alpha=0.8,label='HF events (2.4 Hz + HIGH FREQUENCY)')

LFsols = np.array(LFsols)
LFhours = np.array(LFhours)
plt.scatter(LFhours,LFsols,marker='o',color='deeppink',s=40,edgecolors='k',alpha=0.8,label='LF events (LOW FREQUENCY + BROADBAND)')

plt.legend(loc='lower right',bbox_to_anchor=(0.85, -0.02),borderaxespad=-7)
plt.text(0.02, 0.98, 'Seismic Noise', transform=ax.transAxes, 
         fontsize=16, fontweight='bold', 
         verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='lightgray', edgecolor='black'))
plt.tight_layout()
plt.show()