### suncong 2023

from fmcd import mcd
import numpy as np
import xarray as xr
import yaml
from tqdm import tqdm
import argparse
import os
from joblib import Parallel, delayed
from multiprocessing import Pool, Manager, Lock
# from multiprocessing.dummy import Pool as ThreadPool
import re
# import functools
import time
from rich.console import Console
from rich.theme import Theme

# Defaults
dset = "/mnt/d/pku/science/ljq/data/model data/MCDv6.1_python_interface/python/data/" # path to 'MCD_DATA'

mcd_var_field_ori = { \
	1: "Radial distance from planet center (m)", \
	2: "Altitude above areoid (Mars geoid) (m)", \
	3: "Altitude above local surface (m)", \
	4: "orographic height (m) (surface altitude above areoid)", \
	5: "GCM orography (m)", \
	6: "Local slope inclination (deg) (HR mode only)", \
	7: "Local slope orientation (deg) (0 deg Northward) (HR mode only)", \
	8: "Sun-Mars distance (in Astronomical Unit AU)", \
	9: "Ls, solar longitude of Mars (deg)", \
	10: "LST:Local true solar time (hrs)", \
	11: "LMT:Local mean time (hrs) at sought longitude", \
	12: "Universal solar time (LST at lon=0) (hrs)", \
	13: "Solar zenith angle (deg)", \
	14: "Surface temperature (K)", \
	15: "Surface pressure (Pa)", \
	16: "GCM surface pressure (Pa)", \
	17: "Potential temperature (K) (reference pressure=610Pa)", \
	18: "Vertical wind component (m/s) (Up-Down)", \
	19: "Zonal slope wind component (m/s) (HR mode only)", \
	20: "Meridional slope wind component (m/s) (HR mode only)", \
	21: "Surface pressure RMS day to day variations (Pa)", \
	22: "Surface temperature RMS day to day variations (K)", \
	23: "Atmospheric pressure RMS day to day variations (Pa)", \
	24: "Density RMS day to day variations (kg/m^3)", \
	25: "Temperature RMS day to day variations (K)", \
	26: "Zonal wind RMS day to day variations (m/s)", \
	27: "Meridional wind RMS day to day variations (m/s)", \
	28: "Vertical wind RMS day to day variations (m/s)", \
	29: "Incident solar flux at top of the atmosphere (W/m2)", \
	30: "solar flux reflected to space (W/m2)", \
	31: "Incident solar flux on horizontal surface (W/m2)", \
	32: "Incident solar flux on local slope (W/m2) (HR mode only)", \
	33: "Reflected solar flux on horizontal surface (W/m2)", \
	34: "thermal IR flux to space (W/m2)", \
	35: "thermal IR flux on surface (W/m2)", \
	36: "GCM surface roughness length z0 (m)", \
	37: "GCM surface thermal inertia", \
	38: "GCM surface bare ground albedo", \
	39: "Monthly mean dust column visible optical depth above surface", \
	40: "Daily mean dust column visible optical depth above surface", \
	41: "Dust mass mixing ratio (kg/kg)", \
	42: "Dust effective radius (m)", \
	43: "Daily mean dust deposition rate on horizontal surface (kg m-2 s-1)", \
	44: "Monthly mean surface CO2 ice layer (kg/m2)", \
	45: "Monthly mean surface H2O layer (kg/m2) (non perennial frost)", \
	46: "GCM perennial surface water ice (0 or 1)", \
	47: "Water vapor column (kg/m2)", \
	48: "Water vapor vol. mixing ratio (mol/mol)", \
	49: "Water ice column (kg/m2)", \
	50: "Water ice mixing ratio (mol/mol)", \
	51: "Water ice effective radius (m)", \
	52: "Convective Planetary Boundary Layer (PBL) height (m)", \
	53: "Max. upward convective wind within the PBL (m/s)", \
	54: "Max. downward convective wind within the PBL (m/s)", \
	55: "Convective vertical wind variance at level z (m2/s2)", \
	56: "Convective eddy vertical heat flux at level z (m/s/K)", \
	57: "Surface wind stress (kg/m/s2)", \
	58: "Surface sensible heat flux (W/m2) (<0 when flux from surf to atm.)", \
	59: "Air heat capacity Cp (J kg-1 K-1)", \
	60: "gamma=Cp/Cv Ratio of specific heats", \
	61: "R:Molecular gas constant (J K-1 kg-1)", \
	62: "Air viscosity estimation (N s m-2)", \
	63: "Scale height H(p) (m)", \
	64: "[CO2] volume mixing ratio (mol/mol)", \
	65: "[N2] volume mixing ratio  (mol/mol)", \
	66: "[Ar] volume mixing ratio  (mol/mol)", \
	67: "[CO] volume mixing ratio  (mol/mol)", \
	68: "[O] volume mixing ratio   (mol/mol)", \
	69: "[O2] volume mixing ratio  (mol/mol)", \
	70: "[O3] volume mixing ratio  (mol/mol)", \
	71: "[H] volume mixing ratio   (mol/mol)", \
	72: "[H2] volume mixing ratio  (mol/mol)", \
	73: "[He] volume mixing ratio  (mol/mol)", \
	74: "CO2 column (kg/m2)", \
	75: "N2 column  (kg/m2)", \
	76: "Ar column  (kg/m2)", \
	77: "CO column  (kg/m2)", \
	78: "O column   (kg/m2)", \
	79: "O2 column  (kg/m2)", \
	80: "O3 column  (kg/m2)", \
	81: "H column   (kg/m2)", \
	82: "H2 column  (kg/m2)", \
	83: "He column  (kg/m2)", \
	84: "Electron number density (particules/cm3)", \
	85: "Total electonic content (TEC) (particules/m2)"\
        }


def tidy_var_name(s):
    s = re.sub(r'\(.*?\)', '', s)
    s =  s.replace('.', '')
    s = s.strip()
    s = s.replace(' ','_')
    return s

mcd_var_field = {}
for key,value in mcd_var_field_ori.items():
    mcd_var_field[key] = tidy_var_name(value)


parser = argparse.ArgumentParser(description='MCD Extraction Scripts')
parser.add_argument("--input_path",'-i',help='input config path')
parser.add_argument("--save_path",'-s',help='data storge path')
args = parser.parse_args()

input_path = args.input_path
save_path = args.save_path

input_path = 'input.yaml'
save_path = './mcd_data'

custom_theme = Theme({'var_3d': 'bold black', 'var_2d': 'bold blue','success':'bold green','error':'bold red'})
console = Console(theme=custom_theme)

if not os.path.exists(save_path):
    print('save path: %s is not exist, mkdir for it!' %save_path)
    os.makedirs(save_path)

with open(input_path) as file:
    input_dict = yaml.load(file, Loader=yaml.FullLoader)

console.print(input_dict)

# format inputs
perturkey = input_dict['perturkey'] # default to no perturbation
seedin = 0 # perturbation seed (unused if perturkey=1)
gwlength = input_dict['gwlength'] # Gravity Wave length for perturbations (unused if perturkey=1)

if input_dict['choice_date']=='e':
    datekey=0 # Earth date
    loct=0 # local time must then also be set to zero
    day,month,year,hour,minute,second = \
    day=input_dict['day']
    month=input_dict['month']
    year=input_dict['year']
    hour=input_dict['hour']
    minute=input_dict['minute']
    second=input_dict['second']
    # now call Julian routine to convert to julian date
    (ier,xdate)=mcd.julian(month,day,year,hour,minute,second)
elif input_dict['choice_date']=='m':
    datekey=1
else:
    print('Wrong choice_date parameter!')

if (input_dict['extvarkey'] == 0) :
    extvarkeys = np.zeros(100,dtype=np.int32)
elif input_dict['extvarkey'] == 1:
    extvarkeys = np.zeros(100,dtype=np.int32)
    var_list_addon = []
    for idx in input_dict['extvarkeys']:
        if idx is not None:
            extvarkeys[np.int32(idx)-1] = 1
            var_list_addon.append(mcd_var_field[np.int32(idx)])
else:
  extvarkeys = np.ones(100,dtype=np.int32)

var_list_2d = []
if input_dict['is2d'] == 1:
    for idx in input_dict['2dvarkeys']:
        if idx is not None:
            extvarkeys[np.int32(idx)-1] = 1
            var_list_2d.append(mcd_var_field[np.int32(idx)])

# set default settings
nlon = 1 # number of
nlat = 1
nloc = 24
ls_start = input_dict['xdate_start']
ls_end = input_dict['xdate_end']
ls_step = input_dict['xdate_step']
Ls = np.arange(ls_start,ls_end+ls_step,ls_step,dtype=np.float64)
#zz=np.arange(0,36,1) # for normal use up to 100 km
zz=[0] # up to 180 km
nz = 1
nLs = len(Ls)
if input_dict['zkey'] == 4:
    xzs = 600*np.exp(-1*(1*zz)/3.0)
elif input_dict['zkey'] == 2:
    xzs = zz*3
elif input_dict['zkey'] == 3:
    xzs = zz*3
'''
lons = np.linspace(-180,180,nlon)
lats = np.linspace(-90,90,nlat)
locs = np.linspace(0,23,nloc)
'''
lons = [130]
lats = [40]
locs = np.linspace(0,23,nloc)
n_jobs = np.min([2,nLs])

# initialize data dict
def init_data_dict(var_list,shape):
    data_dict = {}
    for var_name in var_list:
        data_dict[var_name] = np.zeros(shape,dtype=np.float64)
    return data_dict

# save data to numpy array
def save_data(data_dict,var_list,data_slice,retrived_data,ierr):
    # retrived_data is a dict contains retrive data list from call mcd
    l = data_slice
    if len(l) == 4: #3D data set
        if ierr == 17:
            for var_name in var_list:
                data_dict[var_name][l[0],l[1],l[2],l[3]] = np.nan
        elif ierr == 0:
            for var_name in var_list:
                data_dict[var_name][l[0],l[1],l[2],l[3]] = retrived_data[var_name]
    elif len(l) == 3: # 2D data set
        for var_name in var_list: # no need to check under ground error
            data_dict[var_name][l[0],l[1],l[2]] = retrived_data[var_name]
    # return data_dict

# call MCD
data_dict = {}
var_list_3d = ["temp"]

console.print('extracing 3D vars:',style='var_3d')
console.print(var_list_3d,style='var_3d')
console.print('extracing 2D vars:',style='var_2d')
console.print(var_list_2d,style='var_2d')

data_dict = init_data_dict(var_list_3d,(nlon,nlat,nz,nloc,nLs))
if len(var_list_2d)>0:
    data_dict_2d = init_data_dict(var_list_2d,((nlon,nlat,nloc,nLs)))

def get_mcd_data(ils):
    # 只需要初始化U和V的数据结构
    data_dict_ = init_data_dict(var_list_3d, (nlon, nlat, nloc))
    
    for iloc in range(nloc):
        for ilon in range(nlon):
            for ilat in range(nlat):
                # 只需要处理地表1m高度 (iz=0)
                iz = 0
                (pres, dens, temp, zonwind, merwind, 
                 meanvar, extvar, seedout, ierr) = \
                mcd.call_mcd(input_dict['zkey'], xzs[iz],
                             lons[ilon], lats[ilat],
                             input_dict['hrkey'],
                             datekey, np.mod(Ls[ils], 360),
                             np.mod(locs[iloc], 24),
                             dset, input_dict['dust'],
                             perturkey, seedin, gwlength, extvarkeys)
                
                # 只保存U和V数据
                rdata_dict = {}
                rdata_dict['temp'] = temp
                
                # 保存数据 - 注意现在只有3个维度 (经度, 纬度, 时间)
                for var_name in var_list_3d:
                    data_dict_[var_name][ilon, ilat, iloc] = rdata_dict[var_name]
    
    return data_dict_


# with Pool(processes=n_jobs) as pool:
#     input_values = [ils for ils in range(nLs)]
#     results = pool.map(get_mcd_data,input_values)
#     results = list(tqdm(pool.imap(get_mcd_data,input_values),total=nLs,desc='loop in Ls'))

# with Pool(processes=n_jobs) as pool:
#     input_values = [ils for ils in range(nLs)]
# #     results = pool.map(get_mcd_data,input_values)
#     results = list(tqdm(pool.imap(get_mcd_data,input_values),total=nLs,desc='loop in Ls'))

start_time = time.time()
results = Parallel(n_jobs=n_jobs,backend='multiprocessing')(delayed(get_mcd_data)(ils) for ils in tqdm(range(nLs),desc='loop in Ls with %i workers' %n_jobs))
end_time = time.time()
elapsed_time = end_time - start_time
console.print('elapsed_time of get_mcd_data: %.2f min' %(elapsed_time/60),style='success')

# results = []
# for ils in tqdm(range(nLs)):
#     results.append(get_mcd_data(ils))


data_dict_3d_list, data_dict_2d_list = zip(*results)

for i in tqdm(range(nLs),desc='merge results'): # should be the same length as the Ls
    for var_name in var_list_3d:
        data_dict[var_name][:,:,:,:,i] = data_dict_3d_list[i][var_name]

    for var_name in var_list_2d:
        data_dict_2d[var_name][:,:,:,i] = data_dict_2d_list[i][var_name]

# for var_name in var_list_3d:
#     data_dict[var_name] = xr.DataArray(data_dict[var_name],dims=['lontitude','latitude','level','ut','Ls'])

# for var_name in var_list_2d:
#     data_dict_2d[var_name] = xr.DataArray(data_dict_2d[var_name],dims=['lontitude','latitude','ut','Ls'])





    # pool = ThreadPool(n_jobs)
    # out = pool.map(get_mcd_data,input_values)
    # pool.close()
    # pool.join()
    # with Pool(processes=n_jobs) as pool:
    #     results = pool.map(get_mcd_data,input_values)
    # out = Parallel(n_jobs=n_jobs,verbose=5)(delayed(get_mcd_data)(ils,iloc) for iloc in range(nloc))
    # get_mcd_data(ils,iloc)

# for ils in tqdm(range(nLs),desc='Loop in Ls'):
#     for iloc in range(nloc):
#         for ilon in range(nlon):
#             for ilat in range(nlat):
#                 for iz in range(nz):
#                     (pres, dens, temp, zonwind, merwind, \
#                     meanvar, extvar, seedout, ierr) \
#                     = \
#                     mcd.call_mcd(input_dict['zkey'],xzs[iz],
#                     lons[ilon],lats[ilat],\
#                     input_dict['hrkey'], \
#                     datekey,Ls[ils],\
#                     np.mod(locs[iloc]+lons[ilon]/360*24,24),\
#                     dset,input_dict['dust'], \
#                     perturkey,seedin,gwlength,extvarkeys )

#                     # save data
#                     rdata_dict = {}
#                     rdata_dict['pres'] = pres
#                     rdata_dict['dens'] = dens
#                     rdata_dict['temp'] = temp
#                     rdata_dict['U'] = zonwind
#                     rdata_dict['V'] = merwind
#                     if len(var_list_3d)>5:
#                         for idx,var_name in enumerate(var_list_3d[5:]):
#                             rdata_dict[var_name] = extvar[np.int32(input_dict['extvarkeys'][idx])-1]
#                     save_data(data_dict,var_list_3d,[ilon,ilat,iz,iloc,ils],rdata_dict,ierr)

#                 if len(var_list_2d)>0:
#                     for idx,var_name in enumerate(var_list_2d):
#                         rdata_dict[var_name] = extvar[np.int32(input_dict['2dvarkeys'][idx])-1]
#                     # if iz == nz-1:
#                     save_data(data_dict_2d,var_list_2d,[ilon,ilat,iloc,ils],rdata_dict,ierr)

# check if there is a already saved files with the same config
def save_file_checker(file_path):
    if os.path.exists(file_path):
        console.print('There is already a file: %s' %file_path,style='error')
        console.print('The file will be covered!',style='error')
        os.remove(file_path)
    console.print('Preparing save to %s' %file_path)

# save data to netcdf file
'''
save_file_checker( os.path.join(save_path,'3D_vars_ds_%s_LS_%.2f-%.2f_rev1.1.nc'\
    %(input_dict['dust'],Ls[0],Ls[-1])))
save_file_checker(os.path.join(save_path,'2D_vars_ds_%s_LS_%.2f-%.2f_rev1.1.nc'\
    %(input_dict['dust'],Ls[0],Ls[-1])))
'''
save_file_checker( os.path.join(save_path,'3D_temp.nc'\
    %(input_dict['dust'],Ls[0],Ls[-1])))
save_file_checker(os.path.join(save_path,'2D_vars_ds_%s_LS_%.2f-%.2f_rev1.1.nc'\
    %(input_dict['dust'],Ls[0],Ls[-1])))
# 修改数据保存部分
nc_save_dict_3d = {}
for var_name in var_list_3d:
    # 维度变为 (经度, 纬度, 时间, Ls)
    nc_save_dict_3d[var_name] = (('longitude', 'latitude', 'ut', 'Ls'), data_dict[var_name])

save_dataset_3d = xr.Dataset(data_vars=nc_save_dict_3d,
                             coords={'longitude': lons, 
                                     'latitude': lats, 
                                     'ut': locs, 
                                     'Ls': Ls},
                             attrs=dict(description="地表1m处水平风速 - MCD v6.1"))


console.print('Done!~',style='success')
console.print(':thumbs_up:', 'Nice work!')
console.print('Check the data path: %s' %save_path,style='success')
