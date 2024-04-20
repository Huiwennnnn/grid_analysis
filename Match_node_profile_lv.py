import os
import pandapower as pp
import pandas as pd
import numpy as np
import pickle
import folium
from folium import FeatureGroup

import geopandas as gpd
from shapely.geometry import Point,Polygon
import numba
import matplotlib.pyplot as plt
import imageio
from pandapower.control import ConstControl
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries
import pandapower.topology as top
import ast
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)  # or specify a number if you want a limit
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)



# Basic Setting


LV_list_converged = ['211-1_0_4','39-1_1_2','214-1_0_2','214-1_1_5','216-2_1_3','214-1_2_2','216-2_0_2','216-1_1_3','216-1_2_4','216-3_0_5','225-2_0_3','225-4_0_3','225-1_0_6','225-3_0_6', '298-1_1_4','298-4_0_2','230-106_0_2','298-7_1_3','298-7_0_3','298-6_0_2','298-3_0_3','227-12_0_5','227-7_0_5','227-10_1_3','230-150_0_2','230-202_0_2','230-180_0_4','230-197_0_5','230-200_0_4','230-202_1_3','230-201_0_6','230-211_0_2','230-212_0_3','230-212_1_3','230-108_0_6','227-13_0_3','227-14_0_4','227-6_1_4','298-9_0_5']
LV_list_converged_1p05 = ['39-1_0_4','225-2_1_5','227-11_0_4','227-13_1_3', '227-1_0_5','298-5_0_5']
LV_list_converged_1p1 = ['298-2_0_5','298-6_1_4','227-10_0_6','225-5_0_5','227-3_0_5']
LV_list_converged_1p3 = ['298-8_0_7','227-9_0_5','227-8_0_10'] 


grid = "369_0"
scenario_year = 2050
# weekday_dict = {'Monday':(1,10), 'Tuesday':(1,11),'Wednesday':(1,12),'Thursday':(1,13),'Friday':(1,14),'Saturday':(1,15),'Sunday':(1,16)}
day_start_ts = pd.to_datetime(f"{scenario_year}-01-09 00:00:00")
day_start = day_start_ts.day
day_end_ts = pd.to_datetime(f"{scenario_year}-01-10 00:00:00")
monitor_hr = int((day_end_ts - day_start_ts).total_seconds()/3600)
empty_power_list = [0]*monitor_hr
weekday = "Sunday"
path = f"{grid}/{scenario_year}_{weekday}_01_09_test2"
lv_pf = 0.97
mv_pf = 0.9


def generate_zero_list(row, hr):
    if pd.isna(row['emob_prof']):
        return [0] * hr
    else:
        return row['emob_prof']
        
def sum_lists(series):
    return [sum(values)/1000 for values in zip(*series)] # convert from kW to MW for pandapower

def get_load_bus_geometries(net):
    # Identifying buses with connected loads
    load_buses = net.load['bus'].unique()
    # Extracting coordinates of these buses
    load_bus_coords = net.bus_geodata.loc[load_buses]
    # Create a list of Point geometries
    load_coords = [Point(xy) for xy in zip(load_bus_coords['x'], load_bus_coords['y'])]
    return load_coords

def create_controllers(net, ds_P, ds_Q):
    ConstControl(net,element='load',variable='p_mw',element_index=net.load.index.to_list(),data_source=ds_P,profile_name=net.load.index.to_list())
    ConstControl(net,element='load',variable='q_mvar',element_index=net.load.index.to_list(),data_source=ds_Q,profile_name=net.load.index.to_list())
    return net
def create_output_writer(net,time_steps,output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ow = OutputWriter(net,time_steps,output_path=output_dir,output_file_type=".csv")
    # net.res_bus['vm_kv'] = net.res_bus['vm_pu']*net.bus.vn_kv.loc[net.res_bus.index]
    # ow.log_variable('res_bus','vm_kv')
    ow.log_variable('res_bus','p_mw')
    ow.log_variable('res_bus','vm_pu')
    ow.log_variable('res_line','loading_percent')
    ow.log_variable('res_trafo','loading_percent')
    return ow

def run_emob_timeseries(network,P,Q,output_dir):
    # create test net and load data source
    net = network
    n_timesteps =monitor_hr
    ds_P = DFData(P)
    ds_Q = DFData(Q)

    # create controllers (to control P values of the load and the sgen)
    net = create_controllers(net,ds_P=ds_P,ds_Q=ds_Q)

    # time steps to be calculated. Could also be a list with non-consecutive time steps
    time_steps = range(0,n_timesteps)

    # the output writer with the desired results to be stored to files.
    ow = create_output_writer(net, time_steps, output_dir)
    # the main time series function
    run_timeseries(net,time_steps,init='results',algorithm='nr',max_iter=5000)

# Load emob data
with open(f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/mobility/{path}/grid_{grid}_matched_{day_start_ts}_{monitor_hr}_{scenario_year}.pkl","rb") as emob:
    emob = pickle.load(emob)
emob_geometry = [Point(xy) for xy in zip(emob.end_x, emob.end_y)]
emob_gdf = gpd.GeoDataFrame(emob, geometry=emob_geometry)
emob_gdf=emob_gdf.set_crs('epsg:2056')

for folder in LV_list_converged_1p3:
    # Load network as geodataframe
    network = pp.from_json(f'/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/modified_swiss_pdg/ppnetwork/{folder}_ppnetwork.json')
    node_geometry = [Point(xy) for xy in zip(network.bus_geodata.x,network.bus_geodata.y)]
    node_gdf = gpd.GeoDataFrame(network.bus,geometry=node_geometry)
    node_gdf = node_gdf.set_crs('epsg:2056')
    load_gdf = gpd.GeoDataFrame(network.load)
    load_gdf = gpd.GeoDataFrame(load_gdf.merge(node_gdf[['geometry','vn_kv']], left_on='bus', right_index=True, how='left'))
    LV_load_gdf = load_gdf.loc[load_gdf.vn_kv==0.4] # Filter for LV loads

    # Assign profile to node
    emob_assigned = gpd.sjoin_nearest(emob_gdf, LV_load_gdf, how='left', max_distance=100, distance_col="distance") # Assign emob profile only to LV nodes
    
    # Sum up profile at each node
    emob_prof = emob_assigned.groupby('index_right')['optimized_power_list'].apply(sum_lists)
    load_gdf['emob_prof']=load_gdf.index.map(emob_prof)
    load_gdf.loc[load_gdf['emob_prof'].isnull(), 'emob_prof'] = load_gdf[load_gdf['emob_prof'].isnull()].apply(generate_zero_list,hr=monitor_hr, axis=1)
    emob_P = load_gdf[['emob_prof']].copy()['emob_prof'].apply(pd.Series).T
    emob_Q = emob_P*np.tan(np.arccos(lv_pf))

    # Creat daily base profile respective to el_dmd
    base = pd.DataFrame()
    baseload = pd.read_csv(f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/extracted_data/Load_Solar.csv")
    base['timeSeries'] = baseload.loc[(baseload.name=='CH_load_ConvRail') & (baseload.year==2050)]['timeSeries'].apply(ast.literal_eval)
    base=base.explode('timeSeries').reset_index(drop=True)

    base.index = pd.date_range(start=f"2050-01-01", end=f"2051-01-01", periods=8761, inclusive='left')
    day_base = base.loc[day_start_ts:day_end_ts][:-1]
    day_base_max = day_base.max()
    day_base_norm = (day_base/day_base_max).rename(columns={'timeSeries':'norm'}).reset_index(drop=True)

    # Multiply with peak power to create daily base profile
    base_load_P_array = np.outer(day_base_norm['norm'],load_gdf['p_mw'])
    base_load_P = pd.DataFrame(base_load_P_array, index=day_base_norm['norm'].index, columns=load_gdf['p_mw'].index)
    power_factors = load_gdf['vn_kv'].map({0.4: lv_pf, 20: mv_pf})
    base_load_Q = base_load_P*np.tan(np.arccos(power_factors))

    # Add up all loads
    load_P = emob_P+base_load_P
    load_Q = emob_Q+base_load_Q

    output_dir = f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/mobility/{path}/{folder}"
    network.ext_grid.vm_pu=1.3
    run_emob_timeseries(network,load_P,load_Q,output_dir)

    res_bus_dict = {}
    res_line_dict = {}
    res_trafo_dict = {}
    res_ext_grid_dict = {}
    hourly_images=[]
    for t in range(monitor_hr):
        network.load.loc[:,'p_mw'] = load_P.loc[t]
        network.load.loc[:,'q_mvar']=load_Q.loc[t]
        pp.runpp(network, init='results',algorithm='nr',max_iter=5000)
        network.res_bus['vm_kv'] = network.res_bus['vm_pu']*network.bus.vn_kv.loc[network.res_bus.index]
        print(f'timestep:{t}')
        res_bus_dict[t]=network.res_bus
        res_line_dict[t]=network.res_line
        res_trafo_dict[t]=network.res_trafo
        res_ext_grid_dict[t]=network.res_ext_grid
        hour_res = pp.plotting.pf_res_plotly(network,line_width=1,bus_size=5,climits_volt=(0.5, 1.3), climits_load=(0, 130), auto_open=False)
        os.makedirs(f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/mobility/{path}/{folder}/res_plot",exist_ok=True)
        hour_res.write_html(f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/mobility/{path}/{folder}/res_plot/{weekday}_{t}.html")
        hour_res.write_image(f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/mobility/{path}/{folder}/res_plot/{weekday}_{t}.png")
        hourly_images.append(imageio.imread(f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/mobility/{path}/{folder}/res_plot/{weekday}_{t}.png"))
    imageio.mimsave(f'/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/mobility/{path}/{folder}/res_plot/hourly_plot.gif', hourly_images, duration=0.5)
    with open(f'/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/mobility/{path}/{folder}/res_bus_dict.pkl', 'wb') as f:
        pickle.dump(res_bus_dict, f)
    with open(f'/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/mobility/{path}/{folder}/res_line_dict.pkl', 'wb') as f:
        pickle.dump(res_line_dict, f)
    with open(f'/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/mobility/{path}/{folder}/res_trafor_dict.pkl', 'wb') as f:
        pickle.dump(res_trafo_dict, f)
    with open(f'/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/mobility/{path}/{folder}/res_ext_grid_dict.pkl', 'wb') as f:
        pickle.dump(res_ext_grid_dict, f)




