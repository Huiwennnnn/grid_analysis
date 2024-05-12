import pypsa
import geopandas as gpd
import math
import json
import pickle
import matplotlib.pyplot as plt
from shapely.geometry import Point,LineString
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def read_LV_zone_dict():
    # Data Processing
    LV_zone_dict = {}
    LV_path = ("LV/")

    with open("/Users/huiwen/Desktop/OneDrive/ETHZ/Thesis/file_folder_lv.json", 'r') as file:
        LV_zone_dict = json.load(file)
    return LV_zone_dict

def read_MV_nodes_all():
    MV_nodes_all = gpd.read_file('/Users/huiwen/Desktop/OneDrive/ETHZ/Thesis/MV_nodes_all.geojson')
    # Convert dtype
    MV_nodes_all['osmid'] = MV_nodes_all['osmid'].astype(int)
    MV_nodes_all['consumers'] = MV_nodes_all['consumers'].astype(bool)
    return MV_nodes_all

def read_MV_edges_all():
    MV_edges_all = gpd.read_file('/Users/huiwen/Desktop/OneDrive/ETHZ/Thesis/MV_edges_all.geojson')
    return MV_edges_all

def get_Q(P, pf):
    """
    Calculate the reactive power (Q) given the real power (P) and the power factor (PF).
    pf: Power factor, ranging from -1(leading) to 1(lagging).
    """
    S = P / pf
    Q = math.sqrt(S ** 2 - P ** 2)
    if pf < 0:
        Q = -Q
    return Q

def build_MV_LV_net(MV_case_id):
    HV_bus_name = ""
    LV_zone_dict=read_LV_zone_dict()
    MV_nodes_all = read_MV_nodes_all()
    LV_pf=0.97
    MV_pf=0.9
    MV_zone_trafo = {"trafo_type": "25 MVA 110/20 kV"}
    # Read MV Grid Data
    MV_nodes, MV_edges = gpd.read_file(f"/Users/huiwen/Desktop/OneDrive/ETHZ/Thesis/MV/{MV_case_id}_nodes"), gpd.read_file(
        f"/Users/huiwen/Desktop/OneDrive/ETHZ/Thesis/MV/{MV_case_id}_edges")

    # Convert dtype
    MV_nodes['osmid'] = MV_nodes['osmid'].astype(int)
    MV_nodes['consumers'] = MV_nodes['consumers'].astype(bool)
    MV_nodes=MV_nodes.to_crs(epsg=4326)
    MV_nodes['x'],MV_nodes['y']=MV_nodes.geometry.x,MV_nodes.geometry.y

    net=pypsa.Network()
    for index,node in MV_nodes.iterrows():
        net.add("Bus",name=f"{MV_case_id}_node_{node['osmid']}",v_nom=20,x=node['x'],y=node['y'],carrier='AC')#v_mag_pu_max=1.03,v_mag_pu_min=0.97
        # Place holder for base load
        # MV consumer
        if (node['consumers']==True) and (node['lv_grid']=='-1'):
            net.add("Load",name=f"base_load_{MV_case_id}_{node['osmid']}",bus=f"{MV_case_id}_node_{node['osmid']}",p_set=node['el_dmd'],q_set=get_Q(node['el_dmd'],MV_pf))
        # Connvected to LV consumers
        if (node['consumers']==True) and (node['lv_grid']!='-1'):
            LV_case_id = node['lv_grid']
            LV_zone = LV_zone_dict[node['lv_grid']]
            resp_MV = MV_nodes_all[MV_nodes_all['lv_grid'] == LV_case_id]
            net = build_LV_net(LV_case_id,LV_zone,net,resp_MV)
        # Add transformer
        if node['source']:
            net.add("Bus",name=f"HV_bus_{MV_case_id}_{node['osmid']}",v_nom=110)
            HV_bus_name = f"HV_bus_{MV_case_id}_{node['osmid']}"
            net.add("Transformer",name=f"HV_{MV_case_id}_{node['osmid']}",bus0=f"HV_bus_{MV_case_id}_{node['osmid']}",bus1=f"{MV_case_id}_node_{node['osmid']}",type=MV_zone_trafo['trafo_type'])
    net.add("Generator",name=f"External_grid_{MV_case_id}",bus=HV_bus_name,control="Slack")
    for index,edge in MV_edges.iterrows():
        net.add("Line",name=f"{MV_case_id}_{edge['u']}_{edge['v']}",bus0=f"{MV_case_id}_node_{edge['u']}",bus1=f"{MV_case_id}_node_{edge['v']}",r=edge['r'],x=edge['x'],s_nom=edge['s_nom'],b=edge['b'],s_nom_extendable=True)
    print(f"Finish building MV {MV_case_id}")
    return net


def build_LV_net(case_id,zone=None,net=None,resp_MV=None):
    if zone is None:
        LV_zone_dict=read_LV_zone_dict()
        zone = LV_zone_dict[case_id]
    expand = False if net is None else True
    LV_pf=0.97
    # Line types
    if zone.endswith('Urban'):
        LV_zone_line = {"line_type": "NAYY 4x240 SE", "line_type_in_lib": False, "r_ohm_per_km": 0.125,
                        "x_ohm_per_km": 0.08, "c_nf_per_km": 260, "g_us_per_km": 81.995568, "max_i_ka": 0.364, "type": "cs"} # c_nf_per_km not correct
    else:
        LV_zone_line = {"line_type": "NAYY 4x150 SE", "line_type_in_lib": True,"r_ohm_per_km": 0.208,
                        "x_ohm_per_km": 0.08, "c_nf_per_km": 261, "g_us_per_km": 81.995568, "max_i_ka": 0.27, "type": "cs"}

    if zone.endswith('rban'):
        LV_zone_trafo = {"trafo_type": "0.63 MVA 20/0.4 kV"}
    else:
        LV_zone_trafo = {"trafo_type": "0.25 MVA 20/0.4 kV"}

    # Read LV Grid Data
    grid_nodes, grid_edges = gpd.read_file(f"/Users/huiwen/Desktop/OneDrive/ETHZ/Thesis/LV/{zone}/{case_id}_nodes"), gpd.read_file(
        f"/Users/huiwen/Desktop/OneDrive/ETHZ/Thesis/LV/{zone}/{case_id}_edges")
    # Convert dtype
    grid_nodes['osmid'] = grid_nodes['osmid'].astype(int)
    grid_nodes['consumers'] = grid_nodes['consumers'].astype(bool)

    grid_nodes = grid_nodes.to_crs(epsg=4326)
    grid_nodes['x'] = grid_nodes.geometry.x
    grid_nodes['y'] = grid_nodes.geometry.y

    if not expand:
        net = pypsa.Network()
        MV_nodes_all = read_MV_nodes_all()
        resp_MV = MV_nodes_all[MV_nodes_all['lv_grid'] == case_id]
    else:
        net = net
    for index, node in grid_nodes.iterrows():
        resp_MV_case_id,resp_MV_osmid=str(resp_MV['MV_case_id'].iloc[0]),str(resp_MV['osmid'].iloc[0])
        # Add bus
        net.add("Bus",name=f"{case_id}_node_{node['osmid']}",v_nom=0.4,x=node['x'],y=node['y'],carrier='AC')#v_mag_pu_max=1.03,v_mag_pu_min=0.97
        # Placeholder for base load
        if node['consumers']==True:
            net.add("Load",name=f"base_load_{case_id}_{node['osmid']}",bus=f"{case_id}_node_{node['osmid']}",p_set=node['el_dmd'],q_set=get_Q(node['el_dmd'],LV_pf))
 
        # Add Transformer
        if node['source']:
            if not expand:
                net.add("Bus",name=f"{resp_MV_case_id}_node_{resp_MV_osmid}",v_nom=20,x=resp_MV['x'].iloc[0],y=resp_MV['y'].iloc[0])
            net.add("Transformer",name=f"{resp_MV_case_id}_{case_id}_{node['osmid']}",bus0=f"{resp_MV_case_id}_node_{resp_MV_osmid}",bus1=f"{case_id}_node_{node['osmid']}",type=LV_zone_trafo['trafo_type'])
    
    # Add external network
    if not expand:
        net.add("Generator",name=f'External_grid_{case_id}',bus=f"{resp_MV_case_id}_node_{resp_MV_osmid}",control="Slack")
        
    for index, edge in grid_edges.iterrows():
        net.add("Line",name=f"{case_id}_{edge['u']}_{edge['v']}",bus0=f"{case_id}_node_{edge['u']}",bus1=f"{case_id}_node_{edge['v']}",r=edge['r'],x=edge['x'],s_nom=edge['s_nom'],b=edge['b'],s_nom_extendable=True)
    
    print(f"Finish building LV {case_id}")
    return net


def get_net_boundary(net, buffer=50, ratio=0.5,save=False,case_id=None): # Winterthur ratio=0.5, 260_0 ratio=0.2
    net_bus_gdf = gpd.GeoDataFrame(net.buses,geometry=gpd.points_from_xy(net.buses.x,net.buses.y),crs="epsg:4326")
    net_bus_gdf = net_bus_gdf.to_crs(epsg=2056) 
    net_bus_gs = net_bus_gdf.geometry.buffer(buffer).unary_union
    net_bus_gs = gpd.GeoSeries(net_bus_gs, crs="epsg:2056")
    area = net_bus_gs.concave_hull(ratio=ratio, allow_holes=True)
    fig, ax = plt.subplots()
    area.plot(ax=ax, color='white', edgecolor='blue')
    net_bus_gdf.plot(ax=ax, marker='o', color='red', markersize=3)
    if save:
        area.to_file(f'/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/modified_swiss_pdg/boundary/{case_id}_boundary.geojson', driver="GeoJSON")
        plt.savefig(f'/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/modified_swiss_pdg/boundary/{case_id}_boundary_plot.jpeg')
    return area

# EV functions
def get_lv_consumer_bus(net):
    lv_consumer_bus = net.buses.loc[
        net.buses.index.isin(net.loads.loc[net.loads.index.str.contains("base")].bus) & (net.buses.v_nom == 0.4)]
    lv_consumer_bus_gdf = gpd.GeoDataFrame(lv_consumer_bus.index,
                                           geometry=gpd.points_from_xy(x=lv_consumer_bus.x, y=lv_consumer_bus.y),
                                           crs="epsg:4326")
    lv_consumer_bus_gdf = lv_consumer_bus_gdf.to_crs("epsg:2056")
    lv_consumer_bus_gdf.set_index('Bus', inplace=True)
    return lv_consumer_bus_gdf


def load_emob(path, grid, day_start_ts, monitor_hr, scenario_year):
    with open(
            f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/mobility/{path}/grid_{grid}_matched_{day_start_ts}_{monitor_hr}_{scenario_year}.pkl",
            "rb") as emob:
        emob = pickle.load(emob)
    emob_geometry = [Point(xy) for xy in zip(emob.end_x, emob.end_y)]
    emob_gdf = gpd.GeoDataFrame(emob, geometry=emob_geometry, crs='epsg:2056')
    return emob_gdf



def add_ev_profile(net,path,grid,day_start_ts,monitor_hr,scenario_year):
    LV_pf=0.97
    MV_pf=0.9
    emob_gdf = load_emob(path,grid,day_start_ts,monitor_hr,scenario_year)
    lv_consumer_bus_gdf = get_lv_consumer_bus(net)
    # Spatial Join Nearest lv_consumer_bus_gdf to the EV parking location
    emob_assigned = gpd.sjoin_nearest(emob_gdf, lv_consumer_bus_gdf, how='left', max_distance=50, distance_col="distance") # Assign emob profile only to LV nodes
    emob_assigned = emob_assigned.loc[~pd.isnull(emob_assigned.index_right)] #filter for successfully mapped EVs
    # map bus geometry
    emob_assigned.rename(columns={"index_right":"Bus"},inplace=True)
    emob_assigned['bus_geometry'] = emob_assigned['Bus'].map(lv_consumer_bus_gdf['geometry']) 
    # create ev bus connection string geoemtry
    emob_assigned['map_bus'] = emob_assigned.apply(lambda row:LineString([row['geometry'], row['bus_geometry']]),axis=1)

    """
    Plot mapping result
    """
    fig, ax = plt.subplots(figsize=(20,20))
    emob_assigned['geometry'].plot(ax=ax,marker='x',color='blue',markersize=3)
    emob_assigned['bus_geometry'].plot(ax=ax, marker='o', color='red', markersize=3)
    emob_assigned['map_bus'].plot(ax=ax,color='green')
    plt.show()


    ev_load = emob_assigned.copy().drop(columns=['geometry','bus_geometry','map_bus'],axis=1)
    for index,ev_profile in ev_load.iterrows():
        if ev_profile['process_cnt']!=0: # Only add ev profiles with charge/discharge actions
            net.add("Load",name=f"ev_{ev_profile['person']}_{index}",bus=f"{ev_profile['Bus']}",p_set=[p/1000 for p in ev_profile['optimized_power_list']],q_set=[p/1000*np.tan(np.arccos(LV_pf)) for p in ev_profile['optimized_power_list']])
    return net


# Base Load
def load_real_base_profile():
    winti_profile = pd.DataFrame()
    # Winit Load Profile with base load of year 2024
    winti = pd.read_csv(f"./winterthur_load.csv")
    winti.zeitpunkt=pd.to_datetime(pd.to_datetime(winti.zeitpunkt).dt.strftime('%Y-%m-%d %H:%M:%S'))
    winti['datehour'] = winti.zeitpunkt.dt.strftime('%Y-%m-%d %H:00:00')
    winti_hourly = winti.groupby(['datehour']).bruttolastgang_kwh.sum()#kWh
    return winti_hourly

def create_normalized_base_profile(day_start_ts):
    winti_profile = pd.DataFrame()
    winti_hourly = load_real_base_profile()
    winti_day_start = day_start_ts.replace(year=2023)
    selection_start_date = winti_day_start - pd.DateOffset(days=10)
    selection_end_date = winti_day_start + pd.DateOffset(days=10)
    for selected_date in pd.date_range(start=selection_start_date, end=selection_end_date):
        day_base_start = pd.to_datetime(pd.to_datetime(selected_date).strftime('%Y-%m-%d %H:%M:%S'))
        day_base_end = day_base_start+timedelta(hours=23)
        day_base = winti_hourly.loc[str(day_base_start):str(day_base_end)]
        day_base_norm = pd.DataFrame((day_base/day_base.max())).rename(columns={'bruttolastgang_kwh':'norm'}).reset_index(drop=True)
        winti_profile = pd.concat([winti_profile, day_base_norm], ignore_index=True,axis=1)
    return winti_profile

def add_base_load(net,day_start_ts,monitor_hr):
    MV_pf=0.9
    LV_pf=0.97
    net.snapshots = pd.date_range(day_start_ts,freq='h',periods=monitor_hr)
    winti_profile=create_normalized_base_profile(day_start_ts)
    base_load = net.loads.loc[net.loads.index.str.contains('base')]
    for idx,base in base_load.iterrows():
        v_lvl = net.buses.loc[net.loads.loc[idx].bus].v_nom
        if v_lvl==0.4:
            pf = LV_pf
        else:
            pf = MV_pf
        el_dmd = net.loads.loc[idx].p_set
        rand_norm_profile = winti_profile[np.random.choice(winti_profile.columns)].to_list()
        net.loads_t.p_set[idx]=[el_dmd*p for p in rand_norm_profile]
        net.loads_t.q_set[idx]=[el_dmd*p*np.tan(np.arccos(pf)) for p in rand_norm_profile]
    return net

# PV Generation
