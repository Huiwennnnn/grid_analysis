import os
import pypsa
import geopandas as gpd
import pandas as pd
import math
import json
from shapely.geometry import Point
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

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
        net.add("Bus",name=f"{MV_case_id}_node_{node['osmid']}",v_nom=20,x=node['x'],y=node['y'],v_mag_pu_max=1.03,v_mag_pu_min=0.97,carrier='AC')
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
        net.add("Line",name=f"{MV_case_id}_{edge['u']}_{edge['v']}",bus0=f"{MV_case_id}_node_{edge['u']}",bus1=f"{MV_case_id}_node_{edge['v']}",r=edge['r'],x=edge['x'],s_nom=edge['s_nom'],b=edge['b'])
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
        net.add("Bus",name=f"{case_id}_node_{node['osmid']}",v_nom=0.4,x=node['x'],y=node['y'],v_mag_pu_max=1.03,v_mag_pu_min=0.97,carrier='AC')
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
        net.add("Generator",name=f'External_grid_{case_id}',bus=f"{resp_MV_case_id}_osmid_{resp_MV_osmid}",control="Slack")
        
    for index, edge in grid_edges.iterrows():
        net.add("Line",name=f"{case_id}_{edge['u']}_{edge['v']}",bus0=f"{case_id}_node_{edge['u']}",bus1=f"{case_id}_node_{edge['v']}",r=edge['r'],x=edge['x'],s_nom=edge['s_nom'],b=edge['b'])
    
    print(f"Finish building LV {case_id}")
    return net