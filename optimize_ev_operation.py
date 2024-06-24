import os
import imageio.v2 as imageio
import cartopy.crs as ccrs
import pypsa
import xarray as xr
from pypsa.descriptors import Dict
import pypsa_create_basic_network as basic
import pypsa_pf_stat as psastat
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


LV_list = ['211-1_0_4', '39-1_1_2', '214-1_0_2', '227-10_0_6',
           '214-1_1_5', '216-2_1_3', '214-1_2_2', '216-2_0_2', '216-1_1_3',
           '216-1_2_4', '216-3_0_5', '39-1_0_4', '225-2_0_3', '225-4_0_3',
           '225-1_0_6', '225-2_1_5', '225-3_0_6', '298-1_1_4', '298-4_0_2',
           '225-5_0_5', '230-106_0_2', '298-2_0_5', '298-7_1_3', '298-7_0_3',
           '298-5_0_5', '298-6_0_2','298-3_0_3', '298-8_0_7', '227-12_0_5',
           '227-7_0_5', '227-10_1_3', '227-11_0_4', '230-150_0_2',
           '227-9_0_5', '230-202_0_2', '230-180_0_4', '230-197_0_5',
           '230-200_0_4', '230-202_1_3', '230-201_0_6', '230-211_0_2',
           '230-212_0_3', '230-212_1_3', '230-108_0_6', '227-13_0_3',
           '227-14_0_4', '227-8_0_10', '227-13_1_3', '227-1_0_5', '227-6_1_4',
           '227-3_0_5', '298-9_0_5', '298-6_1_4', '298-4_1_5']


MV_feeder = False
grid = "369_0"
folder = '211-1_0_4'
scenario_year = 2050
weekday = "Friday"
day_start_ts = pd.to_datetime(f"{scenario_year}-01-07 00:00:00")
day_start = day_start_ts.day
day_end_ts = pd.to_datetime(f"{scenario_year}-01-08 00:00:00")
# month = day_start_ts.month
monitor_hr = int((day_end_ts - day_start_ts).total_seconds() / 3600)
path_controlled = f"{grid}/{scenario_year}_{weekday}_01_07_controlled"
path_uncontrolled = f"{grid}/{scenario_year}_{weekday}_01_07_uncontrolled"
lv_pf = 0.97
mv_pf = 0.9
experiment = 'ev_pv_optimize'

def make_save_path(MV_feeder, path_controlled, path_uncontrolled, folder=None, experiment=None):
    if MV_feeder:
        os.makedirs(
            f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/mobility/{path_controlled}/{experiment}",
            exist_ok=True)
        os.makedirs(
            f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/mobility/{path_uncontrolled}/{experiment}",
            exist_ok=True)
        save_path_controlled = f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/mobility/{path_controlled}/{experiment}"
        save_path_uncontrolled = f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/mobility/{path_uncontrolled}/{experiment}"
    else:
        os.makedirs(
            f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/mobility/{path_controlled}/{folder}/{experiment}",
            exist_ok=True)
        os.makedirs(
            f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/mobility/{path_uncontrolled}/{folder}/{experiment}",
            exist_ok=True)
        save_path_controlled = f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/mobility/{path_controlled}/{folder}/{experiment}"
        save_path_uncontrolled = f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/mobility/{path_uncontrolled}/{folder}/{experiment}"
    return save_path_controlled, save_path_uncontrolled

def build_net(MV_feeder, folder, grid, day_start_ts, monitor_hr, scenario_year,override_components,override_component_attrs):
    if MV_feeder:
        net = basic.build_MV_LV_net(MV_case_id=grid,override_components=override_components,override_component_attrs=override_component_attrs);
    else:
        net = basic.build_LV_net(case_id=folder,override_components=override_components,override_component_attrs=override_component_attrs);

    net_base = basic.add_base_load(net, day_start_ts, monitor_hr);
    net_hp = basic.add_hp_profile(net_base.copy(),day_start_ts);
    net_pv = basic.add_pv_profile(net_hp.copy(), day_start_ts);
    net_controlled = basic.add_ev_profile_storage(net_pv.copy(), path_controlled, grid, day_start_ts, monitor_hr, scenario_year)
    net_uncontrolled = basic.add_ev_profile_storage(net_pv.copy(), path_uncontrolled, grid, day_start_ts, monitor_hr,
                                            scenario_year)
    return net_controlled, net_uncontrolled


def override_storage_unit():
    override_components = pypsa.components.components.copy()
    override_component_attrs = Dict(
        {k: v.copy() for k, v in pypsa.components.component_attrs.items()}
    )
    override_component_attrs['StorageUnit'].loc['parking_hr']=["static or series","hr","0","Ev hourly parking time connected at this bus","Input(optional)"]
    override_component_attrs['StorageUnit'].loc['day_end_SoC']=["static","kWh","0","Emob day end SoE","Input(optional)"]
    override_component_attrs['StorageUnit'].loc['next_consumption'] = ["static","kWh","0","Energy next 2 trips required","Input(optional)"]
    override_component_attrs['StorageUnit'].loc['park_end_time_idx'] = ["static","hr","23","Parking end i-th hour of monitoring period","Input(optional)"]
    return override_components,override_component_attrs


def optimize_ev(net):
    M=1e10

    m=net.optimize.create_model()

    m.constraints.remove("StorageUnit-energy_balance")
    m.constraints.remove("Generator-p_set")

    ev = pd.Index(set(net.storage_units.carrier.unique()), name='ev')
    carrier = net.storage_units.carrier.to_xarray()
    pv_idx = net.generators.loc[net.generators.index.str.contains('PV')].index

    m.add_variables(coords=[net.storage_units.index],name='initial_SoC')
    m.add_variables(coords=[net.snapshots,net.storage_units.index],name='charge_event',binary=True)
    m.add_variables(coords=[net.snapshots,net.storage_units.index],name='discharge_event',binary=True)
    m.add_variables(coords=[net.snapshots,net.storage_units.index],name='store_jump',binary=True)
    m.add_variables(coords=[net.snapshots,net.storage_units.index],name='dispatch_jump',binary=True)
    m.add_variables(coords=[net.snapshots,ev],name='delta_E')
    m.add_variables(coords=[net.snapshots,ev],name='carrier_SoC')
    m.add_variables(coords=[ev],name='initial_carrier_SoC')
    m.add_variables(coords=[net.storage_units.index],name='next_trip_slack')
    m.add_variables(coords=[net.snapshots,net.storage_units.index],name='flex_flag',binary=True)
    m.add_variables(coords=[net.snapshots,net.storage_units.index],lower=0,name='power_shift_abs')

    mask_first_sns = xr.DataArray(net.snapshots==net.snapshots[0],coords=[net.snapshots])

    # Identify charge/discharge from StorageUnit-p_store/StorageUnit-p_dispatch
    charging_1 = m.variables['StorageUnit-p_store']<=M*m.variables['charge_event']
    charging_2 = m.variables['StorageUnit-p_store']+M*m.variables['charge_event']>=1-M
    m.add_constraints(charging_1,name='identify_charging_1')
    m.add_constraints(charging_2,name='identify_charging_2')

    discharging_1 = m.variables['StorageUnit-p_dispatch']<=M*m.variables['discharge_event']
    discharging_2 = m.variables['StorageUnit-p_dispatch']+M*m.variables['discharge_event']>=1-M
    m.add_constraints(discharging_1,name='identify_discharging_1')
    m.add_constraints(discharging_2,name='identify_discharging_2')

    # Exclude simultaneous charging and discharging
    non_simultaneous = m.variables['charge_event']+m.variables['discharge_event']<=1
    m.add_constraints(non_simultaneous,name='non_simultaneous')

    # Park End SoC covers next trip consumption
    end_idx = net.storage_units.park_end_time_idx.to_xarray()
    selected_snapshots = [net.snapshots[min(int(idx)+1,23)] for idx in end_idx.values]
    m.add_constraints(m.variables['StorageUnit-state_of_charge'].sel(snapshot=xr.DataArray(selected_snapshots, coords=[net.storage_units.index]))+m.variables['next_trip_slack']>=net.storage_units.next_consumption, name="next_trip_consumption")


    # Limit V2G participation level at each snapshot
    m.add_constraints(m.variables['power_shift_abs']-m.variables['StorageUnit-p_store']+m.variables['StorageUnit-p_dispatch']>=net.storage_units_t.p_set,name='power_shift_abs_1')
    m.add_constraints(m.variables['power_shift_abs']+m.variables['StorageUnit-p_store']-m.variables['StorageUnit-p_dispatch']>=net.storage_units_t.p_set,name='power_shift_abs_2')
    m.add_constraints(m.variables['power_shift_abs']<=M*m.variables['flex_flag'],name='force_flex_flag')
    m.add_constraints(m.variables['flex_flag'].sum(dim='StorageUnit')<=(0.3*(net.storage_units_t.parking_hr>0).sum(axis=1)).to_xarray(),name='v2g_participation')

    # Detect charge/discharge start
    delta_p_store = m.variables['StorageUnit-p_store']-m.variables['StorageUnit-p_store'].shift(snapshot=1,fill_value=0)
    detect_store_jump_1 = delta_p_store<=M*m.variables['store_jump']
    m.add_constraints(detect_store_jump_1,name='detect_store_jump_1')
    detect_store_jump_2 = delta_p_store+M*m.variables['store_jump']>=1-M
    m.add_constraints(detect_store_jump_2,name='detect_store_jump_2')

    delta_p_dispatch = m.variables['StorageUnit-p_dispatch']-m.variables['StorageUnit-p_dispatch'].shift(snapshot=1,fill_value=0)
    detect_dispatch_jump_1 = delta_p_dispatch<=M*m.variables['dispatch_jump']
    m.add_constraints(detect_dispatch_jump_1,name='detect_dispatch_jump_1')
    detect_dispatch_jump_2 = delta_p_dispatch+M*m.variables['dispatch_jump']>=1-M
    m.add_constraints(detect_dispatch_jump_2,name='detect_dispatch_jump_2')

    # Limit the charge/discharge number for each parking event
    event_cnt = m.variables['dispatch_jump'].sum(dim='snapshot')+m.variables['store_jump'].sum(dim='snapshot')
    m.add_constraints(event_cnt<=1,name='limit_event_cnt')

    # agg EV charge/discharge power
    m.add_constraints(m.variables['delta_E']==((m.variables['StorageUnit-p_store']*net.storage_units_t.parking_hr).groupby(carrier).sum()-(m.variables['StorageUnit-p_dispatch']*net.storage_units_t.parking_hr).groupby(carrier).sum()).sel(carrier=m.variables['delta_E'].coords['ev']), name='agg_ev_delta_E')

    # agg EV SoC Update
    m.add_constraints(m.variables['initial_SoC']==net.storage_units.state_of_charge_initial, name='force_initial_SoC')
    m.add_constraints(m.variables['initial_carrier_SoC']==m.variables['initial_SoC'].groupby(carrier).sum().sel(carrier=m.variables['initial_carrier_SoC'].coords['ev']),name='force_initial_agg_SoC')
    m.add_constraints((m.variables['carrier_SoC']-m.variables['initial_carrier_SoC']-m.variables['delta_E']).where(mask_first_sns)==0,name='update_first_SoC')
    pre_SoC = m.variables['carrier_SoC'].shift(snapshot=1)
    m.add_constraints((pre_SoC+m.variables['delta_E']-m.variables['carrier_SoC']).where(~mask_first_sns)==0,name='update_following_SoC')

    # map carrier_SoC back to parking_events
    m.add_constraints(m.variables['StorageUnit-state_of_charge'] == m.variables['carrier_SoC'].sel(ev=carrier.sel(StorageUnit=m.variables['StorageUnit-state_of_charge'].coords['StorageUnit'])),name='map_to_carrier_SoC')

    # Objective to keep day end soc as the same level as possible
    # quad_soc_diff  = (m.variables['StorageUnit-state_of_charge']**2-2*m.variables['StorageUnit-state_of_charge']*net.storage_units.day_end_SoC/1000).where(xr.DataArray(net.snapshots==net.snapshots[23],coords=[net.snapshots])).sum(dim='StorageUnit').sel(snapshot=net.snapshots[23])
    m.objective = m.variables['next_trip_slack'].sum()-m.variables['Generator-p'].sel(Generator=pv_idx).sum().sum()

    solver_opt = {
        "Method":-1,          # Automatic
        "MIPFocus": 1,         # Focus on finding feasible solutions
        "Heuristics": 0.3,     # Set the time spent on heuristics to 10%
        "MIPGap": 0.3         # Set the acceptable optimality gap to 1%
    }
    result = net.optimize.solve_model(solver_name='gurobi', solver_options=solver_opt)

    # # Check the result status and print infeasibilities if necessary
    if result[0]!='ok':
        print(f'{folder}_Infeasible')
        net.lpf()
        net.pf(use_seed=True)

        # m.print_infeasibilities()

    return net,m,result[0]



def optimize_ev_pv(net):
    M=1e10
    pv_idx = net.generators.loc[net.generators.carrier=='solar'].index
    pv_to_ev_eta = 1 # charging with PV power efficiency
    ev = pd.Index(set(net.storage_units.carrier.unique()), name='ev')
    carrier = net.storage_units.carrier.to_xarray()
    ev_to_bus = net.storage_units.bus.to_xarray()
    pv_to_bus = net.generators.loc[pv_idx].bus.to_xarray()

    m=net.optimize.create_model()

    m.constraints.remove("StorageUnit-energy_balance")
    m.constraints.remove("Generator-p_set")

    mask_first_sns = xr.DataArray(net.snapshots==net.snapshots[0],coords=[net.snapshots])

    m.add_variables(coords=[net.snapshots,net.storage_units.index],name='flex_flag',binary=True)
    m.add_variables(coords=[net.snapshots,net.storage_units.index],lower=0,name='power_shift_abs')

    m.add_variables(coords=[net.snapshots,net.storage_units.index],name='grid_charge_event',binary=True)
    m.add_variables(coords=[net.snapshots,net.storage_units.index],name='PV_charge_event',binary=True)
    m.add_variables(coords=[net.snapshots,net.storage_units.index],name='discharge_event',binary=True)
    m.add_variables(coords=[net.snapshots,net.storage_units.index],name='store_jump',binary=True)
    m.add_variables(coords=[net.snapshots,net.storage_units.index],name='dispatch_jump',binary=True)

    m.add_variables(coords=[net.storage_units.index],name='initial_SoC')
    m.add_variables(coords=[net.snapshots,ev],name='delta_E')
    m.add_variables(coords=[net.snapshots,ev],name='carrier_SoC')
    m.add_variables(coords=[ev],name='initial_carrier_SoC')
    m.add_variables(coords=[net.storage_units.index],name='next_trip_slack')
    m.add_variables(coords=[net.snapshots,pv_idx],lower=0, name='PV_to_node') # direct charging to EV on the same node

    m.add_variables(coords=[net.snapshots,net.storage_units.index],lower=0, name='node_to_EV') # direct charging from PV on the same node

    # Generator-p upper limit considering pv to ev charing
    m.constraints.remove("Generator-fix-p-upper")
    m.add_constraints(m.variables['Generator-p'].sel(Generator=pv_idx)+m.variables['PV_to_node']<=net.generators_t.p_set,name='Generator-fix-p-upper') #+m.variables['PV_slack']

    # Exculde charging by PV when EV not connected
    mask_not_parking=xr.DataArray(net.storage_units_t.parking_hr==0,coords=[net.snapshots,net.storage_units.index])
    m.add_constraints(m.variables['node_to_EV'].where(mask_not_parking)==0,name='charging_by_PV_not_possible')

    # Park End SoC covers next trip consumption
    end_idx = net.storage_units.park_end_time_idx.to_xarray()
    end_snapshots = [net.snapshots[int(idx)] for idx in end_idx.values]
    m.add_constraints(m.variables['StorageUnit-state_of_charge'].sel(snapshot=xr.DataArray(end_snapshots, coords=[net.storage_units.index]))+m.variables['next_trip_slack']>=net.storage_units.next_consumption, name="next_trip_consumption")


    # Identify charge/discharge from StorageUnit-p_store/StorageUnit-p_dispatch
    charging_grid_1 = m.variables['StorageUnit-p_store']<=M*m.variables['grid_charge_event'] #+m.variables['node_to_EV']
    charging_grid_2 = m.variables['StorageUnit-p_store']+M*m.variables['grid_charge_event']>=1-M #+m.variables['node_to_EV']
    m.add_constraints(charging_grid_1,name='identify_grid_charging_1')
    m.add_constraints(charging_grid_2,name='identify_grid_charging_2')

    charging_pv_1 = m.variables['node_to_EV']<=M*m.variables['PV_charge_event'] #+m.variables['node_to_EV']
    charging_pv_2 = m.variables['node_to_EV']+M*m.variables['PV_charge_event']>=1-M #+m.variables['node_to_EV']
    m.add_constraints(charging_pv_1,name='identify_pv_charging_1')
    m.add_constraints(charging_pv_2,name='identify_pv_charging_2')

    discharging_1 = m.variables['StorageUnit-p_dispatch']<=M*m.variables['discharge_event']
    discharging_2 = m.variables['StorageUnit-p_dispatch']+M*m.variables['discharge_event']>=1-M
    m.add_constraints(discharging_1,name='identify_discharging_1')
    m.add_constraints(discharging_2,name='identify_discharging_2')

    # Exclude simultaneous charging from grid, PV and discharging
    non_simultaneous = m.variables['grid_charge_event']+m.variables['PV_charge_event']+m.variables['discharge_event']<=1
    m.add_constraints(non_simultaneous,name= 'non_simultaneous')

    # # Limit V2G participation level at each snapshot
    m.add_constraints(m.variables['power_shift_abs']-m.variables['StorageUnit-p_store']-m.variables['node_to_EV']+m.variables['StorageUnit-p_dispatch']>=net.storage_units_t.p_set,name='power_shift_abs_1')
    m.add_constraints(m.variables['power_shift_abs']+m.variables['StorageUnit-p_store']+m.variables['node_to_EV']-m.variables['StorageUnit-p_dispatch']>=net.storage_units_t.p_set,name='power_shift_abs_2')
    m.add_constraints(m.variables['power_shift_abs']<=M*m.variables['flex_flag'],name='force_flex_flag')
    m.add_constraints(m.variables['flex_flag'].sum(dim='StorageUnit')<=(0.3*(net.storage_units_t.parking_hr>0).sum(axis=1)).to_xarray(),name='v2g_participation')


    # Detect charge/discharge start
    delta_p_store = m.variables['StorageUnit-p_store']-m.variables['StorageUnit-p_store'].shift(snapshot=1,fill_value=0)
    detect_store_jump_1 = delta_p_store<=M*m.variables['store_jump']
    m.add_constraints(detect_store_jump_1,name='detect_store_jump_1')
    detect_store_jump_2 = delta_p_store+M*m.variables['store_jump']>=1-M
    m.add_constraints(detect_store_jump_2,name='detect_store_jump_2')

    delta_p_dispatch = m.variables['StorageUnit-p_dispatch']-m.variables['StorageUnit-p_dispatch'].shift(snapshot=1,fill_value=0)
    detect_dispatch_jump_1 = delta_p_dispatch<=M*m.variables['dispatch_jump']
    m.add_constraints(detect_dispatch_jump_1,name='detect_dispatch_jump_1')
    detect_dispatch_jump_2 = delta_p_dispatch+M*m.variables['dispatch_jump']>=1-M
    m.add_constraints(detect_dispatch_jump_2,name='detect_dispatch_jump_2')

    # Limit the charge/discharge number for each parking event
    event_cnt = m.variables['dispatch_jump'].sum(dim='snapshot')+m.variables['store_jump'].sum(dim='snapshot')
    m.add_constraints(event_cnt<=1,name='limit_event_cnt')

    # EV charging by PV production
    m.add_constraints(m.variables['PV_to_node'].groupby(pv_to_bus).sum()==m.variables['node_to_EV'].groupby(ev_to_bus).sum(),name="pv_to_ev_balance")

    # agg EV charge/discharge energy
    m.add_constraints(m.variables['delta_E']==(((m.variables['node_to_EV']+m.variables['StorageUnit-p_store']-m.variables['StorageUnit-p_dispatch'])*net.storage_units_t.parking_hr).groupby(carrier).sum()).sel(carrier=m.variables['delta_E'].coords['ev']),name="agg_ev_delta_E")

    # agg EV SoC Update
    m.add_constraints(m.variables['initial_SoC']==net.storage_units.state_of_charge_initial, name='force_initial_SoC')
    m.add_constraints(m.variables['initial_carrier_SoC']==m.variables['initial_SoC'].groupby(carrier).sum().sel(carrier=m.variables['initial_carrier_SoC'].coords['ev']),name='force_initial_agg_SoC')
    m.add_constraints((m.variables['carrier_SoC']-m.variables['initial_carrier_SoC']-m.variables['delta_E']).where(mask_first_sns)==0,name='update_first_SoC')
    pre_SoC = m.variables['carrier_SoC'].shift(snapshot=1)
    m.add_constraints((pre_SoC+m.variables['delta_E']-m.variables['carrier_SoC']).where(~mask_first_sns)==0,name='update_following_SoC')

    # map carrier_SoC back to parking_events
    m.add_constraints(m.variables['StorageUnit-state_of_charge'] == m.variables['carrier_SoC'].sel(ev=carrier.sel(StorageUnit=m.variables['StorageUnit-state_of_charge'].coords['StorageUnit'])),name='map_to_carrier_SoC')

    # quad_soc_diff  = (m.variables['StorageUnit-state_of_charge']**2-2*m.variables['StorageUnit-state_of_charge']*net.storage_units.day_end_SoC/1000).where(xr.DataArray(net.snapshots==net.snapshots[23],coords=[net.snapshots])).sum(dim='StorageUnit').sel(snapshot=net.snapshots[23])

    m.objective= m.variables['next_trip_slack'].sum().sum()+(m.variables['StorageUnit-p_store']-m.variables['node_to_EV']).sum().sum()#+quad_soc_diff
    solver_opt = {
        "Method":-1,            # Automatic
        "MIPFocus": 3,         # Focus on finding feasible solutions
        "Heuristics": 0.3,     # Set the time spent on heuristics to 10%
        "MIPGap": 0.3       # Set the acceptable optimality gap to 1%
        # "TimeLimit":timeout   # max solving time
    }
    result = net.optimize.solve_model(solver_name='gurobi', solver_options=solver_opt)


    # # Check the result status and print infeasibilities if necessary
    if result[0]!='ok':
        print(f'{folder}_Infeasible')
        net.lpf()
        net.pf(use_seed=True)
        # m.print_infeasibilities()

    return net,m,result[0]


if __name__=='__main__':
    override_components,override_component_attrs = override_storage_unit()
    if MV_feeder:
        save_paths = make_save_path(MV_feeder, path_controlled, path_uncontrolled, None, experiment)
        nets = build_net(MV_feeder=MV_feeder, folder=None, grid=grid, day_start_ts=day_start_ts,
                            monitor_hr=monitor_hr, scenario_year=scenario_year,override_components=override_components,override_component_attrs=override_component_attrs)

    else:
        result = "None"
        ev_res = pd.DataFrame()
        ev_res.index = pd.date_range(day_start_ts,periods=24,freq='h')
        pv_res = pd.DataFrame()
        pv_res.index = pd.date_range(day_start_ts,periods=24,freq='h')
        ev_inflow = pd.DataFrame()
        ev_inflow.index = pd.date_range(day_start_ts,periods=24,freq='h')
        for folder in LV_list:

            save_paths = make_save_path(MV_feeder, path_controlled, path_uncontrolled, folder, experiment)
            nets = build_net(MV_feeder=MV_feeder, folder=folder, grid=grid, day_start_ts=day_start_ts,
                                monitor_hr=monitor_hr, scenario_year=scenario_year,override_components=override_components,override_component_attrs=override_component_attrs)
            
            nets[1].export_to_netcdf(f"{save_paths[1]}/net_uncontrolled.nc")
            if len(nets[1].storage_units.index)>0 and len(nets[1].generators[nets[1].generators.carrier=='solar'])>0:
                net_optimized, optimized_model,result = optimize_ev_pv(nets[1].copy())
                net_optimized.export_to_netcdf(f"{save_paths[1]}/net_optimized.nc") # only ran the network.optimize()
            else: # No ev operation needs to be optimized
                nets[1].lpf()
                nets[1].pf(use_seed=True)
                net_optimized = nets[1]
                net_optimized.export_to_netcdf(f"{save_paths[1]}/net_optimized.nc") # only ran the network.optimize()


            # # Fix opf() result power for pf()
            # PV generation
            net_set = net_optimized.copy()
            pv_id = net_set.generators.loc[net_set.generators.carrier==('solar')].index
            net_set.generators_t.p_set[pv_id] = net_set.generators_t.p[pv_id]
            net_set.generators_t.q_set[pv_id] = np.tan(np.arccos(0.97)) * net_set.generators_t.p_set[pv_id]
            # EV profile
            net_set.storage_units_t.p_set = net_set.storage_units_t.p
            net_set.storage_units_t.q_set = np.tan(np.arccos(0.97)) * net_set.storage_units_t.p_set
            slack_generator = net_set.generators[net_optimized.generators['control'] == 'Slack']
            net_set.buses.loc[slack_generator['bus'], 'v_mag_pu_set'] = 1.03
            if result=='ok':
                net_set.storage_units_t.inflow=optimized_model.variables['node_to_EV'].solution.to_dataframe().reset_index().pivot(index='snapshot',columns='StorageUnit',values='solution') 
            net_set.lpf()
            net_set.pf(use_seed=True)
            net_set.export_to_netcdf(f"{save_paths[1]}/net_set.nc") # ran network.pf() using opf active power result

            line_load = psastat.pf_line_loading(net_set)
            line_stat = psastat.pf_line_overloading_stat(net_set)
            trafo_load = psastat.pf_trafo_loading(net_set)
            trafo_stat = psastat.pf_trafo_overloading_stat(net_set)
            non_slack = net_set.buses.loc[net_set.buses.control != 'Slack'].index
            bus_vmag = net_set.buses_t.v_mag_pu[non_slack]
            bus_stat_under = psastat.pf_undervoltage(net_set)
            bus_stat_over = psastat.pf_overvoltage(net_set)

            ev_res = pd.concat([ev_res,net_set.storage_units_t.p_set],axis=1)
            pv_res = pd.concat([pv_res,net_set.generators_t.p_set[pv_id]],axis=1)
            ev_inflow = pd.concat([ev_inflow,net_set.storage_units_t.inflow],axis=1)


            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(line_load)
            ax.set_title("Line Loading Percentage")
            # ax.set_ylim([-5,line_controlled.max()])
            ax.set_xlabel("Hour")
            ax.set_ylabel("Loading Percentage [%]")
            ax.tick_params(axis='x', rotation=90)
            plt.tight_layout()
            plt.savefig(f"{save_paths[1]}/optimized_line_loading.png")
            plt.close()

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.boxplot(line_load.T)
            ax.set_title("Line Loading Dsitribution")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Loading Distribution")
            ax.tick_params(axis='x', rotation=90)
            plt.tight_layout()
            plt.savefig(f"{save_paths[1]}/optimized_line_load_distribution.png")
            plt.close()

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(line_stat)
            ax.set_title("Hourly Line Overloading")
            # ax.set_ylim([-5,line_controlled.max()])
            ax.set_xlabel("Hour")
            ax.set_ylabel("Loading Percentage [%]")
            ax.tick_params(axis='x', rotation=90)
            plt.tight_layout()
            plt.savefig(f"{save_paths[1]}/optimized_line_overloading.png")
            plt.close()

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(trafo_load)
            ax.set_title("Trafo Loading Percentage")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Loading Percentage [%]")
            ax.tick_params(axis='x', rotation=90)
            plt.tight_layout()
            plt.savefig(f"{save_paths[1]}/optimized_trafo_loading.png")
            plt.close()

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.boxplot(trafo_load.T)
            ax.set_title("Trafo Loading Distribution")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Load Distribution")
            ax.tick_params(axis='x', rotation=90)
            plt.tight_layout()
            plt.savefig(f"{save_paths[1]}/optimized_trafo_load_distribution.png")
            plt.close()

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(trafo_stat)
            ax.set_title("Hourly Trafo Overloading")
            # ax.set_ylim([-5,line_controlled.max()])
            ax.set_xlabel("Hour")
            ax.set_ylabel("Loading Percentage [%]")
            ax.tick_params(axis='x', rotation=90)
            plt.tight_layout()
            plt.savefig(f"{save_paths[1]}/optimized_trafo_overloading.png")
            plt.close()

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(bus_vmag)
            ax.set_title("Bus vmag")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Voltage Magnitude")
            ax.tick_params(axis='x', rotation=90)
            plt.tight_layout()
            plt.savefig(f"{save_paths[1]}/optimized_bus_vmag.png")
            plt.close()

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.boxplot(bus_vmag.T)
            ax.set_title("Bus Vmag Distribution")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Voltage Magnitude Distribution")
            ax.tick_params(axis='x', rotation=90)
            plt.tight_layout()
            plt.savefig(f"{save_paths[1]}/optimized_bus_vmag_distribution.png")
            plt.close()

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(bus_stat_under)
            ax.set_title("Bus undervoltage")
            # ax.set_ylim([-5,line_controlled.max()])
            ax.set_xlabel("Hour")
            ax.set_ylabel("Loading Percentage [%]")
            ax.tick_params(axis='x', rotation=90)
            plt.tight_layout()
            plt.savefig(f"{save_paths[1]}/optimized_bus_under.png")
            plt.close()

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(bus_stat_over)
            ax.set_title("Bus overvoltage")
            # ax.set_ylim([-5,line_controlled.max()])
            ax.set_xlabel("Hour")
            ax.set_ylabel("Loading Percentage [%]")
            ax.tick_params(axis='x', rotation=90)
            plt.tight_layout()
            plt.savefig(f"{save_paths[1]}/optimized_bus_over.png")
            plt.close()

            # Check SoC Result
            end_SoE = net_set.storage_units_t.state_of_charge.iloc[23].groupby(net_set.storage_units.carrier).first()
            last_unique = net_set.storage_units.groupby('carrier')[['max_hours', 'p_nom', 'day_end_SoC']].first()
            cap_unique = last_unique.max_hours * last_unique.p_nom
            end_SoC = end_SoE / cap_unique * 100
            unshifted_end_SoC = last_unique.day_end_SoC / (cap_unique) * 100

            counts, bins = np.histogram(end_SoC, bins=range(0, 110, 10))
            counts_unshifted, bins_unshifted = np.histogram(unshifted_end_SoC, bins=range(0, 110, 10))

            plt.subplots(figsize=(10,6))
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            plt.xticks(range(0, 110, 10))
            plt.plot(bin_centers, counts / counts.sum(), linestyle='-', marker='o', label='shifted day end SoC')
            plt.plot(bin_centers, counts_unshifted / counts_unshifted.sum(), linestyle='--', marker='x',
                    label='unshifted day end SoC')
            plt.xlabel('SoC [%]')
            plt.ylabel('Share [-]')
            plt.title('Day End SoC')
            plt.legend()
            plt.savefig(f"{save_paths[1]}/day_end_soc_distribution.png")
            plt.close()


            plt.figure(figsize=(10, 10))
            plt.scatter(unshifted_end_SoC.values, end_SoC.values, alpha=0.7)
            plt.xlabel('Unshifted Day End SoC')
            plt.ylabel('Shifted Day End SoC')
            plt.title('Scatter Plot of Unshifted End SoC vs End SoC')
            plt.tight_layout()
            plt.savefig(f"{save_paths[1]}/end_soc_shift.png")
            plt.close()


        ev_res.to_csv(f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/mobility/{path_uncontrolled}/ev_pv_optimize/ev_opt_res.csv")
        pv_res.to_csv(f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/mobility/{path_uncontrolled}/ev_pv_optimize/pv_opt_res.csv")
        ev_inflow.to_csv(f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/mobility/{path_uncontrolled}/ev_pv_optimize/pv_opt_res.csv")

