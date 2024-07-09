import matlab.engine
import pp_create_basic_network as basic
import pandapower as pp
import geopandas as gpd
import math
import json
import pickle
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import matlab
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(42)


def ev_max_p_mw(sns,park_t,row,pre_ev_res):
    # parked or not
    parked = (park_t>0)
    if (not parked) or (row.soc_percent>95): 
        return 0
    elif (row.soc_percent-row.next_SoC_change>100) and (sns>=row.arr_time_idx) and (sns<=row.park_end_time_idx):
        avg_discharge_rate = (row.next_SoC_change-row.soc_percent+100)/100*row.max_e_mwh/(min(row.park_end_time_idx,23)-sns+1)
        return min(max(avg_discharge_rate,-row.chg_rate),0)
    else: #max energy can be charged until 100% SoC [MWh]
        # if (pre_ev_res<0).any():
        #     return 0
        if sns==row.arr_time_idx: # Consider SoC jump driving in and out the grid if it's the snapshot when the ev arrives again in the grid
            SoE_left = row.max_e_mwh*(1-(row.soc_percent+row.SoC_change)/100)
        else:
            SoE_left = row.max_e_mwh*(1-row.soc_percent/100)            
        max_p_full = SoE_left/park_t # power needed to achieve 100 SoC within this hour
        return min(max(max_p_full,0), row.chg_rate)
def ev_min_p_mw(sns,park_t,row,pre_ev_res):
    parked = (park_t>0)
    SoC_lower_bound = max(row.next_trip_e/row.max_e_mwh*100,5)
    if (not parked): 
        return 0
    elif (row.soc_percent<SoC_lower_bound): # Current SoC not fulfilling 5% SoC or next trip energy requirement
        avg_charge_rate=((SoC_lower_bound-row.soc_percent)/100*row.max_e_mwh)/(min(row.park_end_time_idx,23)-sns+1) #avg charge power needed in mw assuming charging in every hour later on
        return max(min(avg_charge_rate,row.chg_rate),0)
    elif (row.soc_percent-row.next_SoC_change<0)and (sns>=row.arr_time_idx) and (sns<=row.park_end_time_idx):#current SoC status can not stand next SoE_change coming back to the grid
        avg_charge_rate=(row.next_SoC_change-row.soc_percent)/100*row.max_e_mwh/(min(row.park_end_time_idx,23)-sns+1)
        return max(min(avg_charge_rate,row.chg_rate),0)
    else:
        # if (pre_ev_res>0).any():
        #     return 0
        # max energy can be discharged unitl 5% SoC [MWh]
        if (sns==row['arr_time_idx']): # Consider SoC jump driving in and out the grid if it's the snapshot when the ev arrives again in the grid
            SoE = row['max_e_mwh']*(row['soc_percent']-row['SoC_change']-SoC_lower_bound)/100 
        else:
            SoE = row['max_e_mwh']*(row['soc_percent']-SoC_lower_bound)/100 
        min_p_empty =-SoE/park_t # power needed to completely discharg the battery within this hour
        return max(min(0,min_p_empty), -row['chg_rate'])

def draw_controllable_evs(net_storage,ctrl_frac,current_rand_state):
    net_storage['controllable'] = True
    # Prioritize setting EVs with soc_percent > 95 or soc_percent < 5 as controllable
    critical_soc_condition = (net_storage['soc_percent'] >95) | (net_storage['soc_percent'] < 5)| (net_storage['soc_percent']-net_storage['next_SoC_change']<0) | (net_storage['soc_percent']-net_storage['next_SoC_change']>100)
    critical_evs = net_storage[critical_soc_condition]
    non_critical_evs = net_storage[~critical_soc_condition]
    # Calculate the number of EVs to be controllable
    total_evs = len(net_storage)
    num_controllable = int(ctrl_frac * total_evs)
    num_uncontrollable = total_evs - num_controllable
    # Ensure critical EVs are included in the controllable EVs
    critical_controllable = critical_evs if len(critical_evs) <= num_controllable else critical_evs.sample(n=num_controllable, random_state=current_rand_state)
    remaining_controllable_needed = num_controllable - len(critical_controllable)
    
    if remaining_controllable_needed > 0:
        additional_controllable = non_critical_evs.sample(n=remaining_controllable_needed, random_state=current_rand_state)
    else:
        additional_controllable = pd.DataFrame()

    controllable_evs = pd.concat([critical_controllable, additional_controllable])
    uncontrollable_evs = net.storage.drop(controllable_evs.index)

    net_storage.loc[uncontrollable_evs.index, 'controllable'] = False
    return net_storage


def transfer_matpower_to_pandapower(net, gen_results,bus_results,slack_bus_id,slack_vm_pu,ac):
    # Update controllable sgen and storage p_mw, q_mvar values in network dataframes before runpp
    solar_ev = gen_results.iloc[1:].reset_index(drop=True)
    solar_gen_cnt = len(net.sgen.type=='PV')
    solar_mat_power_res = solar_ev.loc[0:solar_gen_cnt-1]
    net.sgen.loc[net.sgen.type=='PV','p_mw'] = solar_mat_power_res[2-1].values
    if ac:
        net.sgen.loc[net.sgen.type=='PV','q_mvar'] = solar_mat_power_res[3-1].values
    else:
        net.sgen.loc[net.sgen.type=='PV','q_mvar'] = net.sgen.loc[net.sgen.type=='PV','p_mw']*np.tan(np.arccos(0.97))
    if len(net.storage)>0:
        ev_controllable = solar_ev.loc[solar_gen_cnt:].reset_index(drop=True)
        net.storage.loc[(net.storage.type=='ev')&(net.storage.controllable==True),'p_mw'] = ev_controllable[2-1].values*(-1)
        if ac:
            net.storage.loc[(net.storage.type=='ev')&(net.storage.controllable==True),'q_mvar'] = ev_controllable[3-1].values*(-1)
        else:
            net.storage.loc[(net.storage.type=='ev')&(net.storage.controllable==True),'q_mvar'] = 0 #ev_controllable[3-1].values*(-1)
    
    # # Set slack bus vm pu
    # net.ext_grid.vm_pu = slack_vm_pu
    # net.bus.loc[slack_bus_id].vn_kv=net.bus.loc[slack_bus_id].vn_kv*slack_vm_pu
    # if not net.sgen.loc[net.sgen.bus==slack_bus_id].empty:
    #     net.sgen.loc[net.sgen.bus==slack_bus_id,'vm_pu']=slack_vm_pu
    return net




# LV_list=   ['214-1_1_5', '216-2_1_3', '214-1_2_2', '216-2_0_2', '216-1_1_3',
#            '216-1_2_4', '216-3_0_5', '39-1_0_4', '225-2_0_3', '225-4_0_3',
#            '225-1_0_6', '225-2_1_5', '225-3_0_6', '298-1_1_4', '298-4_0_2',
#            '225-5_0_5', '230-106_0_2', '298-2_0_5', '298-7_1_3', '298-7_0_3',
#            '298-5_0_5', '298-6_0_2','298-3_0_3', '298-8_0_7', '227-12_0_5',
#            '227-7_0_5', '227-10_1_3', '227-11_0_4', '230-150_0_2',
#            '227-9_0_5', '230-202_0_2', '230-180_0_4', '230-197_0_5',
#            '230-200_0_4', '230-202_1_3', '230-201_0_6', '230-211_0_2',
#            '230-212_0_3', '230-212_1_3', '230-108_0_6', '227-13_0_3',
#            '227-14_0_4', '227-8_0_10', '227-13_1_3', '227-1_0_5', '227-6_1_4',
#            '227-3_0_5', '298-9_0_5', '298-6_1_4', '298-4_1_5']

LV_list = ['214-1_1_5', '216-2_1_3', '214-1_2_2', '216-2_0_2', '216-1_1_3',
           '216-1_2_4', '216-3_0_5', '39-1_0_4', '225-2_0_3', '225-4_0_3',
           '225-1_0_6']


MV_feeder = False
grid = "369_0"
scenario_year = 2050
weekday = "Friday"
day_start_ts = pd.to_datetime(f"{scenario_year}-01-07 00:00:00")
day_start = day_start_ts.day
day_end_ts = pd.to_datetime(f"{scenario_year}-01-08 00:00:00")
monitor_hr =int((day_end_ts - day_start_ts).total_seconds() / 3600)
path_controlled = f"{grid}/{scenario_year}_{weekday}_01_07_controlled"
path_uncontrolled = f"{grid}/{scenario_year}_{weekday}_01_07_uncontrolled"
m = matlab.engine.start_matlab()

if MV_feeder:
    os.makedirs(f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/modified_swiss_pdg/369_0/{day_start_ts}/{grid}_pp_matpower",exist_ok=True)
    mv_folder_path=f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/modified_swiss_pdg/369_0/{day_start_ts}/{grid}_pp_matpower"
    net=basic.build_MV_LV_net(grid)
    base_load_profile = basic.prepare_base_profile(net,day_start_ts) # prepare base_load profile
    hp_load_profile= basic.prepare_hp_profile(net,day_start_ts) # add heatpump load to grid
    pv_load_profile=basic.prepare_pv_profile(net,day_start_ts) # add PV to the grid
    ev_load_profile = basic.prepare_ev_profile(net,path_uncontrolled,grid,day_start_ts,monitor_hr,scenario_year)
    pp.to_excel(net,f"{mv_folder_path}/{grid}.xlsx")
    base_load_profile.to_csv(f"{mv_folder_path}/{grid}_base_load_profile.csv",index=True,header=True)
    hp_load_profile.to_csv(f"{mv_folder_path}/{grid}_hp_load_profile.csv",index=True,header=True)
    pv_load_profile.to_csv(f"{mv_folder_path}/{grid}_pv_load_profile.csv",index=True,header=True)
    ev_load_profile.to_csv(f"{mv_folder_path}/{grid}_ev_load_profile.csv",index=True,header=True)
    if len(ev_load_profile)>0:
        first_record = ev_load_profile[ev_load_profile['parking_cnt']==0]
        first_record['init_SoC'] = first_record['augmented_SoE_bc']/first_record['B']*100
        SoC_dict = dict(zip(first_record['person'],first_record['init_SoC']))
        SoC_t = pd.DataFrame(index=SoC_dict.keys())
        SoC_t[-1] = pd.Series(SoC_dict)
    
    base_p = pd.DataFrame(base_load_profile.p_mw).T.apply(pd.Series.explode).reset_index(drop=True)
    base_q = pd.DataFrame(base_load_profile.q_mvar).T.apply(pd.Series.explode).reset_index(drop=True)
    hp_p = pd.DataFrame(hp_load_profile.p).T.apply(pd.Series.explode).reset_index(drop=True)
    hp_q = pd.DataFrame(hp_load_profile.q).T.apply(pd.Series.explode).reset_index(drop=True)
    pv_p = pd.DataFrame(pv_load_profile.pv_P_daily).T.apply(pd.Series.explode).reset_index(drop=True)
    pv_q = pd.DataFrame(pv_load_profile.pv_Q_daily).T.apply(pd.Series.explode).reset_index(drop=True)
    ev_p = pd.DataFrame(ev_load_profile.optimized_power_list).T.apply(pd.Series.explode).reset_index(drop=True)
    # parking duration of each hour in hours
    ev_park_t = pd.DataFrame(ev_load_profile.hourly_time_dict).T.apply(pd.Series.explode).reset_index(drop=True)/60
else:
    for folder in LV_list:
        os.makedirs(f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/modified_swiss_pdg/369_0/{day_start_ts}/{folder}_pp_matpower_maxPVinject",exist_ok=True)
        lv_folder_path=f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/modified_swiss_pdg/369_0/{day_start_ts}/{folder}_pp_matpower_maxPVinject"
        test = basic.build_LV_net(folder)
        base_load_profile = basic.prepare_base_profile(test,day_start_ts) # prepare base_load profile
        hp_load_profile= basic.prepare_hp_profile(test,day_start_ts) # add heatpump load to grid
        pv_load_profile=basic.prepare_pv_profile(test,day_start_ts) # add PV to the grid
        ev_load_profile = basic.prepare_ev_profile(test,path_uncontrolled,grid,day_start_ts,monitor_hr,scenario_year)

        pp.to_excel(test,f"{lv_folder_path}/{folder}.xlsx")
        base_load_profile.to_csv(f"{lv_folder_path}/{folder}_base_load_profile.csv",index=True,header=True)
        hp_load_profile.to_csv(f"{lv_folder_path}/{folder}_hp_load_profile.csv",index=True,header=True)
        pv_load_profile.to_csv(f"{lv_folder_path}/{folder}_pv_load_profile.csv",index=True,header=True)
        ev_load_profile.to_csv(f"{lv_folder_path}/{folder}_ev_load_profile.csv",index=True,header=True)


        net=test
        if len(ev_load_profile)>0:
            first_record = ev_load_profile[ev_load_profile['parking_cnt']==0]
            first_record['init_SoC'] = first_record['augmented_SoE_bc']/first_record['B']*100
            SoC_dict = dict(zip(first_record['person'],first_record['init_SoC']))
            SoC_t = pd.DataFrame(index=SoC_dict.keys())
            SoC_t[-1] = pd.Series(SoC_dict)

        base_p = pd.DataFrame(base_load_profile.p_mw).T.apply(pd.Series.explode).reset_index(drop=True)
        base_q = pd.DataFrame(base_load_profile.q_mvar).T.apply(pd.Series.explode).reset_index(drop=True)
        hp_p = pd.DataFrame(hp_load_profile.p).T.apply(pd.Series.explode).reset_index(drop=True)
        hp_q = pd.DataFrame(hp_load_profile.q).T.apply(pd.Series.explode).reset_index(drop=True)
        pv_p = pd.DataFrame(pv_load_profile.pv_P_daily).T.apply(pd.Series.explode).reset_index(drop=True)
        pv_q = pd.DataFrame(pv_load_profile.pv_Q_daily).T.apply(pd.Series.explode).reset_index(drop=True)
        ev_p = pd.DataFrame(ev_load_profile.optimized_power_list).T.apply(pd.Series.explode).reset_index(drop=True)
        # parking duration of each hour in hours
        ev_park_t = pd.DataFrame(ev_load_profile.hourly_time_dict).T.apply(pd.Series.explode).reset_index(drop=True)/60

        # Place holder for pandaopwer net with matpower results
        line_res = pd.DataFrame()
        bus_res = pd.DataFrame()
        ev_res = pd.DataFrame(index=net.storage.index)
        matpower_bus_res = pd.DataFrame()
        matpower_converged = {}

        for t in range(24):
            max_attempt=200
            rand_state=1
            attempt=0
            dc_max_attempt=100
            dc_attempt=0
            # Restore 100% max line loading percent
            net.line['max_loading_percent']=100

            net.load.loc[net.load.category == 'base', 'p_mw'] = base_p.loc[t].values
            net.load.loc[net.load.category=='base','q_mvar'] = base_q.loc[t].values

            # #PV
            net.sgen.loc[net.sgen.type=='PV','p_mw'] = pv_p.loc[t].values
            net.sgen.loc[net.sgen.type=='PV','q_mvar'] = pv_q.loc[t].values
            net.sgen.loc[net.sgen.type=='PV','max_p_mw'] = pv_p.loc[t].values
            # PV reactive power +- 0.9pf
            net.sgen.loc[net.sgen.type=='PV','max_q_mvar'] = np.tan(np.arccos(0.85))*pv_p.loc[t].values
            net.sgen.loc[net.sgen.type=='PV','min_q_mvar'] = -np.tan(np.arccos(0.85))*pv_p.loc[t].values

            # #Heatpump
            net.load.loc[net.load.category == 'hp', 'p_mw'] = hp_p.loc[t].values
            net.load.loc[net.load.category == 'hp', 'q_mvar'] = hp_q.loc[t].values

            # #EV
            if len(net.storage)>0:
                net.storage.loc[net.storage.type=='ev','p_mw'] = ev_p.loc[t].values

                # Sync currnet Battery SoC
                net.storage.loc[net.storage.type=='ev','soc_percent'] = net.storage['person'].map(SoC_dict)

                # Draw controllable evs according to SoC status
                net.storage=draw_controllable_evs(net.storage,0.28,rand_state)
                # Force controllable on unrealistic SoC
                net.storage.loc[(net.storage.soc_percent>100)|(net.storage.soc_percent<0),'controllable']=True
                # Recalculate ev's max/min_p_mw
                net.storage.loc[net.storage.type=='ev','max_p_mw'] = net.storage.apply(lambda row:ev_max_p_mw(t,ev_park_t.loc[t][row.name],row,ev_res.loc[row.name]),axis=1)
                net.storage.loc[net.storage.type=='ev','min_p_mw'] = net.storage.apply(lambda row:ev_min_p_mw(t,ev_park_t.loc[t][row.name],row,ev_res.loc[row.name]),axis=1)

            ########################################################
            # Start matpower ACOPF
            ########################################################  
            case_file=f'{lv_folder_path}/{folder}_hour_{t}.mat'
            results=None
            success=0
            pp.to_excel(net,f"{lv_folder_path}/{folder}_hour_{t}_preopf.xlsx")
            while attempt < max_attempt:
                # Convert network to MATPOWER case file
                mpc=pp.converter.to_mpc(net, filename=f"{lv_folder_path}/{folder}_hour_{t}.mat", init='flat')
                try:
                    # Run OPF and get results
                    results =  m.convert_opf_results(case_file,MV_feeder,False,True)
                    success = bool(results['success'])
                except Exception as e:
                    print(f"Error running OPF: {e}")
                    success = False
                if success:
                    break
                else:
                    rand_state += 1
                    attempt += 1
                    if len(net['storage']) > 0:
                        net['storage'] = draw_controllable_evs(net['storage'], 0.28, rand_state)
                        net['storage'].loc[(net['storage']['soc_percent'] > 100) | (net['storage']['soc_percent'] < 0), 'controllable'] = True

                    if attempt >= 10:
                        net['storage']['controllable'] = True
                        sampled_rows = net['storage'].sample(frac=0.7, random_state=rand_state)
                        net['storage'].loc[sampled_rows.index, 'controllable'] = False  

            if success:
                bus_results = pd.DataFrame(results['bus'])
                matpower_bus_res[t] = bus_results[7]
                gen_results = pd.DataFrame(results['gen'])
                branch_results = pd.DataFrame(results['branch'])
                slack_bus_id = results['slack_bus_id']-1
                slack_vm_pu = results['slack_vm_pu']
                transfer_matpower_to_pandapower(net,gen_results,bus_results,slack_bus_id,slack_vm_pu,True)
                matpower_converged[t] = True
                # # Reset vm_pu for slack bus
                # net.ext_grid.vm_pu = 1
                # net.bus.loc[slack_bus_id].vn_kv=net.bus.loc[slack_bus_id].vn_kv/slack_vm_pu
                # if not net.sgen.loc[net.sgen.bus==slack_bus_id].empty:
                #     net.sgen.loc[net.sgen.bus==slack_bus_id,'vm_pu']=1

            else:
                matpower_converged[t] = False
                while dc_attempt<dc_max_attempt:
                    mpc=pp.converter.to_mpc(net, filename=f"{lv_folder_path}/{folder}_hour_{t}.mat", init='flat')
                    try:
                        # Run DCOPF and get results
                        results =  m.convert_opf_results(case_file,MV_feeder,False,False)
                        dc_success = bool(results['success'])
                    except Exception as e:
                        print(f"Error running OPF: {e}")
                        dc_success = False
                    if dc_success:
                        break
                    else:
                        rand_state += 1
                        dc_attempt += 1
                        if len(net['storage']) > 0:
                            net['storage']['controllable'] = True
                            sampled_rows = net['storage'].sample(frac=0.7, random_state=rand_state)
                            net['storage'].loc[sampled_rows.index, 'controllable'] = False  
                if dc_success:
                    bus_results = pd.DataFrame(results['bus'])
                    matpower_bus_res[t] = bus_results[7]
                    gen_results = pd.DataFrame(results['gen'])
                    branch_results = pd.DataFrame(results['branch'])
                    slack_bus_id = results['slack_bus_id']-1
                    slack_vm_pu = results['slack_vm_pu']
                    transfer_matpower_to_pandapower(net,gen_results,bus_results,slack_bus_id,slack_vm_pu,False)

            # Run AC Power Flow calculuation with Matpower runopf results
            pp.runpp(net)
            pp.to_excel(net,f"{lv_folder_path}/{folder}_hour_{t}.xlsx")
            line_res[t] = net.res_line.loading_percent
            bus_res[t] = net.res_bus.vm_pu
            ev_res[t] = net.res_storage.p_mw
            if len(net.storage)>0:
                # #Update SoC
                net.res_storage['SoC_change'] = net.storage.apply(lambda row:row.SoC_change if row.arr_time_idx==t else 0, axis=1)
                net.res_storage['delta_energy'] = net.res_storage['p_mw']*ev_park_t.loc[t] - net.res_storage['SoC_change']/100*net.storage.max_e_mwh
                SoC_change=(net.res_storage.groupby(net.storage.person)['delta_energy'].sum())/(net.storage.groupby('person')['max_e_mwh'].first())*100
                for key, value in SoC_change.items():
                    SoC_dict[key]+=value
                    # Correct soc range for SoC_change in/out the grids
                    if SoC_dict[key]>100:
                        SoC_dict[key]=100
                    if SoC_dict[key]<0:
                        SoC_dict[key]=0
                SoC_t[t] = pd.Series(SoC_dict)
        if len(ev_load_profile)>0:
            SoC_t.to_csv(f"{lv_folder_path}/{folder}_SoC_t.csv")


        # Plot voltage magnitude results
        plt.figure(figsize=(10,6))
        plt.boxplot(bus_res)
        plt.xlabel('Hour')
        plt.ylabel('Voltage Magnitude [pu]')
        plt.title(f"LV_{folder}_acopf_vmpu_results")
        plt.savefig(f"{lv_folder_path}/{folder}_vmpu_distribution.png")
        plt.clf()
        # Plot line loading results
        plt.figure(figsize=(10,6))
        plt.boxplot(line_res)
        plt.xlabel('Hour')
        plt.ylabel('Line Loading [%]')
        plt.title(f"LV_{folder}_acopf_line_loading_results")
        plt.savefig(f"{lv_folder_path}/{folder}_line_loading_distribution.png")
        plt.clf()
        if len(net.storage)>0:
            # Plot SoC compare
            ev_load_profile['day_end_soc'] = ev_load_profile.day_end_soe/ev_load_profile.B*100
            unshifted_soc_end = ev_load_profile.groupby('person')['day_end_soc'].first()
            shifted_soc_end = SoC_t[23]
            plt.figure(figsize=(6, 6))
            plt.scatter(unshifted_soc_end.values, shifted_soc_end.values, marker='o', color='b',s=5)
            plt.xlim((0,100))
            plt.ylim((0,100))
            plt.xlabel('unshifted day end soc')
            plt.ylabel('day end soc after bottom up opf')
            plt.title(f"Day end soc compare: {folder}")
            plt.savefig(f"{lv_folder_path}/day_end_soc.png")
            plt.clf()
            # Check hourly convergency
            pd.DataFrame(list(matpower_converged.items()), columns=['Key', 'Value']).to_csv(f'{lv_folder_path}/hourly_convergency.csv', index=False)

