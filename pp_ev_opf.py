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
import pp_create_basic_network as basic


from pandapower import control
from pandapower.control import ConstControl
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


np.random.seed(42)


def ev_max_p_mw(t,row):
    # parked or not
    parked = (t>0)
    if (not parked) or (row['soc_percent']>95) or ((row['soc_percent']-row['next_SoC_change']>100) and (t==row['arr_time_idx'])):
        return 0
    else: #max energy can be charged until 100% SoC [MWh]
        if t==row['arr_time_idx']: # Consider SoC jump driving in and out the grid if it's the snapshot when the ev arrives again in the grid
            SoE_left = row['max_e_mwh']*(1-(row['soc_percent']+row[ 'SoC_change'])/100)
        else:
            SoE_left = row['max_e_mwh']*(1-row['soc_percent']/100)            
        max_p_full = SoE_left/t # power needed to achieve 100 SoC within this hour
        return min(max(max_p_full,0), row['chg_rate'])
def ev_min_p_mw(t,row):
    # parked or not
    parked = (t>0)
    if (not parked) or (row['soc_percent']<5) or ((row['soc_percent']-row['next_SoC_change']<0)and (t==row['arr_time_idx'])):
        return 0
    else:
        # max energy can be discharged unitl 5% SoC [MWh]
        if (t==row['arr_time_idx']): # Consider SoC jump driving in and out the grid if it's the snapshot when the ev arrives again in the grid
            SoE = row['max_e_mwh']*(row['soc_percent']-row['SoC_change']-5)/100 
        else:
            SoE = row['max_e_mwh']*(row['soc_percent']-5)/100 
        min_p_empty =-SoE/t # power needed to achieve 100 SoC within this hour
        return max(min(0,min_p_empty), -row['chg_rate'])
    

LV_list=   ['214-1_1_5', '216-2_1_3', '214-1_2_2', '216-2_0_2', '216-1_1_3',
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


MV_feeder = True
grid = "369_0"
scenario_year = 2050
weekday = "Friday"
day_start_ts = pd.to_datetime(f"{scenario_year}-01-07 00:00:00")
day_start = day_start_ts.day
day_end_ts = pd.to_datetime(f"{scenario_year}-01-08 00:00:00")
monitor_hr =int((day_end_ts - day_start_ts).total_seconds() / 3600)
path_controlled = f"{grid}/{scenario_year}_{weekday}_01_07_controlled"
path_uncontrolled = f"{grid}/{scenario_year}_{weekday}_01_07_uncontrolled"



if MV_feeder:
        os.makedirs(f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/modified_swiss_pdg/369_0/2050_01_07/{grid}_tap0",exist_ok=True)
        mv_folder_path=f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/modified_swiss_pdg/369_0/2050_01_07/{grid}_tap0"
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


        for t in range(24):
            max_attempt=100
            rand_state=1
            attempt=0
            # Reduce Line Loading Limit
            net.line['max_loading_percent']=90

            net.load.loc[net.load.category == 'base', 'p_mw'] = base_p.loc[t].values
            net.load.loc[net.load.category=='base','q_mvar'] = base_q.loc[t].values

            # #PV
            net.sgen.loc[net.sgen.type=='PV','p_mw'] = pv_p.loc[t].values
            net.sgen.loc[net.sgen.type=='PV','q_mvar'] = pv_q.loc[t].values
            net.sgen.loc[net.sgen.type=='PV','max_p_mw'] = pv_p.loc[t].values
            # No limit on PV reactive power yet, later apply pf = 0.97
            net.sgen.loc[net.sgen.type=='PV','max_q_mvar'] = pv_p.loc[t].values
            net.sgen.loc[net.sgen.type=='PV','min_q_mvar'] = pv_p.loc[t].values*(-1)

            # #Heatpump
            net.load.loc[net.load.category == 'hp', 'p_mw'] = hp_p.loc[t].values
            net.load.loc[net.load.category == 'hp', 'q_mvar'] = hp_q.loc[t].values
            
            # #EV
            if len(net.storage)>0:
                net.storage['controllable']=True

                net.storage.loc[net.storage.type=='ev','p_mw'] = ev_p.loc[t].values
                # Sync currnet Battery SoC
                net.storage.loc[net.storage.type=='ev','soc_percent'] = net.storage['person'].map(SoC_dict)

                # Randomly sample 70% of the evs
                sampled_rows = net.storage.sample(frac=0.7, random_state=rand_state)
                # Change the 'controllable' column value to False for the sampled rows
                net.storage.loc[sampled_rows.index, 'controllable'] = False

                # Recalculate ev's max/min_p_mw
                net.storage.loc[net.storage.type=='ev','max_p_mw'] = net.storage.apply(lambda row:ev_max_p_mw(ev_park_t.loc[t][row.name],row),axis=1)
                net.storage.loc[net.storage.type=='ev','min_p_mw'] = net.storage.apply(lambda row:ev_min_p_mw(ev_park_t.loc[t][row.name],row),axis=1)

            ########################################################
            ########################################################
            while attempt<max_attempt:
                try:
                    # Run DCOPF with 90% line loading limit
                    pp.rundcopp(net)
                    break
                except Exception as e: 
                    rand_state+=1
                    net.line['max_loading_percent']=100
                    if len(net.storage)>0:
                        net.storage['controllable']=True
                        # Randomly sample 70% of the evs
                        sampled_rows = net.storage.sample(frac=0.7, random_state=rand_state)
                        # Change the 'controllable' column value to False for the sampled rows
                        net.storage.loc[sampled_rows.index, 'controllable'] = False
                    try:
                        pp.rundcopp(net)
                        break
                    except Exception:
                        attempt+=1
                        if attempt>max_attempt:
                            raise pp.optimal_powerflow.OPFNotConverged("Optimal Power Flow did not converge after several attempts!")

                    
            #Clean up small charging power
            net.res_storage.p_mw.loc[((net.res_storage.p_mw>-1e-7) & (net.res_storage.p_mw<0))|((net.res_storage.p_mw>0) & (net.res_storage.p_mw<1e-7))]=0 
            net.res_sgen.p_mw.loc[((net.res_sgen.p_mw>-1e-7) & (net.res_sgen.p_mw<0))|((net.res_sgen.p_mw>0) & (net.res_sgen.p_mw<1e-7))]=0 

            net.sgen.loc[net.sgen.type=='PV','p_mw'] = net.res_sgen.p_mw
            net.sgen.loc[net.sgen.type=='PV','q_mvar'] = np.tan(np.arccos(0.97))*net.res_sgen.p_mw
            net.sgen.loc[net.sgen.type=='PV','max_p_mw'] = net.res_sgen.p_mw
            net.sgen.loc[net.sgen.type=='PV','max_q_mvar'] = np.tan(np.arccos(0.97))*net.res_sgen.p_mw
            net.sgen.loc[net.sgen.type=='PV','min_q_mvar'] = np.tan(np.arccos(0.97))*net.res_sgen.p_mw*(-1)
            net.storage.loc[net.storage.type == 'ev', 'p_mw'] = net.res_storage.p_mw
            # Restore max Line Loading Percent
            net.line['max_loading_percent']=100
            # Run AC power flow to check voltage and line loading
            pp.runpp(net)
            plt.figure()
            plt.boxplot(net.res_bus.vm_pu)
            plt.ylabel("Bus Voltage Magnitude [pu]")
            plt.savefig(f"{mv_folder_path}/bus_vm_pu_{t}.jpg")
            plt.figure()
            plt.boxplot(net.res_line.loading_percent)
            plt.ylabel("Line Loading [%]")
            plt.savefig(f"{mv_folder_path}/line_loading_{t}.jpg")

            # Save net at time t
            pp.to_excel(net,f"{mv_folder_path}/time_{t}_{grid}.xlsx")

            if len(net.storage)>0:
                # #Update SoC
                net.res_storage['SoC_change'] = net.storage.apply(lambda row:row.SoC_change if row.arr_time_idx==t else 0, axis=1)
                net.res_storage['delta_energy'] = net.res_storage['p_mw']*ev_park_t.loc[t] - net.res_storage['SoC_change']/100*net.storage.max_e_mwh
                SoC_change=(net.res_storage.groupby(net.storage.person)['delta_energy'].sum())/(net.storage.groupby('person')['max_e_mwh'].first())*100
                for key, value in SoC_change.items():
                    SoC_dict[key]+=value
                SoC_t[t] = pd.Series(SoC_dict)
        if len(ev_load_profile)>0:
            SoC_t.to_csv(f"{mv_folder_path}/{grid}_SoC_t.csv")


else:
    for folder in LV_list:
        os.makedirs(f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/modified_swiss_pdg/369_0/2050_01_07/{folder}",exist_ok=True)
        lv_folder_path=f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/modified_swiss_pdg/369_0/2050_01_07/{folder}"
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

        for t in range(24):
            max_attempt=50
            rand_state=1
            attempt=0
            # Reduce Line Loading Limit
            net.line['max_loading_percent']=90


            net.load.loc[net.load.category == 'base', 'p_mw'] = base_p.loc[t].values
            net.load.loc[net.load.category=='base','q_mvar'] = base_q.loc[t].values

            # #PV
            net.sgen.loc[net.sgen.type=='PV','p_mw'] = pv_p.loc[t].values
            net.sgen.loc[net.sgen.type=='PV','q_mvar'] = pv_q.loc[t].values
            net.sgen.loc[net.sgen.type=='PV','max_p_mw'] = pv_p.loc[t].values
            # No limit on PV reactive power
            net.sgen.loc[net.sgen.type=='PV','max_q_mvar'] = pv_p.loc[t].values
            net.sgen.loc[net.sgen.type=='PV','min_q_mvar'] = pv_p.loc[t].values*(-1)

            # #Heatpump
            net.load.loc[net.load.category == 'hp', 'p_mw'] = hp_p.loc[t].values
            net.load.loc[net.load.category == 'hp', 'q_mvar'] = hp_q.loc[t].values
            
            # #EV
            if len(net.storage)>0:
                net.storage['controllable']=True

                net.storage.loc[net.storage.type=='ev','p_mw'] = ev_p.loc[t].values
                # Sync currnet Battery SoC
                net.storage.loc[net.storage.type=='ev','soc_percent'] = net.storage['person'].map(SoC_dict)

                # Randomly sample 70% of the evs
                sampled_rows = net.storage.sample(frac=0.7, random_state=rand_state)
                # Change the 'controllable' column value to False for the sampled rows
                net.storage.loc[sampled_rows.index, 'controllable'] = False

                # Recalculate ev's max/min_p_mw
                net.storage.loc[net.storage.type=='ev','max_p_mw'] = net.storage.apply(lambda row:ev_max_p_mw(ev_park_t.loc[t][row.name],row),axis=1)
                net.storage.loc[net.storage.type=='ev','min_p_mw'] = net.storage.apply(lambda row:ev_min_p_mw(ev_park_t.loc[t][row.name],row),axis=1)

            ########################################################
            ########################################################
            while attempt<max_attempt:
                try:
                    # Run DCOPF with 80% line loading limit
                    pp.rundcopp(net)
                    break
                except Exception as e: 
                    rand_state+=1
                    net.line['max_loading_percent']=100
                    if len(net.storage)>0:
                        net.storage['controllable']=True
                        # Randomly sample 70% of the evs
                        sampled_rows = net.storage.sample(frac=0.7, random_state=rand_state)
                        # Change the 'controllable' column value to False for the sampled rows
                        net.storage.loc[sampled_rows.index, 'controllable'] = False
                    try:
                        pp.rundcopp(net)
                        break
                    except Exception:
                        attempt+=1
                        if attempt>max_attempt:
                            raise pp.optimal_powerflow.OPFNotConverged("Optimal Power Flow did not converge after several attempts!")

                    
            #Clean up small charging power
            net.res_storage.p_mw.loc[((net.res_storage.p_mw>-1e-7) & (net.res_storage.p_mw<0))|((net.res_storage.p_mw>0) & (net.res_storage.p_mw<1e-7))]=0 
            net.res_sgen.p_mw.loc[((net.res_sgen.p_mw>-1e-7) & (net.res_sgen.p_mw<0))|((net.res_sgen.p_mw>0) & (net.res_sgen.p_mw<1e-7))]=0 

            net.sgen.loc[net.sgen.type=='PV','p_mw'] = net.res_sgen.p_mw
            net.sgen.loc[net.sgen.type=='PV','q_mvar'] = np.tan(np.arccos(0.97))*net.res_sgen.p_mw
            net.sgen.loc[net.sgen.type=='PV','max_p_mw'] = net.res_sgen.p_mw
            net.sgen.loc[net.sgen.type=='PV','max_q_mvar'] = np.tan(np.arccos(0.97))*net.res_sgen.p_mw
            net.sgen.loc[net.sgen.type=='PV','min_q_mvar'] = np.tan(np.arccos(0.97))*net.res_sgen.p_mw*(-1)
            net.storage.loc[net.storage.type == 'ev', 'p_mw'] = net.res_storage.p_mw
            # Restore max Line Loading Percent
            net.line['max_loading_percent']=100
            # Run AC power flow to check voltage and line loading
            pp.runpp(net)
            plt.figure()
            plt.boxplot(net.res_bus.vm_pu)
            plt.ylabel("Bus Voltage Magnitude [pu]")
            plt.savefig(f"{lv_folder_path}/bus_vm_pu_{t}.jpg")
            plt.figure()
            plt.boxplot(net.res_line.loading_percent)
            plt.ylabel("Line Loading [%]")
            plt.savefig(f"{lv_folder_path}/line_loading_{t}.jpg")

            # Save net at time t
            pp.to_excel(net,f"{lv_folder_path}/time_{t}_{folder}.xlsx")

            if len(net.storage)>0:
                # #Update SoC
                net.res_storage['SoC_change'] = net.storage.apply(lambda row:row.SoC_change if row.arr_time_idx==t else 0, axis=1)
                net.res_storage['delta_energy'] = net.res_storage['p_mw']*ev_park_t.loc[t] - net.res_storage['SoC_change']/100*net.storage.max_e_mwh
                SoC_change=(net.res_storage.groupby(net.storage.person)['delta_energy'].sum())/(net.storage.groupby('person')['max_e_mwh'].first())*100
                for key, value in SoC_change.items():
                    SoC_dict[key]+=value
                SoC_t[t] = pd.Series(SoC_dict)
        if len(ev_load_profile)>0:
            SoC_t.to_csv(f"{lv_folder_path}/{folder}_SoC_t.csv")
