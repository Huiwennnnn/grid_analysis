import numpy as np
import pandas as pd
import pyomo.kernel as pmo
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.util.infeasible import log_infeasible_constraints
import logging
import re
import pandapower as pp
import scipy
from scipy.sparse import csr_matrix
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import networkx as nx
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Run ACOPF with DistFlow linearization.')
parser.add_argument('--scenario_year', type=int, required=True, help='Scenario year')
parser.add_argument('--month', type=int, required=True, help='Month')
parser.add_argument('--day', type=int, required=True, help='Day')
parser.add_argument('--weekday', type=str, required=True, help='Weekday')
parser.add_argument('--monitor_hr', type=int, required=True, help='Monitor hours')

args = parser.parse_args()

# Extract arguments
scenario_year = args.scenario_year
month = args.month
day = args.day
weekday = args.weekday
monitor_hr = args.monitor_hr
folder = '369_0'
zone='Midlands-Periurban'
MV_feeder = True
grid = "369_0"

##############
# scenario_year = 2050
# month = 7
# day = 8
# weekday = "Friday"
# monitor_hr = 24
##############
day_start_ts = pd.to_datetime(f"{scenario_year}-{month:02d}-{day:02d} 00:00:00")
day_start = day_start_ts.day
day_end_ts = day_start_ts+pd.Timedelta(hours=monitor_hr)
path = f"/cluster/home/huiluo/mt/grid_369_0/DistFlow/{scenario_year}_{month:02d}_{day:02d}_{monitor_hr}_v1g"
os.makedirs(path,exist_ok=True)
TP_dict = {1:2,2:2,3:2,4:3,5:3,6:4,7:4,8:4,9:4,10:3,11:2,12:1}
TP = TP_dict[day_start_ts.month] # 1:Dec, 2:Jan,Feb,Mar,Nov, 3:Apr,May,Oct, 4:Jun,Jul,Aug,Sep
hourly_timestamps = pd.date_range(start=day_start_ts, periods=monitor_hr, freq='h')
hourly_labels = hourly_timestamps.strftime('%m-%d %H:%M')

#########################################
# Load Network
#########################################
net = pp.from_pickle(f'{path}/369_0.p')
base_load_profile = pd.read_pickle(f'{path}/369_0_base.pkl')
hp_load_profile=pd.read_pickle(f'{path}/369_0_hp.pkl')
pv_load_profile=pd.read_pickle(f'{path}/369_0_pv.pkl')
ev_load_profile=pd.read_pickle(f'{path}/369_0_ev.pkl')


base_p = pd.DataFrame(base_load_profile.p_mw).T.apply(pd.Series.explode).reset_index(drop=True)
base_q = pd.DataFrame(base_load_profile.q_mvar).T.apply(pd.Series.explode).reset_index(drop=True)
hp_p = pd.DataFrame(hp_load_profile.p).T.apply(pd.Series.explode).reset_index(drop=True)
hp_q = pd.DataFrame(hp_load_profile.q).T.apply(pd.Series.explode).reset_index(drop=True)
pv_p = pd.DataFrame(pv_load_profile.pv_P_daily).T.apply(pd.Series.explode).reset_index(drop=True)
pv_q = pd.DataFrame(pv_load_profile.pv_Q_daily).T.apply(pd.Series.explode).reset_index(drop=True)
ev_p = pd.DataFrame(ev_load_profile.optimized_power_list).T.apply(pd.Series.explode).reset_index(drop=True)


#######################################
# Benchmark Result
#######################################
line_loading_origin = pd.DataFrame()
trafo_loading_origin = pd.DataFrame()
vm_pu_origin = pd.DataFrame()

for t in range(monitor_hr):
    net.load.loc[net.load.category == 'base', 'p_mw'] = base_p.loc[t].values
    net.load.loc[net.load.category=='base','q_mvar'] = base_q.loc[t].values
    net.load.loc[net.load.category == 'hp', 'p_mw'] = hp_p.loc[t].values
    net.load.loc[net.load.category == 'hp', 'q_mvar'] = hp_q.loc[t].values
    net.sgen.loc[net.sgen.type=='PV','p_mw'] = pv_p.loc[t].values
    net.sgen.loc[net.sgen.type=='PV','q_mvar'] = pv_q.loc[t].values*np.tan(np.arccos(0.97))
    net.storage.loc[net.storage.type=='ev','p_mw'] = ev_p.loc[t].values
    pp.runpp(net)
    # pp.to_pickle(net,filename=f'{path}/{folder}_hour_{t}_dcopf_res_origin.p')
    vm_pu_origin[t] = net.res_bus.vm_pu
    line_loading_origin[t] = net.res_line.loading_percent
    trafo_loading_origin[t] = net.res_trafo.loading_percent
    # trafo_p_origin[t] = net.res_trafo.p_mw
    # trafo_q_origin[t] = net.res_trafo.q_mvar

plt.figure(figsize=(10*monitor_hr//24, 6))
plt.boxplot(vm_pu_origin)
plt.title('vm_pu_results')
plt.xlabel('Time')
plt.ylabel('Loading Percentage [%]')
plt.xticks(ticks=range(1, len(hourly_labels) + 1), labels=hourly_labels, rotation=90)
plt.tight_layout()
plt.savefig(f'{path}/{folder}_vmpu_origin.jpg')

plt.figure(figsize=(10*monitor_hr//24, 6))
plt.boxplot(trafo_loading_origin)
plt.title('trafo_loading_results')
plt.xlabel('Time')
plt.ylabel('Loading Percentage [%]')
plt.xticks(ticks=range(1, len(hourly_labels) + 1), labels=hourly_labels, rotation=90)
plt.tight_layout()
plt.savefig(f'{path}/{folder}_trafoloading_origin.jpg')

plt.figure(figsize=(10*monitor_hr//24, 6))
plt.boxplot(line_loading_origin)
plt.title('line_loading_results')
plt.xlabel('Time')
plt.ylabel('Loading Percentage [%]')
plt.xticks(ticks=range(1, len(hourly_labels) + 1), labels=hourly_labels, rotation=90)
plt.tight_layout()
plt.savefig(f'{path}/{folder}_lineloading_origin.jpg')


#####################################################
# Model Parameter
#####################################################
try:
    pp.runpp(net)
except pp.LoadflowNotConverged:
    print("Load flow did not converge.")   
coo_Ybus = net._ppc["internal"]["Ybus"].tocoo()
Ybus_g_dict = {(i, j): v for i, j, v in zip(coo_Ybus.row, coo_Ybus.col, coo_Ybus.data.real)}
Ybus_b_dict = {(i, j): v for i, j, v in zip(coo_Ybus.row, coo_Ybus.col, coo_Ybus.data.imag)} 
Zbus_r_dict = {(i, j): 1/v for i, j, v in zip(coo_Ybus.row, coo_Ybus.col, coo_Ybus.data.real)}
Zbus_x_dict = {(i, j): 1/v for i, j, v in zip(coo_Ybus.row, coo_Ybus.col, coo_Ybus.data.imag)}

branch_data = net._ppc['branch']
fbus = branch_data[:,0].astype(int)
tbus = branch_data[:,1].astype(int)
branch_ft_bus_tuple = list(zip(fbus,tbus))

BRANCH_SMAX_trafo = dict(zip(list(zip(net.trafo.hv_bus,net.trafo.lv_bus)),net.trafo.sn_mva)) 
BRANCH_SMAX_line = dict(zip(list(zip(net.line.from_bus,net.line.to_bus)),net.line.parallel * net.line.max_i_ka * net.bus.loc[net.line.from_bus.values].vn_kv.values* np.sqrt(3))) 
BRANCH_SMAX_line.update(BRANCH_SMAX_trafo)

# Base Load Data
base_to_BUS_dict = base_load_profile.Bus.to_dict()
base_PD_coo = scipy.sparse.coo_matrix(np.array(base_load_profile.p_mw.tolist()))
base_PD_dict = {(base_to_BUS_dict[base],t):v for base, t, v in zip(base_PD_coo.row,base_PD_coo.col,base_PD_coo.data)}
base_QD_coo = scipy.sparse.coo_matrix(np.array(base_load_profile.q_mvar.tolist()))
base_QD_dict = {(base_to_BUS_dict[base],t):v for base, t, v in zip(base_QD_coo.row,base_QD_coo.col,base_QD_coo.data)} 

# HP Data
HP_to_BUS_dict = hp_load_profile.Bus.to_dict()
hp_bus_tuple =[(HP,int(BUS)) for HP,BUS in  HP_to_BUS_dict.items()]
HP_max_dict = {}
HPQ_dict = {}
for hp, row in hp_load_profile.iterrows():
    bus = row['Bus']
    for t, p in enumerate(row['p']):
        HP_max_dict[(hp, bus, t)] = p 
    for t, q in enumerate(row['q']):
        HPQ_dict[(hp,bus,t)] = q 
# PV data
PV_to_BUS_dict = pv_load_profile.Bus.to_dict()
pv_bus_tuple =[(PV,int(BUS)) for PV,BUS in  PV_to_BUS_dict.items()]
PV_Pmax_coo = scipy.sparse.coo_matrix(np.array(pv_load_profile.pv_P_daily.tolist())) # sparse matrix:row: pv index -> column: monitor_hr -> data: maximal PV power generation
PV_Pmax_dict = {(pv,int(PV_to_BUS_dict[pv]),t): v for pv, t, v in zip(PV_Pmax_coo.row, PV_Pmax_coo.col, PV_Pmax_coo.data)} 

# EV Data
PARK_to_EVBAT_dict = ev_load_profile.person.to_dict() # key: parking event -> value: person
PARK_to_BUS_dict = ev_load_profile.Bus.to_dict() # key: parking event -> vlaue: bus_id
PARK_to_ENDIDX_dict = ev_load_profile.park_end_time_idx.to_dict() # key:parking event -> value: park_end_time_idx
PARK_to_ARRIDX_dict = ev_load_profile.arr_time_idx.to_dict() # key:parking event -> value: arr_time_idx
park_bus_tuple = [(PARK,int(PARK_to_EVBAT_dict[PARK]),max(PARK_to_ARRIDX_dict[PARK],0),min(int(PARK_to_ENDIDX_dict[PARK]),monitor_hr-1),int(BUS)) for PARK,BUS in  PARK_to_BUS_dict.items()] # (Parking event person,arr_time_idx, park_end_time_idx, bus)

PARK_PD_coo = scipy.sparse.coo_matrix(ev_load_profile.optimized_power_list.tolist()) # sparse matrix:row: parking event -> column: monitor_hr -> data: origin charging power of battery in consumer system
PARK_PD_dict = {(PARK,int(PARK_to_EVBAT_dict[PARK]),max(PARK_to_ARRIDX_dict[PARK],0),min(int(PARK_to_ENDIDX_dict[PARK]),monitor_hr-1),int(PARK_to_BUS_dict[PARK]), t): -v for PARK, t, v in zip(PARK_PD_coo.row, PARK_PD_coo.col, PARK_PD_coo.data)} # key:(parking event,perosn,arr_time_idx,park_end_time_idx, bus_id, monitor_hr) -> value: ev battery injectied power from grid's perspective

PARKHR_coo = scipy.sparse.coo_matrix(ev_load_profile.hourly_time_dict.tolist()) 
PARKHR_dict = {(PARK,int(PARK_to_EVBAT_dict[PARK]),max(PARK_to_ARRIDX_dict[PARK],0),min(int(PARK_to_ENDIDX_dict[PARK]),monitor_hr-1), int(PARK_to_BUS_dict[PARK]),t):v/60 for PARK,t,v in zip(PARKHR_coo.row,PARKHR_coo.col,PARKHR_coo.data)} # key:(parking event,perosn,arr_time_idx,park_end_time_idx, bus_id, monitor_hr) -> value: parking duration in Hour for this hour

SOE_dayend_dict = (ev_load_profile[['person','day_end_soe']].groupby('person').last().day_end_soe/1e3).to_dict()

NEXT_E = ev_load_profile[f'next_travel_TP{TP}_consumption']/1e3

NEXT_E_dict =  {(PARK,int(PARK_to_EVBAT_dict[PARK]),max(PARK_to_ARRIDX_dict[PARK],0),min(int(PARK_to_ENDIDX_dict[PARK]),monitor_hr-1),int(PARK_to_BUS_dict[PARK])):next_e for PARK,next_e in zip(NEXT_E.index,NEXT_E.values)}
NEXT_2E = ev_load_profile[f'next_travel_TP{TP}_consumption']/1e3

NEXT_2E_dict =  {(PARK,int(PARK_to_EVBAT_dict[PARK]),max(PARK_to_ARRIDX_dict[PARK],0),min(int(PARK_to_ENDIDX_dict[PARK]),monitor_hr-1),int(PARK_to_BUS_dict[PARK])):next_2e for PARK,next_2e in zip(NEXT_2E.index,NEXT_2E.values)}
SOE_change = scipy.sparse.coo_matrix(np.array(ev_load_profile.SoE_change/1e3))


SOE_change_dict = {(PARK,int(PARK_to_EVBAT_dict[PARK]),max(PARK_to_ARRIDX_dict[PARK],0),min(int(PARK_to_ENDIDX_dict[PARK]),monitor_hr-1),int(PARK_to_BUS_dict[PARK])):soe_change for PARK, soe_change in zip(SOE_change.col,SOE_change.data)}

FIRST_PARKING = ev_load_profile[ev_load_profile.parking_cnt==0]
SOE_init_dict = dict(zip(FIRST_PARKING['person'],FIRST_PARKING['update_SoE_bc']/1e3))
EVCAP_dict = dict(zip(FIRST_PARKING.person,FIRST_PARKING.B/1e3))
PLUGIN_dict = (pd.DataFrame(ev_load_profile.hourly_time_dict.tolist())!=0).sum().to_dict()


####################################################
# Network Topology
####################################################
graph = pp.topology.create_nxgraph(net)
root_node = net.ext_grid.bus.at[0]
bfs_tree = nx.bfs_tree(graph, source=root_node)
# Create a dictionary to store parent and children information for each node
node_parent = {}
node_child = {}
# Initialize parent information (root node has no parent, marked as None or -1)
for node in bfs_tree.nodes():
    node_parent[node] = list(bfs_tree.predecessors(node))[0]  if list(bfs_tree.predecessors(node)) else -1
    node_child[node] = list(bfs_tree.successors(node)) if list(bfs_tree.successors(node)) else -1
parent_child_tuple = [(parent, child) for child, parent in node_parent.items() if parent != -1]



##################################
# Optimization
##################################


class Grid(pmo.block):
    def __init__(self,slack=None,bus_set=None, branch_set=None, hp_set=None, pv_set=None, park_set=None, t_set=None,BRANCH_SMAX_line=None,Zbus_r_dict=None,Zbus_x_dict=None,base_PD_dict = None,base_QD_dict = None,HP_max_dict=None,HPQ_dict=None,PV_Pmax_dict=None,ev_load_profile=None): 
        super(Grid,self).__init__()
        # Const
        self._slack = slack
        # set
        self.BUS=bus_set
        self.BRANCH=branch_set
        self.HP = hp_set
        self.PV = pv_set
        self.PARK = park_set
        self.T = t_set

        # Vars
        self.V2 = pmo.variable_dict() # bus,t
        self.V_slack = pmo.variable_dict() #bus, t
        self.PVGENP = pmo.variable_dict() # pv,bus,t
        self.PVGENQ = pmo.variable_dict() # pv,bus,t
        self.QBRANCH = pmo.variable_dict()# parent,child,t
        self.PBRANCH = pmo.variable_dict()# parent,child,t
        self.EVGENP = pmo.variable_dict()# park,ev,arr_idx,end_idx,bus,t
        self.PEXTGENP = pmo.variable_dict()# t
        self.PEXTGENQ = pmo.variable_dict()# t


        # Params
        self._BUILDINGPD = pmo.parameter_dict() #bus,t
        self._BUILDINGQD = pmo.parameter_dict() #bus,t
        self._HPPD = pmo.parameter_dict() #hp,bus,t
        self._HPQD = pmo.parameter_dict()#hp,bus,t
        self._PV_PMAX = pmo.parameter_dict()#pv,bus,t
        self._BRANCH_MAXS = pmo.parameter_dict()#parent,chid
        self._ZBUSR = pmo.parameter_dict()#parent,child
        self._ZBUSX = pmo.parameter_dict()#parent,child
        self._ev_load_profile =ev_load_profile
        self._BRANCH_SMAX_line = BRANCH_SMAX_line
        self._Zbus_r_dict = Zbus_r_dict
        self._Zbus_x_dict = Zbus_x_dict

        # Initalize variables
        for t in self.T:
            self.PEXTGENP[t]=pmo.variable(domain=pmo.Reals,value=0)
            self.PEXTGENQ[t]=pmo.variable(domain=pmo.Reals,value=0)

            for bus in self.BUS:
                self.V2[bus,t]=pmo.variable(domain=pmo.NonNegativeReals,value=1)
                self.V_slack[bus,t] = pmo.variable(domain=pmo.NonNegativeReals,value=0)
                
            for parent,child in self.BRANCH:
                self.PBRANCH[parent,child,t]=pmo.variable(domain=pmo.Reals,value=0)
                self.QBRANCH[parent,child,t]=pmo.variable(domain=pmo.Reals,value=0)

        for t in self.T:
            for pv,bus in self.PV:
                self.PVGENP[pv,bus,t]=pmo.variable(domain=pmo.NonNegativeReals,value=0)
                self.PVGENQ[pv,bus,t]=pmo.variable(domain=pmo.Reals,value=0)

        for t in self.T:
            for park,ev,arr_idx,end_idx,bus in self.PARK:
                EV_bounds = self.EVGENP_bound([park,ev,arr_idx,end_idx,bus],t)
                self.EVGENP[park,ev,arr_idx,end_idx,bus,t]=pmo.variable(lb=EV_bounds[0],ub=EV_bounds[1],value=0)

        # Initialzie parameters
        for t in self.T:
            for bus in self.BUS:
                self._BUILDINGPD[bus,t] = pmo.parameter(value=base_PD_dict.get((bus,t),0))
                self._BUILDINGQD[bus,t] = pmo.parameter(value=base_QD_dict.get((bus,t),0))
                for hp,b in self.HP:
                    if b==bus:
                        self._HPPD[hp,b,t] = pmo.parameter(value=HP_max_dict.get((hp,b,t),0))
                        self._HPQD[hp,b,t] = pmo.parameter(value=HPQ_dict.get((hp,b,t),0))
                for pv,b in self.PV:
                    if b==bus:
                        self._PV_PMAX[pv,b,t] = pmo.parameter(value=PV_Pmax_dict.get((pv,bus,t),0))
        for parent, child in self.BRANCH:
            self._BRANCH_MAXS[parent,child] = pmo.parameter(value=self.branch_smax_init_rule(parent,child))
            self._ZBUSR[parent,child] = pmo.parameter(value=self.zbus_r_init_rule(parent,child))
            self._ZBUSX[parent,child] = pmo.parameter(value=self.zbus_x_init_rule(parent,child))
        
        
        # Constriants
        """
        Power Balance Constraints
        """
        self.ActivePowerBalance = pmo.constraint_list()
        self.active_power_balance_tolist()

        self.ReactivePowerBalance = pmo.constraint_list()
        self.reactive_power_balance_tolist()


        """
        Power Flow Constraints
        """
        self.VoltageDifference = pmo.constraint_dict()
        # self.SlackVoltage = pmo.constraint_dict()
        self.LineUpper = pmo.constraint_dict()
        for parent,child in self.BRANCH:
            for t in self.T:
                self.VoltageDifference[parent,child,t] = pmo.constraint(self.V2[parent,t]-self.V2[child,t]==2*(self.PBRANCH[parent,child,t]*self._ZBUSR[parent,child] + self.QBRANCH[parent,child,t]*self._ZBUSX[parent,child]))
                self.LineUpper[parent,child,t] = pmo.constraint(self.PBRANCH[parent,child,t]*self.PBRANCH[parent,child,t]+self.QBRANCH[parent,child,t]*self.QBRANCH[parent,child,t]<=self._BRANCH_MAXS[parent,child]*self._BRANCH_MAXS[parent,child])

        # for t in self.T:
        #     self.SlackVoltage[t] = pmo.constraint(self.V2[self._slack,t]==1)
        """
        Voltage Limits
        """
        self.VoltageUpper = pmo.constraint_dict()
        for bus in self.BUS:
            for t in self.T:
                self.VoltageUpper[bus,t]=pmo.constraint(self.V2[bus,t]<=1.21+self.V_slack[bus,t])

        self.VoltageLower = pmo.constraint_dict()
        for bus in self.BUS:
            for t in self.T:
                self.VoltageLower[bus,t]=pmo.constraint(self.V2[bus,t]+self.V_slack[bus,t]>=0.8464) 
                
        """
        PV Limits
        """
        self.PvPUpper = pmo.constraint_dict()
        self.PvQUpper = pmo.constraint_dict()
        self.PvQLower = pmo.constraint_dict()
        for pv, bus in m.PV:
            for t in m.T:
                self.PvPUpper[pv,bus,t]=pmo.constraint(self.PVGENP[pv,bus,t]<=self._PV_PMAX[pv,bus,t])
                self.PvQUpper[pv,bus,t]=pmo.constraint(self.PVGENQ[pv,bus,t]<=self.PVGENP[pv,bus,t]*np.tan(np.arccos(0.85)))
                self.PvQLower[pv,bus,t]=pmo.constraint(self.PVGENQ[pv,bus,t]>=-self.PVGENP[pv,bus,t]*np.tan(np.arccos(0.85)))
                
            
    def zbus_r_init_rule(self,parent,child):
        if (parent,child) in self._Zbus_r_dict.keys():
            return self._Zbus_r_dict[(parent,child)]
        else:
            return self._Zbus_r_dict.get((child,parent),0)
        
    def zbus_x_init_rule(self,parent,child):
        if (parent,child) in self._Zbus_x_dict.keys():
            return self._Zbus_x_dict[(parent,child)]
        else:
            return self._Zbus_x_dict.get((child,parent),0)
        
    def branch_smax_init_rule(self,parent,child):
        if (parent,child) in self._BRANCH_SMAX_line.keys():
            return self._BRANCH_SMAX_line[(parent,child)]
        else:
            return self._BRANCH_SMAX_line.get((child,parent),0)

    def active_power_balance_tolist(self):
        for bus in self.BUS:
            for t in self.T:
                self.ActivePowerBalance.append(pmo.constraint(self.active_power_balance_lhs(bus,t)==0))

    def active_power_balance_lhs(self,bus,t):
        gen_power = (
                sum(self.PVGENP[pv,bus,t] for pv,b in self.PV if b==bus) + 
                sum(self.EVGENP[park,ev,arr_idx,end_idx,bus,t] for park,ev,arr_idx,end_idx,b in self.PARK if b==bus)
                +(self.PEXTGENP[t] if bus==self._slack else 0)
            )
        load_power = self._BUILDINGPD[bus,t] + sum(self._HPPD[HP,b,t] for HP,b in self.HP if b==bus)
        net_power_injection = gen_power - load_power
        incoming_flow = sum(self.PBRANCH[parent,child,t] for parent, child in self.BRANCH if child==bus)
        outgoing_flow = sum(self.PBRANCH[parent,child,t] for parent, child in self.BRANCH if parent==bus)
        return net_power_injection+incoming_flow-outgoing_flow
    
    def reactive_power_balance_tolist(self):
        for bus in self.BUS:
            for t in self.T:
                self.ActivePowerBalance.append(pmo.constraint(self.reactive_power_balance_lhs(bus,t)==0))

    def reactive_power_balance_lhs(self,bus,t):
        gen_power = (
                sum(self.PVGENQ[pv,bus,t] for pv,b in self.PV if b==bus) + 
                +(self.PEXTGENQ[t] if bus==self._slack else 0)
            )
        load_power = self._BUILDINGQD[bus,t] + sum(self._HPQD[HP,b,t] for HP,b in self.HP if b==bus)
        net_power_injection = gen_power - load_power
        incoming_flow = sum(self.QBRANCH[parent,child,t] for parent, child in self.BRANCH if child==bus)
        outgoing_flow = sum(self.QBRANCH[parent,child,t] for parent, child in self.BRANCH if parent==bus)
        return net_power_injection+incoming_flow-outgoing_flow

    def EVGENP_bound(self, PARK, t):
        if ev_load_profile.loc[PARK[0]].hourly_time_dict[t] <= 0:
            return (0, 0)
        else:
            chg_rate_MW = ev_load_profile.loc[PARK[0]]['chg rate'] / 1e3
            return (-chg_rate_MW, 0)    

class Car(pmo.block):
    def __init__(self,cars,EVGENP,monitor_hr,PARK=None,EVBAT=None,T=None,PLUGIN_dict=None,ev_load_profile=None): 
        super(Car,self).__init__()

        # const
        M=1e10

        # set
        self.PARK=PARK
        self.EVBAT=EVBAT
        self.T = T


        # Vars
        self.SOE = pmo.variable_dict()
        for person in self.EVBAT:
            for t in self.T:
                self.SOE[person,t]=pmo.variable(domain=pmo.NonNegativeReals,value=0)

        self.EVP_SHIFT_ABS = pmo.variable_dict()
        for park,ev,arr_idx,end_idx,bus in self.PARK:
            for t in self.T:
                self.EVP_SHIFT_ABS[park,ev,arr_idx,end_idx,bus,t] = pmo.variable(domain=pmo.NonNegativeReals,value=0)
        
        self.SOE_dayend_slack = pmo.variable_dict()
        for person in self.EVBAT:
            self.SOE_dayend_slack[person]=pmo.variable(domain=pmo.NonNegativeReals,value=0)

        self.FLEX = pmo.variable_dict()
        for park,ev,arr_idx,end_idx,bus in self.PARK:
            for t in self.T:
                self.FLEX[park,ev,arr_idx,end_idx,bus,t]=pmo.variable(domain=pmo.Binary,value=0)

        # Parameters
        self._SOE_init = pmo.parameter_dict()
        self._EVCAP = pmo.parameter_dict()
        self._SOE_dayend = pmo.parameter_dict()
        self._Plugin = PLUGIN_dict

        for person in self.EVBAT:
            self._EVCAP[person] = pmo.parameter(value=EVCAP_dict.get(person))
            self._SOE_dayend[person] = pmo.parameter(value=SOE_dayend_dict.get(person))
            self._SOE_init[person] = pmo.parameter(value=SOE_init_dict.get(person,0))

        self._NEXT_E = pmo.parameter_dict()
        self._SOE_change = pmo.parameter_dict()
        for park,ev,arr_idx,end_idx,bus in self.PARK:
            self._NEXT_E[park,ev,arr_idx,end_idx,bus] = pmo.parameter(value=NEXT_E_dict.get((park,ev,arr_idx,end_idx,bus)))
            self._SOE_change[park,ev,arr_idx,end_idx,bus] = pmo.parameter(value=SOE_change_dict.get((park,ev,arr_idx,end_idx,bus),0))
        
        self._PARK_HR = pmo.parameter_dict()
        self.EVPPLAN = pmo.parameter_dict()
        for park,ev,arr_idx,end_idx,bus in self.PARK:
            for t in self.T:
                self._PARK_HR[park,ev,arr_idx,end_idx,bus,t]=pmo.parameter(value=PARKHR_dict.get((park,ev,arr_idx,end_idx,bus,t),0))
                self.EVPPLAN[park,ev,arr_idx,end_idx,bus,t]=pmo.parameter(value=PARK_PD_dict.get((park,ev,arr_idx,end_idx,bus,t),0))
        
        # Constraints
        """
        SoE updates and upper limits
        """
        self.SoeUpdate = pmo.constraint_dict()
        self.soe_update_tolist(EVGENP)

        self.SoeUpper = pmo.constraint_list()
        self.soe_upper_tolist()

        self.SoeLower = pmo.constraint_list()
        self.soe_lower_tolist()

        """
        Next trip requirements
        """
        self.NextE = pmo.constraint_dict()
        self.next_e_tolist(EVGENP,monitor_hr)

        """
        Flexibility Constraints
        """
        self.ShiftAbs1 = pmo.constraint_dict()
        self.evp_shift_1_tolist(EVGENP)

        self.ShiftAbs2 = pmo.constraint_dict()
        self.evp_shift_2_tolist(EVGENP)



        # self.DetermineShift1=pmo.constraint_dict()
        # for park,ev,arr_idx,end_idx,bus in self.PARK:
        #     for t in self.T:
        #         self.DetermineShift1[park,ev,arr_idx,end_idx,bus] = pmo.constraint(self.P_shift[park,ev,arr_idx,end_idx,bus,t]<=M*self.FLEX[park,ev,arr_idx,end_idx,bus,t])

        # self.DetermineShift2=pmo.constraint_dict()
        # for park,ev,arr_idx,end_idx,bus in self.PARK:
        #     for t in self.T:
        #         self.DetermineShift2[park,ev,arr_idx,end_idx,bus,t] = pmo.constraint(self.P_shift[park,ev,arr_idx,end_idx,bus,t]>=1e-3-M*(1-self.FLEX[park,ev,arr_idx,end_idx,bus,t]))

        self.Participate = pmo.constraint_dict()
        for t in self.T:
            self.Participate[t] =pmo.constraint(sum(self.FLEX[park,ev,arr_idx,end_idx,bus,t] for park,ev,arr_idx,end_idx,bus in self.PARK) <= 0.3*self._Plugin[t])

        """
        Period End SOE
        """
        self.EndSoe = pmo.constraint((self.SOE[person,monitor_hr-1]-sum(EVGENP[park,ev,arr_idx,end_idx,bus,monitor_hr-1]*self._PARK_HR[park,ev,arr_idx,end_idx,bus,monitor_hr-1] for park,ev,arr_idx,end_idx,bus in self.PARK if ev==person))+self.SOE_dayend_slack[person] >= self._SOE_dayend[person])

    def soe_update_tolist(self,EVGENP):
        for person in self.EVBAT:
            for t in self.T:
                self.SoeUpdate[person,t] = pmo.constraint(self.soe_update_rule(person,t,EVGENP) == self.SOE[person, t])

    def soe_update_rule(self,person,t,EVGENP):
        if t==0:
            return self._SOE_init[person]
        else:
            return (self.SOE[person,t-1]
                    +sum(-EVGENP[park,ev,arr_idx,end_idx,bus,t-1]*self._PARK_HR[park,ev,arr_idx,end_idx,bus,t-1] for park,ev,arr_idx,end_idx,bus in self.PARK if ev==person)
                    -sum(self._SOE_change[park,ev,arr_idx,end_idx,bus] for park,ev,arr_idx,end_idx,bus in m.PARK if (ev==person and t==arr_idx))
            )
    
    def soe_upper_tolist(self):
        for person in self.EVBAT:
            for t in self.T:
                self.SoeUpper.append(pmo.constraint(self.SOE[person,t]<=self._EVCAP[person]))

    def soe_lower_tolist(self):
        for person in self.EVBAT:
            for t in self.T:
                self.SoeUpper.append(pmo.constraint(self.SOE[person,t]>=0.05*self._EVCAP[person]))

    def next_e_tolist(self,EVGENP,monitor_hr):
        for park,ev,arr_idx,end_idx,bus in self.PARK:
            for t in self.T:
                if t==end_idx and t<monitor_hr-1 and arr_idx!=end_idx:
                    self.NextE[park]=pmo.constraint(self.SOE[ev,t]-EVGENP[park,ev,arr_idx,end_idx,bus,t]*self._PARK_HR[park,ev,arr_idx,end_idx,bus,t]>=self._NEXT_E[park,ev,arr_idx,end_idx,bus]) 

    def evp_shift_1_tolist(self,EVGENP):
        for park,ev,arr_idx,end_idx,bus in self.PARK:
            for t in self.T:
                self.ShiftAbs1[park,t]=pmo.constraint(M*self.FLEX[park,ev,arr_idx,end_idx,bus,t]>=1000*(EVGENP[park,ev,arr_idx,end_idx,bus,t]-self.EVPPLAN[park,ev,arr_idx,end_idx,bus,t]))
                # self.ShiftAbs1[park,t]=pmo.constraint(self.EVP_SHIFT_ABS[park,ev,arr_idx,end_idx,bus,t]>=(EVGENP[park,ev,arr_idx,end_idx,bus,t]-self.EVPPLAN[park,ev,arr_idx,end_idx,bus,t]))

    def evp_shift_2_tolist(self,EVGENP):
        for park,ev,arr_idx,end_idx,bus in self.PARK:
            for t in self.T:
                self.ShiftAbs2[park,t]=pmo.constraint(M*self.FLEX[park,ev,arr_idx,end_idx,bus,t]>=-1000*(EVGENP[park,ev,arr_idx,end_idx,bus,t]-self.EVPPLAN[park,ev,arr_idx,end_idx,bus,t])) 
                # self.ShiftAbs2[park,t]=pmo.constraint(self.EVP_SHIFT_ABS[park,ev,arr_idx,end_idx,bus,t]>=-(EVGENP[park,ev,arr_idx,end_idx,bus,t]-self.EVPPLAN[park,ev,arr_idx,end_idx,bus,t]))      


      
 
m = pmo.block()
# Consts
M=1e8

epsilon = 1e-5
slack=net._ppc['internal']['ref'][0]
cars = len(net.storage.person.unique())

# Sets
m.BUS = net.bus.index.values
m.BRANCH = parent_child_tuple
m.EVBAT = net.storage.person.unique()
m.HP = hp_bus_tuple
m.PV = pv_bus_tuple
m.PARK = park_bus_tuple
m.T = range(monitor_hr)
m.Grids = Grid(slack,m.BUS,m.BRANCH,m.HP,m.PV,m.PARK,m.T,BRANCH_SMAX_line,Zbus_r_dict,Zbus_x_dict,base_PD_dict,base_QD_dict,HP_max_dict,HPQ_dict,PV_Pmax_dict,ev_load_profile) # Grid Part 
m.Cars = Car(cars,m.Grids.EVGENP,monitor_hr,m.PARK,m.EVBAT,m.T,PLUGIN_dict,ev_load_profile) # EV part

m.obj = pmo.objective(
    sum(1e3*m.Cars.SOE_dayend_slack[person]-m.Cars.SOE[person,monitor_hr-1] for person in m.EVBAT)
    # +sum(m.Cars.FLEX[park,ev,arr_idx,end_idx,bus,t] for park,ev,arr_idx,end_idx,bus in m.PARK for t in m.T)
    # +sum(m.Cars.EVP_SHIFT_ABS[park,ev,arr_idx,end_idx,bus,t] for park,ev,arr_idx,end_idx,bus in m.PARK for t in m.T)
    +sum(m.Grids.V_slack[bus,t] for bus in m.BUS for t in m.T),
    sense=pmo.minimize
    )
##################################
# Solve model
##################################
# Step 1: Set up logging configuration
logging.basicConfig(filename=f'{path}/infeasible_constraints.log', 
                    level=logging.INFO, 
                    format='%(asctime)s %(levelname)s: %(message)s')

# Step 2: Define a custom function to log infeasible constraints
def log_infeasible_constraints_to_file(model):
    # Redirect output to a string
    from io import StringIO
    import sys
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    # Call the existing function to print infeasible constraints
    log_infeasible_constraints(model)
    sys.stdout = old_stdout
    infeasible_constraints_output = mystdout.getvalue()
    logging.info(infeasible_constraints_output)



solver = pmo.SolverFactory('gurobi')
solver.options['PreSparsify']=2
# solver.options['Method'] = 1
# solver.options['presolve'] = 2
solver.options['Cuts'] = 3
solver.options['NoRelHeurTime'] = 300
# solver.options['NodeMethod'] = 2
solver.options['MIPGap']=0.3
solver.options['MIPFocus'] = 1
solver.options['NodefileStart']=0
results = solver.solve(m, tee=True)
print(results.solver.status)
# log_infeasible_constraints(m)
log_infeasible_constraints_to_file(m)



###################################
# Data Treatment
###################################
save_path = path
# next_e_slack = {park:pmo.value(m.Cars.NEXT_E_slack[park,ev,arr_idx,end_idx,bus]) for park,ev,arr_idx,end_idx,bus in m.PARK}
# next_e_slack_df = pd.DataFrame.from_dict(next_e_slack, orient='index', columns=['slack'])
# next_e_slack_df.to_csv(f"{save_path}/next_slack.csv")

soe_data = {(person, t): pmo.value(m.Cars.SOE[person, t]) for person in m.EVBAT for t in m.T}
soe_df = pd.DataFrame.from_dict(soe_data, orient='index', columns=['SOE'])
soe_df.index = pd.MultiIndex.from_tuples(soe_df.index, names=['person', 'time'])
soe_df = soe_df.unstack(level='time')
soe_df.columns = soe_df.columns.get_level_values(1)
soe_df.to_csv(f"{save_path}/soe_df.csv")

soc_df = soe_df.apply(lambda row:row/EVCAP_dict[row.name]*100,axis=1)
plt.figure(figsize=(10,10))
for person,profile in soc_df.iterrows():
    plt.plot(profile)
plt.show()
soc_df.to_csv(f"{save_path}/SOC_result.csv")

dayendsoe_df = pd.DataFrame(index=SOE_dayend_dict.keys())
dayendsoe_df['planned'] = SOE_dayend_dict.values()
dayendsoe_df['shifted'] = soe_df[monitor_hr-1]
dayendsoe_df['cap'] = EVCAP_dict.values()
dayendsoe_df['plannedsoc'] = dayendsoe_df.planned/dayendsoe_df.cap*100
dayendsoe_df['shiftedsoc'] = dayendsoe_df.shifted/dayendsoe_df.cap*100
dayendsoe_df.to_csv(f"{path}/dayendsoe_df.csv")

plt.figure(figsize=(10,10))
plt.scatter(dayendsoe_df.plannedsoc,dayendsoe_df.shiftedsoc)
plt.xlabel("SOC dayend planned")
plt.ylabel("SOC dayend shifted")
plt.title("Period End SOC")
plt.tight_layout()
plt.savefig(f'{save_path}/period_end_soc.png')

bins=range(-110,int((max(dayendsoe_df.shiftedsoc-dayendsoe_df.plannedsoc)//10+1)*10),10)
# plt.hist(dayendsoe_df.shiftedsoc-dayendsoe_df.plannedsoc,density=True)
data = dayendsoe_df['shiftedsoc'] - dayendsoe_df['plannedsoc']
total_count = len(data)
plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(data, bins, weights=(100 / total_count) * np.ones_like(data), alpha=0.6, color='b')

plt.xlabel('Shifted End SoC - Planned End SoC [%]')
plt.ylabel('Percentage [%]')
plt.title('Percentage of End SoC Differences')
plt.tight_layout()
plt.savefig(f'{save_path}/SOC_change_distribution.jpg')


EVGEN_data = {
    (park, t): pmo.value(-m.Grids.EVGENP[park, ev, arr_idx, end_idx, bus, t])
    for park, ev, arr_idx, end_idx, bus, t in m.Grids.EVGENP
}
EVGEN_df = pd.DataFrame.from_dict(EVGEN_data, orient='index', columns=['EVGEN'])
EVGEN_df.index = pd.MultiIndex.from_tuples(EVGEN_df.index,names=['park','time'])
EVGEN_df = EVGEN_df.unstack(level='park')
EVGEN_df.columns = EVGEN_df.columns.get_level_values(1)
EVGEN_df.to_csv(f"{save_path}/EVGEN_result.csv")

EVP_SHIFT_ABS_data = {
    (park, t): pmo.value(m.Cars.EVP_SHIFT_ABS[park, ev, arr_idx, end_idx, bus, t])
    for park, ev, arr_idx, end_idx, bus, t in m.Cars.EVP_SHIFT_ABS
}
EVP_SHIFT_ABS_df = pd.DataFrame.from_dict(EVP_SHIFT_ABS_data, orient='index',columns=['shift'])
EVP_SHIFT_ABS_df.index = pd.MultiIndex.from_tuples(EVP_SHIFT_ABS_df.index,names=['park','hour'])
EVP_SHIFT_ABS_df = EVP_SHIFT_ABS_df.unstack(level='park')
EVP_SHIFT_ABS_df.columns = EVP_SHIFT_ABS_df.columns.get_level_values(1)
EVP_SHIFT_ABS_df.to_csv(f"{save_path}/EV_shift.csv")

EVPLAN_data = {(park,t):pmo.value(m.Cars.EVPPLAN[park,person,arr_idx,end_idx,bus,t]) for (park,person,arr_idx,end_idx,bus) in m.PARK for t in m.T}
EVPLAN_data_df = pd.DataFrame.from_dict(EVPLAN_data, orient='index',columns=['plan'])
EVPLAN_data_df.index = pd.MultiIndex.from_tuples(EVPLAN_data_df.index,names=['park','hour'])
EVPLAN_data_df = EVPLAN_data_df.unstack(level='park')
EVPLAN_data_df.columns = EVPLAN_data_df.columns.get_level_values(1)
participation_percent = (abs(EVPLAN_data_df-EVGEN_df)>1e-10).sum(axis=1)/ev_load_profile.person.nunique()*100
plt.figure(figsize=(10,6))
plt.plot(participation_percent)
plt.xlabel("Hour")
plt.xticks(ticks=range(len(hourly_labels)), labels=hourly_labels, rotation=90)
plt.ylabel("EV Flexibility Participation Rate")
plt.title("Hourly Plugged-in Participation Rate")
plt.tight_layout()
plt.savefig(f"{save_path}/participation_rate.jpg")

PBRANCH_data = {
    ((i, j), t): pmo.value(m.Grids.PBRANCH[i, j, t])
    for (i, j) in m.BRANCH
    for t in m.T
}
PBRANCH_df = pd.DataFrame.from_dict(PBRANCH_data, orient='index')
PBRANCH_df.index = pd.MultiIndex.from_tuples(PBRANCH_df.index,names=['BRANCH','time'])
PBRANCH_df = PBRANCH_df.unstack(level='BRANCH')
PBRANCH_df.columns = PBRANCH_df.columns.get_level_values(1)

QBRANCH_data = {
    ((i, j), t): pmo.value(m.Grids.QBRANCH[i, j, t])
    for (i, j) in m.BRANCH
    for t in m.T
}
QBRANCH_df = pd.DataFrame.from_dict(QBRANCH_data, orient='index')
QBRANCH_df.index = pd.MultiIndex.from_tuples(QBRANCH_df.index,names=['BRANCH','time'])
QBRANCH_df = QBRANCH_df.unstack(level='BRANCH')
QBRANCH_df.columns =QBRANCH_df.columns.get_level_values(1)

PVGENP_data = {(pv,t):pmo.value(m.Grids.PVGENP[pv,bus,t]) for (pv,bus) in m.PV for t in m.T}
PVGENP_df = pd.DataFrame.from_dict(PVGENP_data, orient='index', columns=['PVGEN'])
PVGENP_df.index = pd.MultiIndex.from_tuples(PVGENP_df.index,names=['pv','time'])
PVGENP_df = PVGENP_df.unstack(level='pv')
PVGENP_df.columns = PVGENP_df.columns.get_level_values(1)
PVGENQ_data = {(pv,t):pmo.value(m.Grids.PVGENQ[pv,bus,t]) for (pv,bus) in m.PV for t in m.T}
PVGENQ_df = pd.DataFrame.from_dict(PVGENQ_data, orient='index', columns=['PVGEN'])
PVGENQ_df.index = pd.MultiIndex.from_tuples(PVGENQ_df.index,names=['pv','time'])
PVGENQ_df = PVGENQ_df.unstack(level='pv')
PVGENQ_df.columns = PVGENQ_df.columns.get_level_values(1)
PVGENMAX = pd.DataFrame(pv_load_profile.pv_P_daily.tolist()).T
PVGEN_df = PVGENP_df.where(PVGENP_df <= PVGENMAX, PVGENMAX).clip(lower=0)
PVGENQ_df = PVGENQ_df.where((PVGENP_df <= PVGENMAX) & (PVGENP_df >= 0), 0)


PVRATIO = (PVGEN_df/PVGENMAX)*100
PVGEN_df.to_csv(f"{save_path}/PVGENP_result.csv")
PVGENQ_df.to_csv(f"{save_path}/PVGENQ_result.csv")
PVRATIO.to_csv(f"{save_path}/PV_ratio.csv")

plt.figure(figsize=(10,10))
plt.boxplot(PVRATIO.T)
plt.xlabel("Hour")
plt.ylabel("PV Active Power Injection Ratio [%]")
plt.title("PV Active Power Injection Ratio")
plt.xticks(ticks=range(1,len(hourly_labels)+1), labels=hourly_labels, rotation=90)
plt.tight_layout()
plt.savefig(f'{path}/PV_ratio.png')

pf_df = PVGEN_df/np.sqrt(PVGEN_df**2+PVGENQ_df**2)
plt.figure(figsize=(10,10))
plt.boxplot(pf_df.T)
plt.xlabel("Hour")
plt.ylabel("PV power factor optimized")
plt.title("PV Power factor distribution")
plt.xticks(ticks=range(1,len(hourly_labels)+1), labels=hourly_labels, rotation=90)
plt.tight_layout()
plt.savefig(f'{path}/PV_pf.png')


vm_pu = pd.DataFrame()
line_loading = pd.DataFrame()
trafo_loading = pd.DataFrame()
for t in range(monitor_hr):
    net.load.loc[net.load.category == 'base', 'p_mw'] = base_p.loc[t].values
    net.load.loc[net.load.category=='base','q_mvar'] = net.load.loc[net.load.category == 'base'].p_mw*np.tan(np.arccos(0.97))
    net.load.loc[net.load.category == 'hp', 'p_mw'] = hp_p.loc[t].values
    net.load.loc[net.load.category == 'hp', 'q_mvar'] = net.load.loc[net.load.category == 'hp'].p_mw*np.tan(np.arccos(0.97))
    net.sgen.loc[net.sgen.type=='PV','p_mw'] = PVGENP_df.loc[t].values
    net.sgen.loc[net.sgen.type=='PV','q_mvar'] = PVGENQ_df.loc[t].values
    net.storage.loc[net.storage.type=='ev','p_mw'] = EVGEN_df.loc[t].values
    pp.runpp(net)
    pp.to_pickle(net,filename=f'{save_path}/{folder}_hour_{t}_acopf_res.p')
    vm_pu[t] = net.res_bus.vm_pu
    line_loading[t] = net.res_line.loading_percent
    trafo_loading[t] = net.res_trafo.loading_percent

plt.figure(figsize=(10*monitor_hr//24, 6))   
plt.boxplot(vm_pu)
plt.xlabel('Time')
plt.ylabel('Voltage Magnitude [%]')
plt.xticks(ticks=range(1,len(hourly_labels)+1), labels=hourly_labels, rotation=90)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.title('vm_pu_results')
plt.tight_layout()
plt.savefig(f'{save_path}/{folder}_vmpu_res_changed.jpg')
vm_pu.to_csv(f'{save_path}/{folder}_vmpu_res_changed.csv')

plt.figure(figsize=(10*monitor_hr//24, 6))
plt.boxplot(trafo_loading)
plt.xlabel('Time')
plt.ylabel('Loading Percentage [%]')
plt.xticks(ticks=range(1,len(hourly_labels)+1), labels=hourly_labels, rotation=90)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.title('trafo_loading_results')
plt.tight_layout()
plt.savefig(f'{save_path}/{folder}_trafoloading_res.jpg')
trafo_loading.to_csv(f'{save_path}/{folder}_trafo_loading_res_changed.csv')

plt.figure(figsize=(10*monitor_hr//24, 6))
plt.boxplot(line_loading)
plt.xlabel('Time')
plt.ylabel('Loading Percentage [%]')
plt.xticks(ticks=range(1,len(hourly_labels)+1), labels=hourly_labels, rotation=90)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.title('line_loading_results')
plt.tight_layout()
plt.savefig(f'{save_path}/{folder}_lineloading_res.jpg')
line_loading.to_csv(f'{save_path}/{folder}_line_loading_res_changed.csv')