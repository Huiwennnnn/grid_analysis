import os
import pypsa
import pypsa_create_basic_network as basic
import pypsa_pf_stat as psastat
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

"""
Basic Settings for Data Input and Output
"""
MV_feeder=True
grid = "369_0"
folder = '39-1_0_4'
scenario_year = 2050
weekday = "Friday"
day_start_ts = pd.to_datetime(f"{scenario_year}-01-07 00:00:00")
day_start = day_start_ts.day
day_end_ts = pd.to_datetime(f"{scenario_year}-01-08 00:00:00")
month = day_start_ts.month
monitor_hr = int((day_end_ts - day_start_ts).total_seconds()/3600)
path_controlled = f"{grid}/{scenario_year}_{weekday}_01_07_controlled"
path_uncontrolled = f"{grid}/{scenario_year}_{weekday}_01_07_uncontrolled"
lv_pf = 0.97
mv_pf = 0.9
experiment = 'compare_control'

if MV_feeder:
    os.makedirs(f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/mobility/{path_controlled}/{experiment}", exist_ok=True)
    save_path = f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/mobility/{path_controlled}/{experiment}"
else:
    os.makedirs(f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/mobility/{path_controlled}/{folder}/{experiment}", exist_ok=True)
    save_path=f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/mobility/{path_controlled}/{folder}/{experiment}"

"""
Create Network, Add Profiles, Run Power Flow Calculation 
on both HV level controlled and uncontrolled EV Profiles
"""
net = pypsa.Network()
net.import_from_csv_folder(csv_folder_name=f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/Timeseries/{grid}_pypsa")
# net=basic.build_LV_net("39-1_0_4");
# net.export_to_csv_folder(csv_folder_name=f"")
net_base = basic.add_base_load(net,day_start_ts,monitor_hr);
net_pv = basic.add_pv_profile(net_base,day_start_ts);
net_controlled = basic.add_ev_profile(net_pv.copy(),path_controlled,grid,day_start_ts,monitor_hr,scenario_year)
net_uncontrolled = basic.add_ev_profile(net_pv.copy(),path_uncontrolled,grid,day_start_ts,monitor_hr,scenario_year)
net_controlled.lpf()
net_controlled.pf(use_seed=True)
net_uncontrolled.lpf()
net_uncontrolled.pf(use_seed=True)


"""
Grid violation statistics
"""

line_controlled = psastat.pf_line_loading(net_controlled)
line_stat_controlled = psastat.pf_line_overloading_stat(net_controlled)
line_uncontrolled = psastat.pf_line_loading(net_uncontrolled)
line_stat_uncontrolled = psastat.pf_line_overloading_stat(net_uncontrolled)

trafo_controlled = psastat.pf_trafo_loading(net_controlled)
trafo_stat_controlled = psastat.pf_trafo_overloading_stat(net_controlled)
trafo_uncontrolled = psastat.pf_trafo_loading(net_uncontrolled)
trafo_stat_uncontrolled = psastat.pf_trafo_overloading_stat(net_uncontrolled)

voltage_controlled = net_controlled.buses_t.v_mag_pu
voltage_stat_controlled = psastat.pf_undervoltage(net_controlled)
voltage_uncontrolled = net_uncontrolled.buses_t.v_mag_pu
voltage_stat_uncontrolled = psastat.pf_undervoltage(net_uncontrolled)

"""
Plot Compare Result
"""
# Line Violation
fig,ax=plt.subplots(figsize=(10,6))
n = 24
# Positions for the first set of box plots
positions1 = np.arange(1, n + 1)
positions2 = positions1 + 0.4  # Adjust the shift as needed

box1 = ax.boxplot(line_controlled.T.values, vert=True, positions=positions1, patch_artist=True, widths=0.3,whis=[0,100])
box2 = ax.boxplot(line_uncontrolled.T.values, vert=True, positions=positions2, patch_artist=True, widths=0.3, whis=[0,100])
# Set the title and labels
ax.set_title("Line Loading Percentage")
ax.set_ylim([-5, max(line_controlled.max().max(), line_uncontrolled.max().max())])
ax.set_xlabel("Hour")
ax.set_ylabel("Loading Percentage [%]")

# Set x-ticks and x-tick labels to the original row indices, centered between the two sets of boxes
ax.set_xticks(positions1 + 0.2)
ax.set_xticklabels(line_controlled.index, rotation=90)

# Optional: Color the box plots differently for better distinction
colors = ['blue', 'orange']
for bplot, color in zip([box1, box2], colors):
    for patch in bplot['boxes']:
        patch.set_facecolor(color)

# Create custom legend handles
legend_labels = ['Controlled', 'Uncontrolled']
handles = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]
# Add legend to the plot
ax.legend(handles, legend_labels)
plt.tight_layout()
plt.savefig(f"{save_path}/line_loading_distribution.png")
plt.clf()

fig,ax=plt.subplots(figsize=(10,6))
ax.plot(line_stat_controlled,label='controlled')
ax.plot(line_stat_uncontrolled,label='uncontrolled')
ax.set_title("Line Hourly OverLoading Percentage")
ax.set_ylim([0.6*min(line_stat_controlled.min(),line_stat_uncontrolled.min()),1.2*max(line_stat_controlled.max(),line_stat_uncontrolled.max())])
ax.set_xlabel("Hour")
ax.set_ylabel("Loading Percentage [%]")
ax.tick_params(axis='x', rotation=90)

plt.legend()
plt.tight_layout()
plt.savefig(f"{save_path}/hourly_line_overloading.png")
plt.clf()


# Transformer Violation
fig,ax=plt.subplots(figsize=(10,6))
n = 24
# Positions for the first set of box plots
positions1 = np.arange(1, n + 1)
positions2 = positions1 + 0.4  # Adjust the shift as needed

box1 = ax.boxplot(trafo_controlled.T.values, vert=True, positions=positions1, patch_artist=True, widths=0.3,whis=[0,100])
box2 = ax.boxplot(trafo_uncontrolled.T.values, vert=True, positions=positions2, patch_artist=True, widths=0.3, whis=[0,100])
# Set the title and labels
ax.set_title("Transformer Loading Percentage")
ax.set_ylim([-5, max(trafo_controlled.max().max(), trafo_uncontrolled.max().max())])
ax.set_xlabel("Hour")
ax.set_ylabel("Loading Percentage [%]")

# Set x-ticks and x-tick labels to the original row indices, centered between the two sets of boxes
ax.set_xticks(positions1 + 0.2)
ax.set_xticklabels(trafo_controlled.index, rotation=90)

# Optional: Color the box plots differently for better distinction
colors = ['blue', 'orange']
for bplot, color in zip([box1, box2], colors):
    for patch in bplot['boxes']:
        patch.set_facecolor(color)

# Create custom legend handles
legend_labels = ['Controlled', 'Uncontrolled']
handles = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]
# Add legend to the plot
ax.legend(handles, legend_labels)
plt.tight_layout()
plt.savefig(f"{save_path}/trafo_loading_distribution.png")
plt.clf()



fig,ax=plt.subplots(figsize=(10,6))
ax.plot(trafo_stat_controlled,label='controlled')
ax.plot(trafo_stat_uncontrolled,label='uncontrolled')
ax.set_title("Transformer Hourly OverLoading Percentage")
ax.set_ylim([0,1.2*max(trafo_stat_controlled.max(),trafo_stat_uncontrolled.max())])
ax.set_xlabel("Hour")
ax.set_ylabel("Loading Percentage [%]")
ax.tick_params(axis='x', rotation=90)

plt.legend()
plt.tight_layout()
plt.savefig(f"{save_path}/hourly_trafo_overloading.png")
plt.clf()

# Bus Voltage Magnitude

fig,ax=plt.subplots(figsize=(10,6))
n = 24
# Positions for the first set of box plots
positions1 = np.arange(1, n + 1)
positions2 = positions1 + 0.4  # Adjust the shift as needed

box1 = ax.boxplot(voltage_controlled.T.values, vert=True, positions=positions1, patch_artist=True, widths=0.3,whis=[0,100])
box2 = ax.boxplot(voltage_uncontrolled.T.values, vert=True, positions=positions2, patch_artist=True, widths=0.3, whis=[0,100])
# Set the title and labels
ax.set_title("Bus Voltage Magnitude")
ax.set_ylim([0.98*min(voltage_controlled.min().min(), voltage_uncontrolled.min().min()),1.02* max(voltage_controlled.max().max(), voltage_uncontrolled.max().max())])
ax.set_xlabel("Hour")
ax.set_ylabel("Bus Voltage Magnitude [pu]")

# Set x-ticks and x-tick labels to the original row indices, centered between the two sets of boxes
ax.set_xticks(positions1 + 0.2)
ax.set_xticklabels(voltage_controlled.index, rotation=90)

# Optional: Color the box plots differently for better distinction
colors = ['blue', 'orange']
for bplot, color in zip([box1, box2], colors):
    for patch in bplot['boxes']:
        patch.set_facecolor(color)
    for whisker in bplot['whiskers']:
        whisker.set_color(color)
    for cap in bplot['caps']:
        cap.set_color(color)
    for median in bplot['medians']:
        median.set_color(color)
    for flier in bplot['fliers']:
        flier.set_markerfacecolor(color)

# Create custom legend handles
legend_labels = ['Controlled', 'Uncontrolled']
handles = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]
# Add legend to the plot
ax.legend(handles, legend_labels)
plt.tight_layout()
plt.savefig(f"{save_path}/bus_voltage_distribution.png")
plt.clf()


fig,ax=plt.subplots(figsize=(10,6))
ax.plot(voltage_stat_controlled,label='controlled')
ax.plot(voltage_stat_uncontrolled,label='uncontrolled')
ax.set_title("Hourly Undervoltage Percentage")
ax.set_ylim([0.9*min(voltage_stat_controlled.min(),voltage_stat_uncontrolled.min()),1.1*max(voltage_stat_controlled.max(),voltage_stat_uncontrolled.max())])
ax.set_xlabel("Hour")
ax.set_ylabel("Undervoltage Percentage [%]")
ax.tick_params(axis='x', rotation=90)

plt.legend()
plt.tight_layout()
plt.savefig(f"{save_path}/hourly_bus_undervoltage.png")
plt.clf()

# Plot Profiles
base_load_idx= net_controlled.loads.loc[net_controlled.loads.index.str.contains('base')].index
ev_load_idx = net_controlled.loads.loc[net_controlled.loads.index.str.contains('ev')].index
pv_gen_idx = net_controlled.generators.loc[net_controlled.generators.index.str.contains('PV')].index
plt.subplots(figsize=(10,6))
plt.plot(net_controlled.loads_t.p_set[base_load_idx].sum(axis=1),label='building base')
plt.plot(net_controlled.loads_t.p_set[ev_load_idx][net_controlled.loads_t.p_set[ev_load_idx]>=0].sum(axis=1),label='ev charge')
plt.plot(-net_controlled.loads_t.p_set[ev_load_idx][net_controlled.loads_t.p_set[ev_load_idx]<0].sum(axis=1),label='ev discharge')
plt.plot(net_controlled.generators_t.p[pv_gen_idx].sum(axis=1),label='PV')
plt.legend()
plt.tight_layout()
plt.savefig(f"{save_path}/controlled_profile.png")
plt.clf()