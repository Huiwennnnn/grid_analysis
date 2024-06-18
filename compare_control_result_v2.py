import os
import imageio.v2 as imageio
import cartopy.crs as ccrs
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

"""
Basic Settings for Data Input and Output
"""
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


MV_feeder = True
grid = "369_0"
# folder = '298-4_1_5'
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
experiment = 'compare_control'


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


def build_and_pf(MV_feeder, folder, grid, day_start_ts, monitor_hr, scenario_year):
    if MV_feeder:
        net = basic.build_MV_LV_net(MV_case_id=grid);
    else:
        net = basic.build_LV_net(case_id=folder);

    net_base = basic.add_base_load(net, day_start_ts, monitor_hr);
    net_pv = basic.add_pv_profile(net_base, day_start_ts);
    net_controlled = basic.add_ev_profile(net_pv.copy(), path_controlled, grid, day_start_ts, monitor_hr, scenario_year)
    net_uncontrolled = basic.add_ev_profile(net_pv.copy(), path_uncontrolled, grid, day_start_ts, monitor_hr,
                                            scenario_year)
    net_controlled.lpf()
    net_controlled.pf(use_seed=True)
    net_uncontrolled.lpf()
    net_uncontrolled.pf(use_seed=True)

    return net_controlled, net_uncontrolled


def plot_profiles(nets, save_paths):
    for i in range(2):
        base_load_idx = nets[i].loads.loc[nets[i].loads.index.str.contains('base')].index
        ev_load_idx = nets[i].loads.loc[nets[i].loads.index.str.contains('ev')].index
        pv_gen_idx = nets[i].generators.loc[nets[i].generators.index.str.contains('PV')].index
        plt.subplots(figsize=(10, 6))
        plt.plot(nets[i].loads_t.p_set[base_load_idx].sum(axis=1), label='building base')
        plt.plot(nets[i].loads_t.p_set[ev_load_idx][nets[i].loads_t.p_set[ev_load_idx] >= 0].sum(axis=1),
                 label='ev charge')
        plt.plot(-nets[i].loads_t.p_set[ev_load_idx][nets[i].loads_t.p_set[ev_load_idx] < 0].sum(axis=1),
                 label='ev discharge')
        plt.plot(nets[i].generators_t.p[pv_gen_idx].sum(axis=1), label='PV')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_paths[i]}/network_profiles.png")
        plt.clf()
    return 0


def create_compare_plots(nets, save_paths):
    line_controlled = psastat.pf_line_loading(nets[0])
    line_stat_controlled = psastat.pf_line_overloading_stat(nets[0])
    line_uncontrolled = psastat.pf_line_loading(nets[1])
    line_stat_uncontrolled = psastat.pf_line_overloading_stat(nets[1])

    trafo_controlled = psastat.pf_trafo_loading(nets[0])
    trafo_stat_controlled = psastat.pf_trafo_overloading_stat(nets[0])
    trafo_uncontrolled = psastat.pf_trafo_loading(nets[1])
    trafo_stat_uncontrolled = psastat.pf_trafo_overloading_stat(nets[1])

    voltage_controlled = nets[0].buses_t.v_mag_pu
    voltage_stat_controlled = psastat.pf_undervoltage(nets[0])
    voltage_uncontrolled = nets[1].buses_t.v_mag_pu
    voltage_stat_uncontrolled = psastat.pf_undervoltage(nets[1])

    """
    Plot Compare Result
    """
    # Line Violation
    fig, ax = plt.subplots(figsize=(10, 6))
    n = 24
    # Positions for the first set of box plots
    positions1 = np.arange(1, n + 1)
    positions2 = positions1 + 0.4  # Adjust the shift as needed

    box1 = ax.boxplot(line_controlled.T.values, vert=True, positions=positions1, patch_artist=True, widths=0.3,
                      whis=[0, 100])
    box2 = ax.boxplot(line_uncontrolled.T.values, vert=True, positions=positions2, patch_artist=True, widths=0.3,
                      whis=[0, 100])
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
    plt.savefig(f"{save_paths[0]}/line_loading_distribution.png")
    plt.clf()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(line_stat_controlled, label='controlled')
    ax.plot(line_stat_uncontrolled, label='uncontrolled')
    ax.set_title("Line Hourly OverLoading Percentage")
    ax.set_ylim([0.6 * min(line_stat_controlled.min(), line_stat_uncontrolled.min()),
                 1.2 * max(line_stat_controlled.max(), line_stat_uncontrolled.max())])
    ax.set_xlabel("Hour")
    ax.set_ylabel("Loading Percentage [%]")
    ax.tick_params(axis='x', rotation=90)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_paths[0]}/hourly_line_overloading.png")
    plt.clf()

    # Transformer Violation
    fig, ax = plt.subplots(figsize=(10, 6))
    n = 24
    # Positions for the first set of box plots
    positions1 = np.arange(1, n + 1)
    positions2 = positions1 + 0.4  # Adjust the shift as needed

    box1 = ax.boxplot(trafo_controlled.T.values, vert=True, positions=positions1, patch_artist=True, widths=0.3,
                      whis=[0, 100])
    box2 = ax.boxplot(trafo_uncontrolled.T.values, vert=True, positions=positions2, patch_artist=True, widths=0.3,
                      whis=[0, 100])
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
    plt.savefig(f"{save_paths[0]}/trafo_loading_distribution.png")
    plt.clf()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(trafo_stat_controlled, label='controlled')
    ax.plot(trafo_stat_uncontrolled, label='uncontrolled')
    ax.set_title("Transformer Hourly OverLoading Percentage")
    ax.set_ylim([0, 1.2 * max(trafo_stat_controlled.max(), trafo_stat_uncontrolled.max())])
    ax.set_xlabel("Hour")
    ax.set_ylabel("Loading Percentage [%]")
    ax.tick_params(axis='x', rotation=90)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_paths[0]}/hourly_trafo_overloading.png")
    plt.clf()

    # Bus Voltage Magnitude
    fig, ax = plt.subplots(figsize=(10, 6))
    n = 24
    # Positions for the first set of box plots
    positions1 = np.arange(1, n + 1)
    positions2 = positions1 + 0.4  # Adjust the shift as needed

    box1 = ax.boxplot(voltage_controlled.T.values, vert=True, positions=positions1, patch_artist=True, widths=0.3,
                      whis=[0, 100])
    box2 = ax.boxplot(voltage_uncontrolled.T.values, vert=True, positions=positions2, patch_artist=True, widths=0.3,
                      whis=[0, 100])
    # Set the title and labels
    ax.set_title("Bus Voltage Magnitude")
    ax.set_ylim([0.98 * min(voltage_controlled.min().min(), voltage_uncontrolled.min().min()),
                 1.02 * max(voltage_controlled.max().max(), voltage_uncontrolled.max().max())])
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
    plt.savefig(f"{save_paths[0]}/bus_voltage_distribution.png")
    plt.clf()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(voltage_stat_controlled, label='controlled')
    ax.plot(voltage_stat_uncontrolled, label='uncontrolled')
    ax.set_title("Hourly Undervoltage Percentage")
    ax.set_ylim([0.9 * min(voltage_stat_controlled.min(), voltage_stat_uncontrolled.min()),
                 1.1 * max(voltage_stat_controlled.max(), voltage_stat_uncontrolled.max())])
    ax.set_xlabel("Hour")
    ax.set_ylabel("Undervoltage Percentage [%]")
    ax.tick_params(axis='x', rotation=90)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_paths[0]}/hourly_bus_undervoltage.png")
    plt.clf()
    return {'line_controlled': line_controlled, 'line_uncontrolled': line_uncontrolled,
            'trafo_controlled': trafo_controlled,
            'trafo_uncontrolled': trafo_uncontrolled, 'voltage_controlled': voltage_controlled,
            'voltage_uncontrolled': voltage_uncontrolled}


def hourly_visualize(MV_feeder,nets, save_paths, loading_res):
    for i in range(2):
        if i == 0:
            line_load = loading_res['line_controlled']
            voltage_load = loading_res['voltage_controlled']
            trafo_load = loading_res['trafo_controlled']
        else:
            line_load = loading_res['line_uncontrolled']
            voltage_load = loading_res['voltage_uncontrolled']
            trafo_load = loading_res['trafo_uncontrolled']

        for sns in nets[i].snapshots[0:24]:
            fig, ax = plt.subplots(subplot_kw={"projection": ccrs.EqualEarth()}, figsize=(20, 20))
            loads = nets[i].loads.assign(l=nets[i].loads_t.p.loc[sns]).groupby(["bus"]).l.sum()
            gens = nets[i].generators.assign(gen=nets[i].generators_t.p.loc[sns]).groupby(
                ["bus"]).gen.sum()
            gens = gens.reindex(index=loads.index, fill_value=0)
            net_load = loads - gens
            line_loading = np.sqrt(nets[i].lines_t.p0.loc[sns] ** 2 + nets[i].lines_t.q0.loc[
                sns] ** 2) / nets[i].lines.s_nom * 100
            trafo_loading = np.sqrt(
                nets[i].transformers_t.p0.loc[sns] ** 2 + nets[i].transformers_t.q0.loc[
                    sns] ** 2) / nets[i].transformers.s_nom * 100
            collection = nets[i].plot(
                bus_sizes=abs(net_load/5e6),  # proportional to net loading
                bus_colors=nets[i].buses_t.v_mag_pu.loc[sns],  # proportional to v_mag_pu
                bus_cmap=plt.cm.jet_r,
                bus_alpha=1,
                margin=0.05,
                line_widths=1,  # net_controlled.lines_t.p0.max().max(),
                line_colors=line_loading,
                line_cmap=plt.cm.jet,
                transformer_colors=trafo_loading,  # 'Black',
                transformer_widths=10,  # net_controlled.transformers_t.p0.loc[sns],
                transformer_alpha=0.5,
                transformer_cmap=plt.cm.jet,
                projection=ccrs.EqualEarth(),
                color_geomap=True,
                title= str(sns) if MV_feeder else str(sns) + "  " + f"{folder}",
                jitter=5
            )

            bussm = plt.cm.ScalarMappable(cmap=plt.cm.jet_r, norm=mcolors.Normalize(vmin=voltage_load.min().min(),
                                                                                    vmax=voltage_load.max().max()))
            bussm.set_array([])
            linesm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=mcolors.Normalize(vmin=line_load.min().min(),
                                                                                   vmax=line_load.max().max()))
            linesm.set_array([])

            cax = fig.add_axes([ax.get_position().x1-0.01, ax.get_position().y0, 0.01, ax.get_position().height])
            cax2 = fig.add_axes([ax.get_position().x1+0.03, ax.get_position().y0, 0.01, ax.get_position().height])
            plt.colorbar(linesm, cax=cax, label="Line Loading [%]")
            plt.colorbar(bussm, cax=cax2, label="Bus v_mag_pu [-]")

            if MV_feeder:
                trafosm = plt.cm.ScalarMappable(cmap=plt.cm.jet,norm=mcolors.Normalize(vmin=trafo_load.min().min(), vmax=trafo_load.max().max()))
                trafosm.set_array([])
                cax3 = fig.add_axes([ax.get_position().x1+0.07, ax.get_position().y0,0.01,ax.get_position().height])
                plt.colorbar(trafosm,cax=cax3,label="Trafo Loading [%]")

            plt.subplots_adjust(right=0.9)
            plt.savefig((f"{save_paths[i]}/hourly_visualize_{sns}.png"))


if __name__ == '__main__':
    if MV_feeder:
        save_paths = make_save_path(MV_feeder, path_controlled, path_uncontrolled, None, experiment)
        nets = build_and_pf(MV_feeder=MV_feeder, folder=None, grid=grid, day_start_ts=day_start_ts,
                            monitor_hr=monitor_hr, scenario_year=scenario_year)
        plot_profiles(nets, save_paths)
        loading_res = create_compare_plots(nets, save_paths)
        hourly_visualize(MV_feeder, nets, save_paths, loading_res)
    else:
        for folder in LV_list:
            save_paths = make_save_path(MV_feeder, path_controlled, path_uncontrolled, folder, experiment)
            nets = build_and_pf(MV_feeder=MV_feeder, folder=folder, grid=grid, day_start_ts=day_start_ts,
                                monitor_hr=monitor_hr, scenario_year=scenario_year)
            plot_profiles(nets, save_paths)
            loading_res = create_compare_plots(nets, save_paths)
            hourly_visualize(MV_feeder,nets, save_paths, loading_res)
