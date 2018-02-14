import pandas as pd
import root_pandas as rpd
import dateutil.parser
from fact.io import read_h5py
from fact.analysis import (
    li_ma_significance,
    split_on_off_source_independent,
    split_on_off_source_dependent,
)
import click
import numpy as np
from fact.instrument import camera_distance_mm_to_deg
import h5py
import sys

data_columns = [
    'night', 'run_id', 'event_num','unix_time_utc'
]
output_columns = [
    'zd_tracking',
    'gamma_energy_prediction',
    'gamma_prediction',
    'width',
    'length',
    'size',
    'm3_trans',
    'm3_long',
    'conc_core',
    'm3l',
    'm3t',
    'concentration_one_pixel',
    'concentration_two_pixel',
    'leakage',
    'leakage2',
    'conc_cog',
    'num_islands',
    'num_pixel_in_shower',
    'ph_charge_shower_mean',
    'ph_charge_shower_variance',
    'ph_charge_shower_max',
]
mc_columns = ['corsika_evt_header_total_energy']
other_columns = ['theta_deg'] + ['theta_deg_off_{}'.format(i) for i in range(1, 6)]
bg_prediction_columns = ['gamma_prediction_off_{}'.format(i) for i in range(1, 6)]


@click.command()
@click.argument('data_path')
@click.argument('gamma_path')
@click.argument('corsika_path')
@click.argument('config_template')
@click.argument('output_base')
@click.option('--threshold', type=float, help='prediction threshold', default=0.7)
@click.option(
    '--theta2-cut', type=float, help='cut for theta^2 in degree^2', default=0.1
)
@click.option(
    '--gamma-fraction', type=float, default=0.5,
    help='Fraction of the simulated gammas used for unfolding')
@click.option('--title', type=str, help='Title of the unfolding project', default="Crab")
@click.option('--start', '-s', help='First night to get data from')
@click.option('--end', '-e', help='Last night to get data from')
@click.option('--zd-min', type=float, default=None,
    help='Minimum Zd angle to unfold')
@click.option('--zd-max', type=float, default=None,
    help='Maximum Zd angle to unfold')
def main(
        data_path,
        gamma_path,
        corsika_path,
        config_template,
        output_base,
        threshold,
        theta2_cut,
        gamma_fraction,
        title,
        start,
        end,
        zd_min,
        zd_max
        ):

    with h5py.File(data_path, 'r') as f:
        source_dependent = 'gamma_prediction_off_1' in f['events'].keys()

    if source_dependent:
        other_columns.extend(bg_prediction_columns)
        theta_cut = np.inf
        theta2_cut = np.inf
        print('Source dependent separation, ignoring theta cut')

    theta_cut = np.sqrt(theta2_cut)

    data = read_h5py(
        data_path,
        key='events',
        columns=data_columns + output_columns + other_columns
    )


    gammas = read_h5py(
        gamma_path,
        key='events',
        columns=mc_columns + output_columns + other_columns,
    )
    gammas.rename(
        columns={'corsika_evt_header_total_energy': 'true_energy'},
        inplace=True,
    )

    runs = read_h5py(data_path, key='runs')

    data['timestamp'] = pd.to_datetime(
        data['unix_time_utc_0'] * 1e6 + data['unix_time_utc_1'],
        unit='us',
    )

    if start:
        data = data.query('timestamp >= @start')
        runs = runs.query('run_start >= @start')
    if end:
        data = data.query('timestamp <= @end')
        runs = runs.query('run_start <= @end')
        
    min_zenith = runs.zenith.min()
    max_zenith = runs.zenith.max()
    
    if zd_min:
        min_zenith = max(min_zenith, zd_min)

    if zd_max:
        max_zenith = min(max_zenith, zd_max)
        
    print('Zenith range of the input data:', min_zenith, max_zenith)

    if source_dependent:
        on_data, off_data = split_on_off_source_dependent(data, threshold)
        on_gammas = gammas.query('gamma_prediction >= {}'.format(threshold))
    else:
        on_data, off_data = split_on_off_source_independent(
            data.query('gamma_prediction >= {}'.format(threshold)),
            theta2_cut=theta2_cut,
        )
        on_gammas = gammas.query(
            '(theta_deg <= {}) & (gamma_prediction >= {})'.format(
                theta_cut, threshold,
            )
        )

    query = '(zd_tracking >= {}) and (zd_tracking <= {})'.format(min_zenith, max_zenith)
    on_gammas = on_gammas.query(query).copy()

    output_columns.append('theta_deg')
    on_gammas = on_gammas.loc[:, output_columns + ['true_energy']]
    on_data = on_data.loc[:, output_columns + data_columns]
    off_data = off_data.loc[:, output_columns + data_columns]

    off_data['weight'] = 0.2
    on_data['weight'] = 1.0
    on_gammas['weight'] = 1.0

    rpd.to_root(on_data, output_base + '_on.root', key='events')
    rpd.to_root(off_data, output_base + '_off.root', key='events')
    rpd.to_root(on_gammas, output_base + '_mc.root', key='events')

    print('N_on: {}'.format(len(on_data)))
    print('N_off: {}'.format(len(off_data)))
    print('S(Li&Ma): {}'.format(li_ma_significance(len(on_data), len(off_data), 0.2)))
    print('N_mc: {}'.format(len(on_gammas)))

    n_excess = len(on_data) - 0.2 * len(off_data)
    fraction = n_excess / len(on_gammas)

    print('N_excess:', n_excess)
    print('Fraction: {:1.4f}'.format(fraction))

    with open(config_template) as f:
        template = f.read()

    t_obs = runs.ontime.sum()

    try:
        corsika = pd.read_hdf(corsika_path, key='table')
    except KeyError:
        f = h5py.File(corsika_path)
        print("given key not in file: possible keys are: {}".format(list(f.keys())))
        return

    corsika['zenith'] = np.rad2deg(corsika['zenith'])
    corsika = corsika.query('(zenith >= {}) and (zenith <= {})'.format(
    min_zenith, max_zenith
    ))
    print('Simulated events after zenith cut: {}'.format(len(corsika)))



    config = template.format(
        t_obs=t_obs,
        selection_fraction=gamma_fraction,
        n_gamma=len(corsika),
        source_file_on=output_base + '_on.root',
        source_file_off=output_base + '_off.root',
        source_file_mc=output_base + '_mc.root',
        tree_name='events',
        output_file=output_base + '_result.root',
        fraction=fraction,
        min_zenith=min_zenith,
        max_zenith=max_zenith,
        title=title,
    )

    with open(output_base + '.config', 'w') as f:
        f.write(config)


if __name__ == '__main__':
    main()
