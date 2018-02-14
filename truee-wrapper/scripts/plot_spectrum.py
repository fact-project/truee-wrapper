import pandas as pd
import numpy as np
import root_pandas as rpd
import ROOT
import re
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import click
import uncertainties as unc
from fact_plots.plotting import add_preliminary


def magic_crab(energy):
    CRAB_PHI_0 = 3.23e-11
    CRAB_A = -2.47
    CRAB_B = -0.24
    return CRAB_PHI_0 * (energy / 1000) ** (CRAB_A + CRAB_B * np.log10(energy / 1000))


def hegra_crab(energy):
    return 2.79e-11 * (energy / 1000) ** (-2.59)


def hess_crab(energy):
    return 3.76e-11 * (energy / 1000)**(-2.39) * np.exp(- energy / 14.3e3)


def get_truee_results(result_file):

    f = ROOT.TFile(result_file)
    d = f.Get('RealDataResults')
    entry_list = d.GetListOfKeys()
    entries = [e.GetName() for e in entry_list]

    results = {}
    for entry in entries:
        m = re.match('result_bins_(\d+)_knots_(\d+)_degFree_(\d+)', entry)

        if m is None:
            continue

        bins, knots, ndf = map(int, m.groups())

        df = pd.DataFrame(
            columns=[
                'x', 'xerr_low', 'xerr_high',
                'y', 'yerr_low', 'yerr_high'
            ]
        )

        hist = d.Get(entry)

        for i in range(1, hist.GetNbinsX() + 1):
            x = hist.GetBinCenter(i)
            low_edge = hist.GetBinLowEdge(i)
            width = hist.GetBinWidth(i)
            y = hist.GetBinContent(i)
            yerr_low = hist.GetBinErrorLow(i)
            yerr_high = hist.GetBinErrorUp(i)

            df.loc[i] = (
                x, x - low_edge, 0.5 * width,
                y, yerr_low, yerr_high,
            )

            results[(bins, knots, ndf)] = df

    return results


def power(x, norm, gamma):
    return norm * x**(-gamma)


def fit_power_law(x, y, yerr):

    try:
        params, cov = curve_fit(
            power, x, y, p0=[1e9, 2],
            sigma=yerr, absolute_sigma=True
        )
        return params, cov
    except Exception as e:
        print(e)
        return np.full(2, np.nan), np.full((2, 2), np.nan)


@click.command()
@click.argument('truee_result')
@click.option('--compare', help='A second truee file to plot for comparison')
@click.option('-l', '--comparison-label', help='Label for the second truee file')
@click.option('--fit', is_flag=True)
@click.option('--magic', is_flag=True)
@click.option('--hegra', is_flag=True)
@click.option('--hess', is_flag=True)
@click.option('-o', '--outputfile')
@click.option('--csvfile', help='Store Data Points in CSV file', type=str, default=None)
def main(truee_result, compare, comparison_label, fit, magic, hegra, hess, outputfile, csvfile):

    results = get_truee_results(truee_result)

    fig, ax = plt.subplots()

    add_preliminary('lower left', ax=ax)
    dfs = []
    x = np.linspace(2.3, 4.3)
    for (n_bins, knots, ndf), df in results.items():
        print((df.x - df.xerr_low).values)
        dfs.append(df)
        ax.errorbar(
            df.x.values,
            df.y.values,
            xerr=(df.xerr_low.values, df.xerr_high.values),
            yerr=(df.yerr_low.values, df.yerr_high.values),
            linestyle='',
            label='FACT Unfolding (this work)'.format(knots, ndf),
        )
        if fit is True:
            params, cov = fit_power_law(10**df.x, df.y, df.yerr_low)
            phi_0, gamma = unc.correlated_values(params, cov)
            ax.plot(
                x, power(10**x, *params),
                label=r'$\phi_0 = {:L}$, $\gamma={:L}$'.format(phi_0, gamma),
            )

    if compare:
        results_compare = get_truee_results(compare)
        for (n_bins, knots, ndf), df in results_compare.items():
            print((df.x - df.xerr_low).values)

            ax.errorbar(
                df.x.values,
                df.y.values,
                xerr=(df.xerr_low.values, df.xerr_high.values),
                yerr=(df.yerr_low.values, df.yerr_high.values),
                linestyle='',
                label=comparison_label,
            )
            if fit is True:
                params, cov = fit_power_law(10**df.x, df.y, df.yerr_low)
                phi_0, gamma = unc.correlated_values(params, cov)
                ax.plot(
                    x, power(10**x, *params),
                    label=r'$\phi_0 = {:L}$, $\gamma={:L}$'.format(phi_0, gamma),
                )

    if magic is True:
        plt.plot(x, magic_crab(10**x), label='MAGIC, JHEAP 5-6')

    if hegra is True:
        plt.plot(x, hegra_crab(10**x), label='HEGRA, APJ 539-1')

    ax.set_yscale('log')

    ax.set_xlabel(r'$\log_{10}(E \,\,/\,\, \mathrm{GeV})$')
    ax.set_ylabel(r'$\mathrm{d}N / \mathrm{d}E \,\,/\,\, (\mathrm{TeV}^{-1} \mathrm{cm}^{-2} \mathrm{s}^{-1})$')
    ax.legend()

    fig.tight_layout(pad=0)

    if outputfile:
        fig.savefig(outputfile)
    else:
        plt.show()

    if csvfile is not None:
        dfs = pd.concat(dfs)
        dfs.to_csv(csvfile,index=False)


if __name__ == '__main__':
    main()
