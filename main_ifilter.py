import sys
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from time import time
from ifilter import iFilter
from simulator import (
    Simulator,
    NUM_SCAN,
    NUM_TARGET,
    TARGET_ENTER,
    TARGET_EXIT,
    EXPECTED_FALSE_TARGETS,
)


SEED = 336
GRID_SIZE = 5
FONTSIZE_TITLE = 17
FONTSIZE_SUBTITLE = 15
FIGURE_SIZE = (18, 16)


def plot_results(sts_all, imap, n_targets, n_clutter, info_all, fname, title):
    fig = plt.figure(figsize=FIGURE_SIZE)
    ss = [y for x in sts_all if x is not None for y in x]
    xs = [s[0] for s in ss]
    ys = [s[1] for s in ss]
    ax = plt.subplot(2, 2, 1)
    ax.tick_params(bottom=False, top=False, left=False, right=False)  # ticks on axes
    ax.scatter(xs, ys, color='green', s=15)
    ax.set_xlim((-800, 800))
    ax.set_ylim((-800, 800))
    ax.set_title('Ground truth tracks', fontsize=FONTSIZE_SUBTITLE)

    ax = plt.subplot(2, 2, 2)
    ax.imshow(imap, cmap='GnBu', origin='lower')
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(bottom=False, top=False, left=False, right=False)
    ax.set_xticks(np.arange(0, 1600 // GRID_SIZE + 1, 200 // GRID_SIZE), np.arange(-800, 801, 200))
    ax.set_yticks(np.arange(0, 1600 // GRID_SIZE + 1, 200 // GRID_SIZE), np.arange(-800, 801, 200))
    ax.set_title('Estimated densities', fontsize=FONTSIZE_SUBTITLE)

    ax = plt.subplot(2, 2, 3)
    ax.tick_params(bottom=True, top=False, left=True, right=False, labeltop=False, labelright=False)
    xs = list(range(NUM_SCAN))
    ys = [info['TRUE_TARGETS'] for info in info_all]
    xticks = list(range(0, NUM_SCAN+1, 50 if NUM_SCAN % 50 == 0 else 10))
    ax.plot(np.arange(len(n_targets)), n_targets, label='Estimate', color='orange')
    ax.plot(xs, ys, label='Ground truth', color='black')
    ax.set_xlim((0, NUM_SCAN))
    ax.set_ylim((0, 4.5))
    ax.set_xticks(xticks, xticks)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='best')
    ax.set_title('Estimating number of true targets', fontsize=FONTSIZE_SUBTITLE)

    ax = plt.subplot(2, 2, 4)
    ax.tick_params(bottom=True, top=False, left=True, right=False, labeltop=False, labelright=False)
    ax.plot(np.arange(len(n_clutter)), n_clutter, label='Estimate', color='orange')
    # ax.scatter(np.arange(len(n_clutter)), [info['FALSE_TARGETS'] for info in info_all], label='Ground truth', color='green')
    ax.plot([0, NUM_SCAN-1], [EXPECTED_FALSE_TARGETS, EXPECTED_FALSE_TARGETS], label='Expected', color='black')
    ax.set_xlim(0, NUM_SCAN)
    ax.set_ylim(20, 45)
    ax.set_xticks(xticks, xticks)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='best')
    ax.set_title('Estimating number of false targets (clutter)', fontsize=FONTSIZE_SUBTITLE)

    plt.suptitle(title, fontsize=FONTSIZE_TITLE)
    # plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')

def main():
    t0 = time()
    sim = Simulator(SEED)
    sts_all, obs_all, info_all = sim.simulate()

    print(f'\nTotal number of observations: {len(obs_all)}\n')

    ifilter = iFilter(rng=sim.rng)
    sup_imap = None
    n_targets = []
    n_clutter = []

    for m in range(len(obs_all)):
        print(f'\n------ iFilter iter {m+1} ------')
        obs = obs_all[m]
        sts = sts_all[m]
        if obs is None:
            print('No observation received, skip this iteration')
            continue

        print('Transition update ...')
        t1 = time()
        ifilter.transition_update(rng=sim.rng)
        print(f'Done ({time() - t1:.1f} seconds).')

        print('Observation update ...')
        t2 = time()
        ifilter.observation_update(obs, rng=sim.rng)
        print(f'Done ({time() - t2:.1f} seconds).')

        nt = ifilter.expected_true_targets()
        nc = ifilter.expected_false_targets()
        n_targets.append(nt)
        n_clutter.append(nc)

        print(f'True targets (G): {info_all[m]["TRUE_TARGETS"]}')
        print(f'True targets (O): {info_all[m]["TRUE_TARGETS_OBS"]}')
        print(f'True targets (D): {nt:.2f}')
        print(f'False targets (G): {info_all[m]["FALSE_TARGETS"]}')
        print(f'False targets (O): {info_all[m]["FALSE_TARGETS_OBS"]}')
        print(f'False targets (D): {nc:.2f}')

        imap = ifilter.get_intensity_map(grid_size=GRID_SIZE)
        if sup_imap is None:
            sup_imap = imap
        else:
            sup_imap += imap

        ifilter.resample_particles(rng=sim.rng)

    fname = f'results_{SEED}_{NUM_TARGET}_{round(ifilter.N0 / 1000)}k.png'
    title = f'Particles: {ifilter.N1} + {ifilter.N2}'
    plot_results(sts_all, sup_imap, n_targets, n_clutter, info_all, fname, title)

    tt = time() - t0
    hh = tt // 3600
    mm = (tt % 3600) // 60
    ss = tt % 60
    print(f'\nTotal time: {hh:g} hours {mm:g} minutes {ss:.1f} seconds')

if __name__ == '__main__':
    main()

