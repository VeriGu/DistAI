import sys
import subprocess
import time
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def run_tradeoff():
    PROBLEM = 'consensus'

    # sample_number_list = [2000, 4000]
    sample_number_list = [100, 2000, 6000, 10000, 14000, 18000, 22000, 26000, 30000, 34000, 38000, 42000, 46000, 50000]
    simulation_time_list = []
    enumeration_time_list = []
    refinement_time_list = []
    if not os.path.exists('src-c/runtime'):
        os.mkdir('src-c/runtime')
    c_runtime_path = 'src-c/runtime/' + PROBLEM
    shutil.rmtree(c_runtime_path, ignore_errors=True)
    os.mkdir(c_runtime_path)

    for sample_number in sample_number_list:
        print('Protocol simulation will terminate after collecting {} samples'.format(sample_number))
        print('')

        simulation_time, enumeration_time, refinement_time = 0, 0, 0
        success = False
        for i in range(3):
            if i > 0:
                print('Re-simulate protocol with larger instances')
            start_time = time.time()
            subprocess.run(['python', 'translate.py', PROBLEM, '--num_attempt={}'.format(i), '--exact_sample={}'.format(sample_number)], cwd='src-py/')
            subprocess.run(['python', '{}.py'.format(PROBLEM)], cwd='auto_samplers/')
            end_time = time.time()
            simulation_time += end_time - start_time
            if i == 0:
                subprocess.run(['./main', PROBLEM], cwd='src-c/')
            else:
                subprocess.run(['./main', PROBLEM, str(i)], cwd='src-c/')
            with open(c_runtime_path + '/refiner_log.txt', 'r') as refiner_log_file:
                refiner_log_lines = refiner_log_file.readlines()
            for line in refiner_log_lines:
                if line.startswith('Success?'):
                    if line[len('Success?') + 1:].strip() == 'Yes':
                        success = True
                elif line.startswith('Enumeration time:'):
                    enumeration_time += int(line[len('Enumeration time:') + 1:].strip())
                elif line.startswith('Refinement time:'):
                    refinement_time += int(line[len('Refinement time:') + 1:].strip())
            if success:
                break

        assert(success is True)
        enumeration_time = enumeration_time / 1000  # convert ms to s
        refinement_time = refinement_time / 1000
        simulation_time_list.append(simulation_time)
        enumeration_time_list.append(enumeration_time)
        refinement_time_list.append(refinement_time)
        print('Breakdown:  simulation {:.3f}s  learning {:.3f}s  refinement {:.3f}s'.format(simulation_time, enumeration_time, refinement_time))
        print('')

    # simulation_time_list, enumeration_time_list, refinement_time_list = [1, 2], [3, 4], [4, 1]

    with open('tradeoff_log.txt', 'w') as outfile:
        outfile.write(', '.join([str(f) for f in simulation_time_list]) + '\n')
        outfile.write(', '.join([str(f) for f in enumeration_time_list]) + '\n')
        outfile.write(', '.join([str(f) for f in refinement_time_list]) + '\n')


if __name__ == '__main__':
    run_tradeoff()
    with open('tradeoff_log.txt', 'r') as infile:
        lines = infile.readlines()
    simulation_time_list = lines[0].strip().split(', ')
    enumeration_time_list = lines[1].strip().split(', ')
    refinement_time_list = lines[2].strip().split(', ')
    simulation_time_list = [float(s) for s in simulation_time_list]
    enumeration_time_list = [float(s) for s in enumeration_time_list]
    refinement_time_list = [float(s) for s in refinement_time_list]
    sample_number_list = [100, 2000, 6000, 10000, 14000, 18000, 22000, 26000, 30000, 34000, 38000, 42000, 46000, 50000]

    fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [1, 3]})
    plt.subplots_adjust(hspace=0)
    ax.plot(sample_number_list, simulation_time_list, '-o', markersize=8)
    ax.plot(sample_number_list, enumeration_time_list, '-x', markersize=9, mew=2.5)
    ax.plot(sample_number_list, refinement_time_list, '-^', markersize=8)
    ax2.plot(sample_number_list, simulation_time_list, '-o', markersize=8)
    ax2.plot(sample_number_list, enumeration_time_list, '-x', markersize=9, mew=2.5)
    ax2.plot(sample_number_list, refinement_time_list, '-^', markersize=8)
    ax.set_ylim(58, 65)  # outliers only
    ax2.set_ylim(-2, 20)  # most of the data
    ax2.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    ax2.set_xlim(xmin=0)
    ax2.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop='off')  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    fig.subplots_adjust(hspace=.2)
    ax.tick_params(labelsize=16)
    ax.set_xticks([])
    ax2.tick_params(labelsize=16)
    plt.xticks(sample_number_list, ('', '2K', '', '10K', '', '18K', '', '26K', '', '34K', '', '42K', '', '50K'))
    plt.xlabel('number of samples', fontsize=18)
    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.ylabel('runtime (s)', fontsize=18)
    # plt.yscale('log')
    # plt.xlim(xmin=0)
    ax.legend(['sampling', 'enumeration', 'refinement'], loc='upper right',
               fontsize=14)  # , handlelength=12.25, markerscale=10.125)
    plt.tight_layout()
    fig.savefig('tradeoff.pdf')
    plt.close(fig)
