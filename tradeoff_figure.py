import sys
import subprocess
import time
import matplotlib.pyplot as plt

if __name__ == '__main__':
    PROBLEM = 'consensus'
    start_time = time.time()
    success = False

    # sample_number_list = [2000, 4000]
    sample_number_list = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000]
    simulation_time_list = []
    enumeration_time_list = []
    refinement_time_list = []

    for sample_number in sample_number_list:
        print('Protocol simulation will terminate after collecting {} samples'.format(sample_number))
        print('')
        simulation_start_time = time.time()
        subprocess.run(['python', 'translate.py', PROBLEM, '--exact_sample={}'.format(sample_number)], cwd='src-py/')
        subprocess.run(['python', '{}.py'.format(PROBLEM)], cwd='auto_samplers/')
        simulation_end_time = time.time()
        simulation_time = simulation_end_time - simulation_start_time
        subprocess.run(['./main', PROBLEM], cwd='src-c/')
        with open('src-c/refiner_log.txt', 'r') as refiner_log_file:
            refiner_log_lines = refiner_log_file.readlines()
        for line in refiner_log_lines:
            if line.startswith('Success?'):
                if line[len('Success?') + 1:].strip() == 'Yes':
                    success = True
            elif line.startswith('Enumeration time:'):
                enumeration_time = int(line[len('Enumeration time:') + 1:].strip())
            elif line.startswith('Refinement time:'):
                refinement_time = int(line[len('Refinement time:') + 1:].strip())
        assert(success is True)
        enumeration_time = enumeration_time / 1000  # convert ms to s
        refinement_time = refinement_time / 1000
        simulation_time_list.append(simulation_time)
        enumeration_time_list.append(enumeration_time)
        refinement_time_list.append(refinement_time)
        print('Breakdown:  simulation {:.3f}s  learning {:.3f}s  refinement {:.3f}s'.format(simulation_time, enumeration_time, refinement_time))
        print('')

    # simulation_time_list, enumeration_time_list, refinement_time_list = [1, 2], [3, 4], [4, 1]

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(sample_number_list, simulation_time_list, '-o', markersize=8)
    ax.plot(sample_number_list, enumeration_time_list, '-x', markersize=9, mew=2.5)
    ax.plot(sample_number_list, refinement_time_list, '-^', markersize=8)
    plt.tick_params(labelsize=16)
    plt.xticks(sample_number_list, ('2K', '4K', '6K', '8K', '10K', '12K', '14K', '16K', '18K', '20K', '22K'))
    plt.xlabel('number of samples', fontsize=18)
    plt.ylabel('runtime (s)', fontsize=18)
    plt.legend(['sampling', 'learning', 'refinement'], loc='upper right', fontsize=14) # , handlelength=12.25, markerscale=10.125)
    plt.tight_layout()
    fig.savefig('tradeoff.png')
    plt.close(fig)
