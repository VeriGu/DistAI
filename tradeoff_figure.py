import sys
import subprocess
import time
import os
import shutil
import matplotlib.pyplot as plt

def run_tradeoff():
    PROBLEM = 'consensus'

    # sample_number_list = [2000, 4000]
    sample_number_list = [1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000]
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
    sample_number_list = [1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000]

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(sample_number_list, simulation_time_list, '-o', markersize=8)
    ax.plot(sample_number_list, enumeration_time_list, '-x', markersize=9, mew=2.5)
    ax.plot(sample_number_list, refinement_time_list, '-^', markersize=8)
    plt.tick_params(labelsize=16)
    plt.xticks(sample_number_list, ('', '2K', '', '6K', '', '10K', '', '14K', '', '18K', '', '22K'))
    plt.xlabel('number of samples', fontsize=18)
    plt.ylabel('runtime (s)', fontsize=18)
    plt.yscale('log')
    plt.legend(['sampling', 'enumeration', 'refinement'], loc='upper right',
               fontsize=14)  # , handlelength=12.25, markerscale=10.125)
    plt.tight_layout()
    fig.savefig('tradeoff.pdf')
    plt.close(fig)
