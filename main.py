import sys
import subprocess
import time
import os
import shutil

PROBLEM = 'leader'

MAX_TEMPLATE_INCREASE = 3

if __name__ == '__main__':
    if len(sys.argv) == 2:
        PROBLEM = sys.argv[1]

    success = False
    counterexample_count, invariant_count = 0, 0
    simulation_time, enumeration_time, refinement_time = 0, 0, 0
    if not os.path.exists('src-c/runtime'):
        os.mkdir('src-c/runtime')
    c_runtime_path = 'src-c/runtime/' + PROBLEM
    shutil.rmtree(c_runtime_path, ignore_errors=True)
    os.mkdir(c_runtime_path)
    output_path = 'outputs/{}'.format(PROBLEM)
    shutil.rmtree(output_path, ignore_errors=True)
    os.mkdir(output_path)
    for i in range(MAX_TEMPLATE_INCREASE):
        if i > 0:
            print('Re-simulate protocol with larger instances')
        start_time = time.time()
        subprocret = subprocess.run(['python', 'translate.py', PROBLEM, '--num_attempt={}'.format(i)], cwd='src-py/')
        if subprocret.returncode != 0:
            print('translate.py fails to parse and translate the input Ivy file. Please use $ivy_check PROTOCOL.ivy to '
                  'check if the Ivy file has the correct syntax and grammar. If it does, the problem may come from the '
                  'limitation of DistAI, which does not support all Ivy features. Exiting...')
            exit(-1)
        subprocess.run(['python', '{}.py'.format(PROBLEM)], cwd='auto_samplers/', check=True)
        end_time = time.time()
        simulation_time += end_time - start_time
        if i == 0:
            subprocess.run(['./main', PROBLEM], cwd='src-c/')
        else:
            subprocess.run(['./main', PROBLEM, '--max_retry={}'.format(i)], cwd='src-c/')
        with open(c_runtime_path + '/refiner_log.txt', 'r') as refiner_log_file:
            refiner_log_lines = refiner_log_file.readlines()
        for line in refiner_log_lines:
            if line.startswith('Success?'):
                if line[len('Success?') + 1:].strip() == 'Yes':
                    success = True
            elif line.startswith('Counterexamples:'):
                counterexample_count += int(line[len('Counterexamples:') + 1:].strip())
            elif line.startswith('Invariants:'):
                invariant_count = int(line[len('Invariants:') + 1:].strip())
            elif line.startswith('Enumeration time:'):
                enumeration_time += int(line[len('Enumeration time:') + 1:].strip())
            elif line.startswith('Refinement time:'):
                refinement_time += int(line[len('Refinement time:') + 1:].strip())
        if success:
            break
    if success:
        enumeration_time = enumeration_time / 1000  # convert ms to s
        refinement_time = refinement_time / 1000
        total_time = simulation_time + enumeration_time + refinement_time
        print('Counterexamples:', counterexample_count)
        print('Invariants:', invariant_count)
        print('DistInv runtime: {:.3f}s'.format(total_time))
        print('Breakdown:  simulation {:.3f}s  learning {:.3f}s  refinement {:.3f}s'.format(simulation_time, enumeration_time, refinement_time))
