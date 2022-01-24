#!python

from __future__ import print_function, division
from builtins import map
import numpy as NP
import os
import subprocess
import psutil
import time 
import argparse
from astroutils import writer_module as WM

def monitor_memory(pids, tint=2.0):

    if not isinstance(pids , list):
        raise TypeError('Input PIDs must be specified as a list')
    try:
        pids = list(map(int, pids))
    except ValueError:
        raise ValueError('Input PIDs could not be specified as integers. Check inputs again.')
        
    if not isinstance(tint, (int,float)):
        raise TypeError('Time interval must be a scalar number')
    if tint <= 0.0:
        tint = 60.0

    while True:
        subprocess.call(['clear'])
        with WM.term.location(0, 0):
            print('Resources under PRISim processes...')
        with WM.term.location(0, 1):
            print('{0:>8} {1:>8} {2:>12}'.format('PID', 'CPU [%]', 'Memory [GB]'))
        cpu = NP.zeros(len(pids))
        mem = NP.zeros(len(pids))
        for pi, pid in enumerate(pids):
            proc = psutil.Process(pid)
            cpu[pi] = proc.cpu_percent(interval=0.01) # CPU usage in percent
            cpu[pi] = proc.cpu_percent(interval=0.01) # CPU usage in percent
            mem[pi] = proc.memory_info().rss / 2.0**30 # memory used in GB
            with WM.term.location(0, 2+pi):
                print('{0:8d} {1:8.1f} {2:12.4f}'.format(pid, cpu[pi], mem[pi]))
        with WM.term.location(0, len(pids)+2):
            print('{0:>8} {1:8.1f} {2:12.4f}'.format('Total', NP.sum(cpu), NP.sum(mem)))
        time.sleep(tint)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Program to monitor live memory usage')
    input_group = parser.add_argument_group('Input parameters', 'Input specifications')
    input_group.add_argument('-p', '--pids', dest='pids', type=int, nargs='+', required=True, help='List of PIDs to be monitored')
    input_group.add_argument('-t', '--tint', dest='tint', type=float, default=2, required=False, help='Time interval for update')

    args = vars(parser.parse_args())
    pids = args['pids']
    tint = args['tint']
    monitor_memory(pids, tint)
    
    
