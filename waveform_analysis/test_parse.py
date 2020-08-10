import numpy as np
import argparse
import sys

""" ---------------------------- PARSER ---------------------------- """
parser = argparse.ArgumentParser(description='Create waveform basis file for given detector.')
parser.add_argument('-e','--exp', type=str, help='Experiment name')
parser.add_argument('-r','--run', type=int, help='Run to be analyzed')
parser.add_argument('--roi', type=int, nargs='*', const=None, help='ROI')
# parser.add_argument('--roi', type=tuple, const=None, help='ROI')


args = parser.parse_args()
# for key in args.keys():
#     print('{}: {}'.format(key, args[key]))

print(args)
    