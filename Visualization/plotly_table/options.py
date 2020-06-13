# Options for command line input
import sys
import argparse
import os

EXTENSIONS = ['.csv']

def is_acceptable(filename):
    return any(filename.endswith(extension) for extension in EXTENSIONS)


# inspiration from https://github.com/neoamos/3d-pix2pix-CycleGAN/blob/master/options/base_options.py
class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('-in', '--input_data_path', type=str, default='', help='Path name of input csv.')
        self.parser.add_argument('-test', '--test', type=bool, default=False, help='Generates smol csv file to display a dummy table.')
        self.parser.add_argument('-out', '--output_html_pathname', type=str, default='plotly_table.html', help='Name of output html file, must have html ending.')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        # check that the input is of an acceptable format
        if self.opt.input_data_path:
            assert is_acceptable(self.opt.input_data_path), 'Please input a file name of extension type ' + EXTENSIONS[0]    

        if self.opt.test:
            import numpy as np
            import pandas as pd

            dummy_output_path = 'in.csv'
            a = np.random.rand(10,3)
            b = {1: a[:,0], 2: a[:,1], 3: a[:,2]}
            c = pd.DataFrame(data=b)
            c.to_csv(dummy_output_path, index=False)

            self.opt.input_data_path = dummy_output_path


        return self.opt