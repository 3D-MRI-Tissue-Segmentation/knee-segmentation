
# Options for command line input
import sys
import argparse
import os


# inspiration from https://github.com/neoamos/3d-pix2pix-CycleGAN/blob/master/options/base_options.py
class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('-dir', '--dataroot', type=str, default='./', help='Path to input folderz with ur 3d numpies :3')
        self.parser.add_argument('-dirL', '--dataroot_left', type=str, default='', help='Becomes plot on left. Path to input folderz with ur 3d numpies :3')
        self.parser.add_argument('-dirR', '--dataroot_right', type=str, default='', help='Becomes plot on right. Path to input folderz with ur 3d numpies :3')
        
        self.parser.add_argument('-f', '--file_name', type=str, default='', help='Name of file you wanna visualize')
        self.parser.add_argument('-out', '--output_dir', type=str, default="voxel_graph", help='Folder name you want as the directory to export to (don\'t include / in front of it)')
        self.parser.add_argument('-out_name', '--output_html_pathname', type=str, default='segmentation_plotly.html', help='Name of output file pathname; must end with html')
        self.parser.add_argument('-shuff', '--shuffle_colors', type=bool, default=True, help='Shuffle the list of colors corresponding to each segmentation class (copy paste in the output if you prefer it)')
        self.parser.add_argument('-toy', '--toy_dataset', type=int, default=False, help='To test voxel graphics with random voxel cube, input cube dimension (1 int)')
        self.parser.add_argument('-slide', '--slider_interval', type=int, default=1, help='Sampling interval for number of input files. Only set if loading more than one file.')
        # self.parser.add_argument('-t', '--test', default='hullo', help="testestets")

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        # check that only a folder or a file were specified
        if self.opt.file_name:
            assert self.opt.dataroot=='./', 'Please input either a filename or a folder with the files you wish to load in (or take default option)'    

        if self.opt.dataroot_left or self.opt.dataroot_right:
            assert self.opt.dataroot_left, 'If specifying directory for plot on right, also specify directory for plot on left'    
            assert self.opt.dataroot_right, 'If specifying directory for plot on left, also specify directory for plot on right'    

        # Output dir
        # TODO: set whatever you wanna output
        self.opt.output_dir = os.path.join(self.opt.dataroot, self.opt.output_dir)
        if not os.path.exists(self.opt.output_dir):
            os.mkdir(self.opt.output_dir)

        return self.opt