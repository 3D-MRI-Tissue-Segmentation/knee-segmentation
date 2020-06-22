
""" Generates voxel plotly graph

Get started with 
python3 run_plotly.py -toy 10

Fig generation slow for noisy, non-sparse data

"""

from options import Options
from data_loader import load_data
from VoxelData import VoxelData
from RenderData import RenderData
from RenderData import get_steps
import chart_studio.plotly
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors
import matplotlib.pyplot as plt
import random
import numpy as np

def get_colors(opt):
    # colors = ["pink", "red", "orange", "yellow", "lightgreen", "lightblue", "purple"]
    colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    if opt.shuffle_colors:
        # Two types of random, one roll one random shuffle
        # colors = np.roll(np.array(colors), np.random.randint(0,np.size(colors)))
        random.shuffle(colors)
    return colors


def make_fig(data, opt, col_num, num_in):
    print("Generating figure")
    
    is_multiple_samples = type(data) == list
    if is_multiple_samples:
        print('There are multiple samples')
        num_samples = len(data)
    else: 
        print('There is 1 sample')
        if opt.dataroot_left and opt.file_name_right:
            num_samples = get_num_samples
        else:
            num_samples = 1

    num_classes_all = []

    for j in range(num_samples):
        
        curr_data = data if not is_multiple_samples else data[j]
        
        Voxels = VoxelData(curr_data)
        # print('np.sum(curr_data)',np.sum(curr_data))
    
        # print("Voxels.data\n",Voxels.data)
        # print("Voxels.vertices\n",Voxels.vertices)
        # print("Voxels.triangles\n",Voxels.triangles)

        # has_background = False
        for i, seg_class in enumerate(Voxels.unique_classes):
            print("Making volume", j, "segmentation voxels of class", (i+1), "/", Voxels.num_classes)
            if seg_class == background_seg:
                # put points so that full graph appears even if empty
                # has_background = True
                fig.add_trace(go.Mesh3d(
                # voxel vertices
                x=[0,Voxels.x_length],
                y=[0,Voxels.y_length],
                z=[0,Voxels.z_length],
                opacity=0,
                showlegend=True
                ), row=1, col=col_num)
                continue

            curr_class = RenderData(Voxels.get_class_voxels(seg_class))
            curr_color = colors if opt.binary_color else colors[int(seg_class)]
            print('For class', int(seg_class), 'curr_color',curr_color)

            fig.add_trace(go.Mesh3d(
                # voxel vertices
                x=curr_class.vertices[0],
                y=curr_class.vertices[1],
                z=curr_class.vertices[2],
                # triangle vertices
                i=curr_class.triangles[0],
                j=curr_class.triangles[1],
                k=curr_class.triangles[2],
                color=curr_color,
                opacity=0.7,
                showlegend=True
                ), row=1, col=col_num)
    
        # print('Appending ',Voxels.num_classes, 'classes')
        num_classes_all.append(Voxels.num_classes)

    return fig, num_classes_all, [num_samples]




def update_fig(fig, num_classes_all, num_samples):
    print('num_classes_all',num_classes_all)
    tot_traces = len(fig.data)
    print('Total traces', tot_traces)
    steps = get_steps(num_samples, tot_traces, num_classes_all) 
    sliders = [dict(
        currentvalue={"prefix": "Vol num: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        showlegend=True,
        sliders=sliders
    )

    return fig




if __name__ == "__main__":
    
    opt = Options().parse()

    # Colors
    colors = get_colors(opt)
    colors = colors[opt.binary_color] if opt.binary_color else colors
    background_seg = 0

    # fig = go.Figure()
    

    if opt.dataroot_left and opt.dataroot_right:
        fig = make_subplots(rows=1, 
                            cols=2,
                            specs=[[{'type': 'mesh3D'}, {'type': 'mesh3D'}]])
        print('type(fig)',type(fig))

        data_l, data_r, num_in_l, num_in_r = load_data(opt)
        col_l = 1
        col_r = 2

        fig, num_classes_all_l, num_samples_l = make_fig(data_l, opt,col_l, num_in_l)
        fig, num_classes_all_r, num_samples_r = make_fig(data_r, opt,col_r, num_in_r)

        num_classes_all = [num_classes_all_l, num_classes_all_r]
        num_samples = [num_samples_l, num_samples_r]

        fig = update_fig(fig,num_classes_all,num_samples)

    else:
        fig = make_subplots(rows=1, 
                            cols=1,
                            specs=[[{'type': 'mesh3D'}]])

        print('type(fig)',type(fig))

        data = load_data(opt)    
        print('np.shape(data)',np.shape(data))
        col_num = 1
        fig, num_classes_all, num_samples = make_fig(data, opt,col_num, 0)
        fig = update_fig(fig,num_classes_all,num_samples)



    fig.show()
    print("Used color vector ", colors)

    # https://stackoverflow.com/questions/60513164/display-interactive-plotly-chart-html-file-on-github-pages
    # https://plotly.com/python/interactive-html-export/
    fig.write_html(opt.output_html_pathname)
    # pio.write_html(fig, file='index.html', auto_open=True)





