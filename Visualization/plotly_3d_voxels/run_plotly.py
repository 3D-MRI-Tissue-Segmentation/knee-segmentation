
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
import plotly.colors
import matplotlib.pyplot as plt
import random
import numpy as np

def get_colors(opt):
    # colors = ["pink", "red", "orange", "yellow", "lightgreen", "lightblue", "purple"]
    colors = plotly.colors.DEFAULT_PLOTLY_COLORS_
    if opt.shuffle_colors:
        # Two types of random, one roll one random shuffle
        # colors = np.roll(np.array(colors), np.random.randint(0,np.size(colors)))
        random.shuffle(colors)
    return colors


def make_fig(data, opt):
    fig = go.Figure()
    print("Generating figure")

    
    is_multiple_samples = type(data) == list
    if is_multiple_samples:
        num_samples = len(data)
    else: 
        num_samples = 1

    num_classes_all = []

    for j in range(num_samples):
        
        if not is_multiple_samples:
            print('There is 1 samplpe')
            curr_data = data
        else:
            print('There are multiple samples')
            curr_data = data[j]
        
        Voxels = VoxelData(curr_data)
        print('np.sum(curr_data)',np.sum(curr_data))
    
        # print("Voxels.data\n",Voxels.data)
        # print("Voxels.vertices\n",Voxels.vertices)
        # print("Voxels.triangles\n",Voxels.triangles)

        # has_background = False
        for i, seg_class in enumerate(Voxels.unique_classes):
            print("Making segmentation voxels ", (i+1), "/", Voxels.num_classes)
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
                ))
                continue

            curr_class = RenderData(Voxels.get_class_voxels(seg_class))
            curr_color = colors[int(seg_class)]
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
                ))
    
        num_classes_all.append(Voxels.num_classes)

    return fig, num_classes_all, num_samples




def updata_fig(fig, num_classes_all, num_samples):
    print('num_classes_all',num_classes_all)
    num_traces = len(fig.data)
    print('Total traces', num_traces)
    steps = get_steps(num_samples, num_traces, num_classes_all) 
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
    background_seg = 0

    if opt.dataroot_left and opt.dataroot_right:
        data_l, data_r = load_data(opt)
        fig_l, classes_counts_l, num_samples_l = make_fig(data_l, opt)
        fig_r, classes_counts_r, num_samples_r = make_fig(data_r, opt)

        fig_l = updata_fig(fig_l)
        fig_r = updata_fig(fig_r)
    else:
        data = load_data(opt)    
        print('np.shape(data)',np.shape(data))
        fig = make_fig(data, opt)



    fig.show()
    print("Used color vector ", colors)

    # https://stackoverflow.com/questions/60513164/display-interactive-plotly-chart-html-file-on-github-pages
    # https://plotly.com/python/interactive-html-export/
    fig.write_html(opt.output_html_pathname)
    # pio.write_html(fig, file='index.html', auto_open=True)





