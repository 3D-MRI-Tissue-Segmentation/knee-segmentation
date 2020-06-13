
""" Generates voxel plotly graph

Get started with 
python3 run_plotly.py -toy 10

"""

from options import Options
from data_loader import load_data
from VoxelData import VoxelData
from RenderData import RenderData
import chart_studio.plotly
import plotly.io as pio
import plotly.graph_objects as go
import plotly.colors as colors
import matplotlib.pyplot as plt
import random
import numpy as np

if __name__ == "__main__":
    
    opt = Options().parse()

    # Colors
    # colors = ["pink", "red", "orange", "yellow", "lightgreen", "lightblue", "purple"]
    colors = colors.DEFAULT_PLOTLY_COLORS
    if opt.shuffle_colors:
        # Two types of random, one roll one random shuffle
        # colors = np.roll(np.array(colors), np.random.randint(0,np.size(colors)))
        random.shuffle(colors)
    background_seg = 0

    data = load_data(opt)


    fig = go.Figure()
    print("Generating figure")

    
    data_is_3d = len(np.shape(data)) <=3
    if data_is_3d:
        num_samples = 1
    else:
        num_samples = np.shape(data)[0]


    for j in range(num_samples):
        
        if data_is_3d:
            curr_data = data
        else:
            curr_data = data[j]
        
        Voxels = VoxelData(curr_data)
    
        # print("Voxels.data\n",Voxels.data)
        # print("Voxels.vertices\n",Voxels.vertices)
        # print("Voxels.triangles\n",Voxels.triangles)

    
        for i, seg_class in enumerate(Voxels.class_colors):
            print("Making segmentation voxels ", i, "/", np.size(Voxels.class_colors))
            if seg_class == background_seg:
                continue

            curr_class = RenderData(Voxels.get_class_voxels(seg_class))
            curr_color = colors[i]

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



    num_classes = 2 # Voxels.num_classes

    steps = []
    # curr_sample=1

    # Loop through number of samples and for each make a 'step' with the 'visible' vector set True for each trace that's part of 1 volume
    num_samples = 6
    for i in range(num_samples):
        step = dict(
            method = 'restyle',  
            args = ['visible', [False] * len(fig.data)],
        )

        for 
        step['args'][1][k] = True




        curr_sample = int(np.ceil((i+1)/num_classes))
        print('curr_sample', curr_sample)
        # Hide all traces
        step = dict(
            method = 'restyle',  
            args = ['visible', [False] * len(fig.data)],
        )
        # Enable the traces we want to see
        print('range(curr_sample, (curr_sample+num_classes))',range(curr_sample, (curr_sample+num_classes)))
        for k in range(curr_sample, (curr_sample+num_classes)):
            step['args'][1][k] = True



        # if (i < curr_sample*num_classes) & ((curr_sample-1)*num_classes < i):
        #     for k in range(num_classes):
        #         step['args'][1][(curr_sample-1) * k] = True
        
        print('step',step)
        # Add step to step list
        steps.append(step)

    sliders = [dict(
        currentvalue={"prefix": "Epoch: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        showlegend=True,
        sliders=sliders
    )

    fig.show()

    print("Used color vector ", colors)

    # username = 'olive004' # your username
    # api_key = '1Zg3g69qKJNwLBiInHaE' # your api key - go to profile > settings > regenerate key
    # chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
    # chart_studio.plot(fig, filename = 'segmentation', auto_open=True)

    # https://stackoverflow.com/questions/60513164/display-interactive-plotly-chart-html-file-on-github-pages
    # https://plotly.com/python/interactive-html-export/
    fig.write_html(opt.output_html_pathname)
    # pio.write_html(fig, file='index.html', auto_open=True)





