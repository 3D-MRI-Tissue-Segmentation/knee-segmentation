
""" Generates voxel plotly graph

Get started with 
python3 run_plotly.py -toy 10

"""

from options import Options
from data_loader import load_data
from VoxelData import VoxelData
from RenderData import RenderData
from RenderData import get_steps
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

    
    print('np.shape(data)',np.shape(data))


    fig = go.Figure()
    print("Generating figure")

    
    is_multiple_samples = type(data) == list
    num_samples = len(data)
    num_classes_all = []
    num_traces = 0

    for j in range(num_samples):
        
        if not is_multiple_samples:
            print('1 samplpe')
            curr_data = data
        else:
            print('multiple samples')
            curr_data = data[j]
        
        Voxels = VoxelData(curr_data)
        print('np.sum(curr_data)',np.sum(curr_data))
    
        # print("Voxels.data\n",Voxels.data)
        # print("Voxels.vertices\n",Voxels.vertices)
        # print("Voxels.triangles\n",Voxels.triangles)

        has_background = False
        for i, seg_class in enumerate(Voxels.class_colors):
            print("Making segmentation voxels ", i, "/", np.size(Voxels.class_colors))
            if seg_class == background_seg:
                has_background = True
                num_traces += 1
                continue

            curr_class = RenderData(Voxels.get_class_voxels(seg_class))
            curr_color = colors[i]
            print(i, 'curr_color',curr_color)

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
            

       
        if has_background:
            num_classes_all.append(Voxels.num_classes-1)
        else:
            num_classes_all.append(Voxels.num_classes)



    num_traces += len(fig.data)
    steps = get_steps(num_samples, num_traces, num_classes_all) 
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





