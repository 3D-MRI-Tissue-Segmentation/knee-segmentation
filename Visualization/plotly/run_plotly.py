
""" Generates voxel plotly graph

Get started with 
python3 run_plotly.py -toy 10

"""

from options import Options
from data_loader import load_data
from VoxelData import VoxelData
from RenderData import RenderData
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    
    opt = Options().parse()
    data = load_data(opt)

    Voxels = VoxelData(data)
    colors = ["pink", "red", "orange", "yellow", "green", "purple", "lightpink", "black"]
    background_seg = 1
    # print("Voxels.data\n",Voxels.data)
    # print("Voxels.vertices\n",Voxels.vertices)
    # print("Voxels.triangles\n",Voxels.triangles)

    print("Generating figure")
    fig = go.Figure()
    
    for i, seg_class in enumerate(Voxels.class_colors):
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
            opacity=0.7
            ))

    fig.show()






