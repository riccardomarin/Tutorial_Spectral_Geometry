import numpy as np
import os
from plyfile import PlyData, PlyElement 
import torch

def load_ply(fname):
    plydata = PlyData.read(fname)
    return [np.ascontiguousarray([np.asarray(plydata['vertex'][d]) for d in ['x','y','z']],dtype='float32').T, \
    np.ascontiguousarray(list(plydata['face']['vertex_indices']),dtype='int64')]



"""
Drawing functions
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

def plot_colormap(verts, trivs, colors=None, colorscale=[[0, 'rgb(0,0,255)'], [0.5, 'rgb(255,255,255)'], [1, 'rgb(255,0,0)']], point_size=2):
    "Draw multiple triangle meshes side by side"
    "colors must be list(range(num colors))"
    if type(verts) is not list:
        verts = [verts]
    if type(trivs) is not list:
        trivs = [trivs]
    if type(colors) is not list:
        colors = [colors]
    if type(verts[0]) == torch.Tensor:
        to_np = lambda x: x.detach().cpu().numpy()
        verts=[ to_np(v) for v in verts]

    
    "Check device for torch tensors"
    def to_cpu(v):
        if torch.is_tensor(v):
            return v.data.cpu()
        return v;
    verts = [to_cpu(x) for x in verts]
    trivs = [to_cpu(x) for x in trivs]
    colors = [to_cpu(x) for x in colors]
        
        
    nshapes = min([len(verts), len(colors), len(trivs)])

    fig = make_subplots(rows=1, cols=nshapes, specs=[[{'type': 'surface'} for i in range(nshapes)]])

    for i, [vert, triv, col] in enumerate(zip(verts, trivs, colors)):
        if triv is not None:
            if col is not None:
                mesh = go.Mesh3d(x=vert[:, 0], z=vert[:, 1], y=vert[:, 2],
                                 i=triv[:, 0], j=triv[:, 1], k=triv[:, 2],
                                 intensity=col,
                                 colorscale=colorscale,
                                 color='lightpink', opacity=1)
            else:
                mesh = go.Mesh3d(x=vert[:, 0], z=vert[:, 1], y=vert[:, 2],
                                 i=triv[:, 0], j=triv[:, 1], k=triv[:, 2])
        else:
            if col is not None:
                mesh = go.Scatter3d(x=vert[:, 0], z=vert[:, 1], y=vert[:, 2],
                                    mode='markers',
                                    marker=dict(
                                        size=point_size,
                                        color=col,  # set color to an array/list of desired values
                                        colorscale='Viridis',
                                        # choose a colorscale
                                        opacity=1
                                    ))
            else:
                mesh = go.Scatter3d(x=vert[:, 0], z=vert[:, 1], y=vert[:, 2],
                                    mode='markers',
                                    marker=dict(
                                        size=point_size,  # set color to an array/list of desired values
                                        colorscale='Viridis',  # choose a colorscale
                                        opacity=1
                                    ))

        fig.add_trace(mesh, row=1, col=i + 1)
        fig.get_subplot(1, i + 1).aspectmode = "data"

        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0, y=4, z=-1)
        )
        fig.get_subplot(1, i + 1).camera = camera

    #     fig = go.Figure(data=[mesh], layout=layout)
    fig.update_layout(
        #       autosize=True,
        margin=dict(l=10, r=10, t=10, b=10))
        #paper_bgcolor="LightSteelBlue")
    #fig.show()
    return fig