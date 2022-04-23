import numpy as np
import os
from plyfile import PlyData, PlyElement 
import torch

def load_off(file):
    import numpy as np
    f=file.readline().strip()
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return np.asarray(verts), faces

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

def plot_colormap(verts, trivs, colors=None, colorscale=[[0, 'rgb(0,0,255)'], [0.5, 'rgb(255,255,255)'], [1, 'rgb(255,0,0)']], point_size=2, wireframe=False):
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
        
        if wireframe and triv is not None:
            tripts = vert[triv]
            tripts = np.concatenate([tripts,tripts[:,0:1,:],tripts[:,0:1,:]*np.nan],1).reshape(-1,3)
            lines = go.Scatter3d(
                   x=tripts[:,0],
                   y=tripts[:,2],
                   z=tripts[:,1],
                   mode='lines',
                   name='',
                   line=dict(color= '#000000', width=2))  
            fig.add_trace(lines, row=1, col=i + 1)
            
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0, y=4, z=-1)
        )
        fig.get_subplot(1, i + 1).camera = camera

    #     fig = go.Figure(data=[mesh], layout=layout)
    fig.update_layout(
        #       autosize=True,
        margin=dict(l=10, r=10, t=2, b=2))
        #paper_bgcolor="LightSteelBlue")
    #fig.show()
    return fig

def plot_RGBmap(verts, trivs, colors=None, colorscale=[[0, 'rgb(0,0,255)'], [0.5, 'rgb(255,255,255)'], [1, 'rgb(255,0,0)']], point_size=2):
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
                                 vertexcolor=col,
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
                                        vertexcolor=col,  # set color to an array/list of desired values
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
