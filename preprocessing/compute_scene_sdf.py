import os
import sys
import trimesh
import mesh2sdf
import numpy as np
sys.path.append("..")
import time
from env.scene.base_scene import Scene


scene_name = 'House_5'
scene = Scene(scene_name)
mesh = scene.trimesh_visual

mesh_scale = 0.8
size = 128
level = 2 / size

# normalize mesh
vertices = mesh.vertices
bbmin = vertices.min(0)
bbmax = vertices.max(0)
center = (bbmin + bbmax) * 0.5
scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
vertices = (vertices - center) * scale

# fix mesh
t0 = time.time()
sdf, mesh = mesh2sdf.compute(
    vertices, mesh.faces, size, fix=True, level=level, return_mesh=True)
t1 = time.time()

# output
mesh.vertices = mesh.vertices / scale + center
# mesh.export(scene_name + '.fixed.obj')

sdf_dict = {
    'sdf_norm_value': sdf,
    'scene_mesh_center': np.array(center),
    'scene_mesh_scale': np.array(scale),
    'resolution': np.array(size),
}
print(sdf_dict)
np.save(scene_name + '.npy', sdf_dict)
print('It takes %.4f seconds to process %s' % (t1-t0, scene_name))