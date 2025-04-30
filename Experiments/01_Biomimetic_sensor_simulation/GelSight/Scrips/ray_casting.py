import open3d as o3d
import numpy as np


cube = o3d.geometry.TriangleMesh.create_box().translate([0, 0, 0])
cube = o3d.t.geometry.TriangleMesh.from_legacy(cube)

faces = np.load('MESH/triangle_faces.npy')

mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(np.column_stack((XP, YP, ZP)))
mesh.triangles = o3d.utility.Vector3iVector(faces)

scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(cube)
scene.add_triangles(mesh)

rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
    fov_deg=120,
    center=[20, 20, 8],
    eye=[20, 20, 42],
    up=[0, 1, 0],
    width_px=640,
    height_px=480,
)

ans = scene.cast_rays(rays)

import matplotlib.pyplot as plt
plt.imshow(ans['t_hit'].numpy())
plt.show()
