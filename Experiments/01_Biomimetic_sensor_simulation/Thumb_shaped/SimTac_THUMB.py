import trimesh
import argparse
import os
import sys
import math
import cv2
import re
import matplotlib.pyplot as plt
import open3d as o3d
import taichi as ti
import numpy as np
from trimesh.ray.ray_pyembree import RayMeshIntersector
from scipy import interpolate
from Scrips.GelTip_Rendering import generate_tactile_rgb
from PIL import Image

## Input Parameters
parser = argparse.ArgumentParser()
parser.add_argument("--object", default="Sphere") # indenter shape
parser.add_argument("--offset_indenter_x", type=float, default=0.0)
parser.add_argument("--offset_indenter_y", type=float, default=0.0)
parser.add_argument("--offset_indenter_z", type=float, default=0.0)
parser.add_argument("--depth_Y", type=float, default=1.8)
parser.add_argument("--indenter_velocity", type=float, default=20)

def save(p_xpos_list, p_ypos_list, p_zpos_list):
    
    x_np = x.to_numpy()
    p_indices = Contact_Surface_set.to_numpy()
    p_xpos = x_np[p_indices, 0]
    p_ypos = x_np[p_indices, 1]
    p_zpos = x_np[p_indices, 2]
    p_xpos_list = np.vstack((p_xpos_list, p_xpos))
    p_ypos_list = np.vstack((p_ypos_list, p_ypos))
    p_zpos_list = np.vstack((p_zpos_list, p_zpos))
     
    return p_xpos_list, p_ypos_list, p_zpos_list

def wide_angle_camera_transfer(xp, yp, p_depth):
  for i in range(len(xp)):
    relative_position = np.array([xp[i], yp[i], p_depth[i]]) - camera_position
    depth[i] = np.linalg.norm(relative_position)
    pixel_v[i] = fx * relative_position[0] / relative_position[2] + cx
    pixel_u[i] = -fy * relative_position[1] / relative_position[2] + cy

  return pixel_u, pixel_v, depth  

def find_covered_vertices(XP, YP, ZP, faces, camera_position, initial_npy):
    # Load the target mesh
    vertices = o3d.utility.Vector3dVector(np.column_stack((XP, YP, ZP)))
    triangles = o3d.utility.Vector3iVector(faces)
    mesh = o3d.geometry.TriangleMesh(vertices, triangles)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    ray_intersector = RayMeshIntersector(trimesh_mesh)
    initial_positions = initial_npy.reshape(-1, 3)
    current_positions = np.column_stack((XP, YP, ZP))
    position_differences = np.linalg.norm(current_positions - initial_positions, axis=1)
    changed_indices = np.where(position_differences > 1e-6)[0]  
    origins = np.tile(camera_position, (len(changed_indices), 1))
    directions = current_positions[changed_indices] - camera_position
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    hit_indices = ray_intersector.intersects_first(origins, directions)
    covered_vertex_indices = set()

    for idx, hit_index in zip(changed_indices, hit_indices):
        if hit_index != -1:
            face_vertices = trimesh_mesh.faces[hit_index]
            if idx not in face_vertices:
                covered_vertex_indices.add(idx)

    covered_vertex_indices = np.array(list(covered_vertex_indices), dtype=int)
    uncovered_mask = np.ones(len(XP), dtype=bool)
    uncovered_mask[covered_vertex_indices] = False
    covered_mask = np.zeros(len(XP), dtype=bool)
    covered_mask[covered_vertex_indices] = True
    XP_uncovered = XP[uncovered_mask]
    YP_uncovered = YP[uncovered_mask]
    ZP_uncovered = ZP[uncovered_mask]
    XP_covered = XP[covered_mask]
    YP_covered = YP[covered_mask]
    ZP_covered = ZP[covered_mask]

    return XP_uncovered, YP_uncovered, ZP_uncovered, XP_covered, YP_covered, ZP_covered

def img_generation(p_xpos_list, p_ypos_list, p_zpos_list, initial_npy, image_name):
  #XP_uncovered, YP_uncovered, ZP_uncovered, XP_covered, YP_covered, ZP_covered = find_covered_vertices(p_xpos_list, p_ypos_list, p_zpos_list, faces-1, camera_position_1, initial_npy)
  pixel_u, pixel_v, depth = wide_angle_camera_transfer(p_xpos_list, p_ypos_list, p_zpos_list)
  i_depth = interpolate.griddata((pixel_v, pixel_u), depth/1000, (xi[None,:],yi[:,None]), method='cubic', fill_value=0.0001)
  i_depth = i_depth.astype(np.float32)
  # npy_path = "./Results/THUMB/Real_time/depth/"+ image_name +".npy"
  # np.save(npy_path,i_depth)
  img = generate_tactile_rgb(i_depth)
  if marker == 0:
    img = img
  img = np.array(img)
  img = np.squeeze(img)
  img = (img).astype(np.uint8)
  # save the rendered tactile image
  image_path = "./Results/THUMB/Real_time/sim/"+ image_name +".png"
  #cv2.imwrite(image_path,img)
  cv2.imshow("tactile_image_MuJoCo",img)
  cv2.waitKey(1) 

# Fixed parameters
parser.add_argument("--particle", default="100", choices=["1","10","100"]) # indenter particle density 
parser.add_argument("--translation_x", type=float, default=20) # translation of the indenter model in X
parser.add_argument("--translation_y", type=float, default=21.8) # translation of the indenter model in Y
parser.add_argument("--translation_z", type=float, default=15) # translation of the indenter model in Z
parser.add_argument("--depth_X", type=float, default=0)
parser.add_argument("--depth_Z", type=float, default=0) # Z is the press direction
parser.add_argument("--shear_X", type=float, default=0)
parser.add_argument("--shear_Y", type=float, default=0)
parser.add_argument("--shear_Z", type=float, default=0)
parser.add_argument("--marker", type=int, default=0)
parser.add_argument("--displacement_interval", type=float, default=0.0125)
args = parser.parse_args()

## Initialization
# taichi 
ti.init(arch=ti.gpu)
t_ti = ti.field(dtype=ti.int32, shape=())
t_ti[None] = 0 # to record the iterations 

## Define Indenter Pose
obj_name = "obj_" + args.particle + "/" + args.object + ".npy"
data_ = np.load(obj_name)
data = data_.copy()
# translation, note the order of the trans/rota 
data[:,0] = data[:,0] + args.translation_x
data[:,1] = data[:,1] + args.translation_y
data[:,2] = data[:,2] + args.translation_z

## Create path
if not os.path.exists("./Results/THUMB/Real_time/Deformation"):
    os.makedirs("./Results/THUMB/Real_time/Deformation") 
if not os.path.exists("Results/THUMB/Real_time/sim"):
    os.makedirs("Results/THUMB/Real_time/sim")
if not os.path.exists("Results/THUMB/Real_time/depth"):
    os.makedirs("Results/THUMB/Real_time/depth")
if not os.path.exists("Results/THUMB/Real_time/PointCloud"):
    os.makedirs("Results/THUMB/Real_time/PointCloud")
if not os.path.exists("Results/THUMB/Real_time/Force_data"):
    os.makedirs("Results/THUMB/Real_time/Force_data")
if not os.path.exists("Results/THUMB/Real_time/Visualization_Frames"):
    os.makedirs("Results/THUMB/Real_time/Visualization_Frames")

## Load Model Mesh Information
# load membrane particles 
npy_file_path = "./Abaqus_model/MESH04/THUMB.npy"
data_abaqus_original = np.load(npy_file_path)
shape_model = data_abaqus_original.shape
data_abaqus = ti.field(ti.f32, shape=(shape_model[0], shape_model[1]))
data_abaqus.from_numpy(data_abaqus_original) 
# load marker initial position 
initial_npy_file_path = "./Abaqus_model/MESH04/initial_npy.npy"
initial_npy = np.load(initial_npy_file_path)
Contact_Surface_set_path = "./Abaqus_model/MESH04/Contact_Surface_index.txt"
Contact_Surface_list = []
Boundary_Condition_list = []
with open(Contact_Surface_set_path, "r") as file:
    for line in file:
        data_1 = int(line.strip())  
        Contact_Surface_list.append(int(data_1))
Contact_Surface_set_num = len(Contact_Surface_list)
Contact_Surface_set = ti.field(dtype=int, shape=Contact_Surface_set_num)
def fill_Contact_Surface_field():
    for i in ti.static(range(Contact_Surface_set_num)):
        Contact_Surface_set[i] = Contact_Surface_list[i]-1
fill_Contact_Surface_field()
Boundary_Condition_set_path = "./Abaqus_model/MESH04/Boundary_Condition_index.txt"
with open(Boundary_Condition_set_path, "r") as file:
    for line in file:
        data_2 = int(line.strip())  
        Boundary_Condition_list.append(int(data_2))
Boundary_Condition_set_num = len(Boundary_Condition_list) 
Boundary_Condition_set = ti.field(dtype=int, shape=Boundary_Condition_set_num) 
def fill_Boundary_Condition_field():
    for i in ti.static(range(Boundary_Condition_set_num)):
        Boundary_Condition_set[i] = Boundary_Condition_list[i]-1
fill_Boundary_Condition_field()

## Define Parameters
p_xpos_list = np.empty([0,Contact_Surface_set_num])
p_ypos_list = np.empty([0,Contact_Surface_set_num])
p_zpos_list = np.empty([0,Contact_Surface_set_num])
p_pos_list = np.empty([0,Contact_Surface_set_num])
v_x = ti.field(dtype=float, shape=())
v_y = ti.field(dtype=float, shape=())
v_z = ti.field(dtype=float, shape=())
# iteration
dt = 1e-4
# end position
end_position_depth_X = args.depth_X
end_position_depth_Y = args.depth_Y
end_position_depth_Z = args.depth_Z
end_position_shear_X = args.shear_X
end_position_shear_Y = args.shear_Y
end_position_shear_Z = args.shear_Z
i_1 = 0
j_1 = 0
# velocity of the indenter
if end_position_depth_X != 0:
   v_x[None] = args.indenter_velocity
else:
   v_x[None] = 0
if end_position_depth_Y != 0:
   v_y[None] = args.indenter_velocity
else:
   v_y[None] = 0
if end_position_depth_Z != 0:
   v_z[None] = args.indenter_velocity
else:
   v_z[None] = 0
end_position_depth = max(end_position_depth_X, end_position_depth_Y, end_position_depth_Z)
# define the parameters to record the movement distance of the indenter
distance = ti.field(dtype=float, shape=())
distance[None] = 0 
# elastomer dimension
r = 10 ## unit: mm
thickness = 4
dim = 3
yield_stress = 1.0
# mass/volume/density
p_rho_elastomer, p_rho_indenter = 1e-6, 8e-6  ## mass density unit: kg/mm3
p_vol_elastomer = r*r*thickness
p_mass_elastomer = p_rho_elastomer * p_vol_elastomer / (shape_model[0])
p_mass_indenter = p_mass_elastomer * (p_rho_indenter / p_rho_elastomer)
p_vol = p_mass_elastomer / p_rho_elastomer
# Young's modulus and Poisson's ratio
#E, nu = 0.1981, 0.4797 ## unit: N/mm2 / MPa
E, nu = 0.145, 0.45
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu)) # Lame parameters - may change these later to model other materials
E_indenter, nu_indenter = 210, 0.3
mu_1, lambda_1 = E_indenter / (2 * (1 + nu_indenter)), E_indenter * nu_indenter / ((1+nu_indenter) * (1 - 2 * nu_indenter)) 
# grid initialization
n_grid = 120
dx = 40 / n_grid
inv_dx = 1 / dx
grid_v = ti.Vector.field(3, dtype=float, shape=(n_grid, n_grid, n_grid)) # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid, n_grid)) # grid node mass
grid_stress = ti.Vector.field(3, dtype=float, shape=(n_grid, n_grid, n_grid)) # grid node stress
# particle initialization
n_particles = shape_model[0]+np.shape(data)[0]  ## n_particle includes particles from indenter
x = ti.Vector.field(3, dtype=float, shape=n_particles) # position
p_mass = ti.field(dtype=float, shape=n_particles) # position
x_indenter = ti.Vector.field(3, dtype=float, shape=shape_model[0]) # position
x_membrane = ti.Vector.field(3, dtype=float, shape=np.shape(data)[0]) # position
v = ti.Vector.field(3, dtype=float, shape=n_particles) # velocity
C = ti.Matrix.field(3, 3, dtype=float, shape=n_particles) # affine velocity field
F = ti.Matrix.field(3, 3, dtype=float, shape=n_particles) # deformation gradient
material = ti.field(dtype=int, shape=n_particles) # material id
#material_single_layer = ti.field(dtype=int, shape=n_particles_single_layer) # material id
# define the camera parameters
fov_degrees = 100
width = 640
height = 640
camera_position = np.array([20, 20, 8])  
# camera_position_1 = np.array([20, 20, -3])
fx = width / (2 * np.tan(np.radians(fov_degrees / 2)))  
fy = fx 
cx = width / 2  
cy = height / 2
img_width = 640
img_height = 480
xi = np.linspace(0,640, num=img_width)
yi = np.linspace(80,560, num=img_height)
i_depth = np.zeros((img_height,img_width))
pixel_u = np.zeros(Contact_Surface_set_num)
pixel_v = np.zeros(Contact_Surface_set_num)
depth = np.zeros(Contact_Surface_set_num)
# marker or markerless
marker = args.marker

@ti.func
def zero_matrix():
    return [zero_vec(), zero_vec(), zero_vec()]

@ti.func
def zero_vec():
    return [0.0, 0.0, 0.0]

# main iteration
@ti.kernel
def substep():
  index = 0
  for i, j, k in grid_m:
    grid_v[i, j, k] = [0, 0, 0]
    grid_m[i, j, k] = 0

  # particle to grid
  for p in x: 

    # first for particle p, compute base index
    base = (x[p] * inv_dx - 0.5).cast(int)
    
    # quadratic kernels 
    fx = x[p] * inv_dx - base.cast(float)
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
    mu, la = mu_0, lambda_0 
    mu_1, la_1 = mu_1, lambda_1
    U, sig, V = ti.svd(F[p])
    J = 1.0
    for d in ti.static(range(3)):
      J *= sig[d, d]
    #r, s = ti.polar_decompose(F[p])
    cauchy = ti.Matrix(zero_matrix())
       
    if material[p] == 0:  

      p_mass[p] = p_mass_elastomer 
      cauchy = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 3) * la * J * (J - 1)  ## P*FT = (cauchy*F)*FT
      
    if material[p] == 1:
    
      p_mass[p] = p_mass_indenter
      cauchy = 2 * mu_1 * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 3) * la_1 * J * (J - 1)  ## P*FT = (cauchy*F)*FT  equal to 0 if there is no deformation, if F[p]=I, then cauchy = 0
    
    stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * cauchy ## LAMBTA*P*FT     
    affine = stress + p_mass[p] * C[p]  ## LAMBTA*P*FT+C*m
    
    #P2G for velocity and mass 
    for i, j, k in ti.static(ti.ndrange(3, 3, 3)): # Loop over 3x3x3 grid node neighborhood
      offset = ti.Vector([i, j, k])
      dpos = (offset.cast(float) - fx) * dx  ## real relative distance
      weight = w[i][0] * w[j][1] * w[k][2]   
      grid_m[base + offset] += weight * p_mass[p] # mass transfer
      grid_v[base + offset] += weight * (p_mass[p] * v[p] + affine @ dpos) 
  
  # grid operation
  for i, j, k in grid_m:
        
    if grid_m[i, j, k] > 0: 
      grid_v[i, j, k] = (1 / grid_m[i, j, k]) * grid_v[i, j, k] # momentum to velocity
      
      #wall collisions - handle all 3 dimensions
      if i < 3 and grid_v[i, j, k][0] < 0:          grid_v[i, j, k][0] = 0 # Boundary conditions
      if i > n_grid - 3 and grid_v[i, j, k][0] > 0: grid_v[i, j, k][0] = 0
      if j < 3 and grid_v[i, j, k][1] < 0:          grid_v[i, j, k][1] = 0
      if j > n_grid - 3 and grid_v[i, j, k][1] > 0: grid_v[i, j, k][1] = 0
      if k < 3 and grid_v[i, j, k][2] < 0:          grid_v[i, j, k][2] = 0
      if k > n_grid - 3 and grid_v[i, j, k][2] > 0: grid_v[i, j, k][2] = 0   
  
  # grid to particle
  for p in x: 

    # compute base index
    base = (x[p] * inv_dx - 0.5).cast(int)

    # quadratic kernels
    fx = x[p] * inv_dx - base.cast(float)
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]

    new_v = ti.Vector.zero(float, 3)
    new_C = ti.Matrix.zero(float, 3, 3)
    new_F = ti.Matrix.zero(float, 3, 3)

    for i, j, k in ti.static(ti.ndrange(3, 3, 3)): # loop over 3x3x3 grid node neighborhood
      dpos = (ti.Vector([i, j, k]).cast(float) - fx) * dx
      g_v = grid_v[base + ti.Vector([i, j, k])]
      weight = w[i][0] * w[j][1] * w[k][2]
      new_v += weight * g_v
      new_C += 4 * inv_dx * inv_dx * weight * g_v.outer_product(dpos)
    
    for i in range(Boundary_Condition_set_num):
      if p == Boundary_Condition_set[i]:
        new_v = ti.Vector([0,0,0])
      
    if material[p] == 1:
      new_C = ti.Matrix.zero(float, 3, 3)  # C equals to 0 for indenter, so F[p]=I, there is no elastic stress.
    
      ## set the velocity of the indenter  
      new_v = ti.Vector([v_x[None],v_y[None],-v_z[None]])  # mentain the velocity of indenter
    
    v[p], C[p] = new_v, new_C

    # move the particles
    x[p] += dt * v[p] 
    #x_2d[p] = [x[p][0]*2+15, x[p][2]*2+3] # update 2d positions
    F[p] = (ti.Matrix.identity(float, 3) + (dt * new_C)) @ F[p]  # update F (explicitMPM way)

    if material[p] == 1:
        x_membrane[i_1] = x[p]
        i_1 = i_1+1

    if material[j_1] == 0:
        x_indenter[p] = x[p]
        j_1 = j_1+1 

@ti.kernel
def initialize(data: ti.types.ndarray(),data_len: ti.f32):
  
  for i in range(shape_model[0]):
    for j in ti.static(range(shape_model[1])):
      x[i][j] = data_abaqus[i, j]
      
  for m in range(shape_model[0]):
    offest = ti.Vector([20,20,5]) 
    x[m] = x[m] + offest 
    v[m] = [0, 0, 0]
    material[m] = 0
    F[m] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  ## initialize the deformation gradient to identity matrix
    C[m] = ti.Matrix.zero(float, 3, 3)
    p_mass[m] = p_mass_elastomer
  
  for i in ti.ndrange(int(data_len)):
    m = i+shape_model[0]
    offest = ti.Vector([args.offset_indenter_x,args.offset_indenter_y,args.offset_indenter_z])
    x[m] = ti.Vector([data[i,0],data[i,1],data[i,2]])+offest
    v[m] = [v_x[None],v_y[None],-v_z[None]]
    material[m] = 1
    F[m] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  ## initialize the deformation gradient to identity matrix
    C[m] = ti.Matrix.zero(float, 3, 3)
    p_mass[m] = p_mass_indenter

# taichi gui visualization
window = ti.ui.Window("Taichi Simulation on GGUI", (1000, 800), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((255, 255, 255))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

# initialization
initialize(data, np.shape(data)[0])

# step sim
while window.running:
  while True:     
    t_ti[None] += 1
    substep()
    distance[None] = t_ti[None]*dt*args.indenter_velocity
    x_np = x.to_numpy()
    p_indices = Contact_Surface_set.to_numpy()
    p_xpos_list = x_np[p_indices, 0]
    p_ypos_list = x_np[p_indices, 1]
    p_zpos_list = x_np[p_indices, 2]
    p_pos_list = x_np[p_indices]
    p_zpos_list_relative = p_pos_list - initial_npy

    if t_ti[None]%(args.displacement_interval / (args.indenter_velocity * dt)) == 0: 
      print("===========================================================")
      print('iteration times: ', t_ti[None])
      # set the data name here
      data_name = args.object + str(t_ti[None])
      img_generation(p_xpos_list, p_ypos_list, p_zpos_list, initial_npy, data_name)

    if distance[None]>=abs(end_position_depth): 
      #print('movement distance in X direction: ', distance[None])
      v_z[None] = 0
      if args.shear_X > 0:
          v_x[None] = args.indenter_velocity
      elif args.shear_X < 0:
          v_x[None] = -args.indenter_velocity
      else:
          v_x[None] = 0

      if args.shear_Y > 0:
          v_y[None] = args.indenter_velocity
      elif args.shear_Y < 0:
          v_y[None] = -args.indenter_velocity
      else:
          v_y[None] = 0
          
    if distance[None] > abs(end_position_depth) + max(abs(end_position_shear_X), abs(end_position_shear_Y)): 
      #print('total movement distance direction: ', distance[None])
      shear_X_str = "0" if args.shear_X == 0 else "{:.1f}".format(args.shear_X)
      shear_Y_str = "0" if args.shear_Y == 0 else "{:.1f}".format(args.shear_Y)
      sys.exit()
    
    camera.position(-80, 80, 30)
    camera.lookat(0, 30, 30)
    camera.up(0, 0, 1)
    scene.set_camera(camera)
    scene.point_light(pos=(-80, 80, 80), color=(0.3, 0.3, 0.3))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.particles(x_indenter, radius=0.3, color=(142/255,182/255,156/255))
    scene.particles(x_membrane, radius=0.1, color=(217/255,79/255,51/255))
    canvas.scene(scene)
    # if t_ti[None]%(args.displacement_interval / (args.indenter_velocity * dt)) == 0: 
    #     window.save_image('./Results/THUMB/Real_time/Visualization_Frames/' + str(args.object) + '_frame_' + str(t_ti[None]) + '.png')
    #     # Open the image that was saved
    #     image_path = './Results/THUMB/Real_time/Visualization_Frames/' + str(args.object) + '_frame_' + str(t_ti[None]) + '.png'
    #     image = Image.open(image_path)

    #     # Convert to RGBA (which has an alpha channel for transparency)
    #     image = image.convert("RGBA")

    #     # Get data of the image
    #     data = image.getdata()

    #     # Modify the data to make the background transparent (assuming white background to be transparent)
    #     new_data = []
    #     for item in data:
    #         # Change all white (also shades of whites) pixels to transparent
    #         if item[:3] == (255, 255, 255):  # White background
    #             new_data.append((255, 255, 255, 0))  # Fully transparent
    #         else:
    #             new_data.append(item)

    #     # Update the image with new data
    #     image.putdata(new_data)

    #     # Save the modified image with transparency
    #     image.save(image_path)
    window.show()
  
  

