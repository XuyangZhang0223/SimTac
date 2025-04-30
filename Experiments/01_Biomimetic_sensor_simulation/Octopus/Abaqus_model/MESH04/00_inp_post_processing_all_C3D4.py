# change the code with 'modified'

import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def findNodes(filepath):
    with open(filepath, 'r') as file:
        contents = file.read()
    
    part_start = "*Part, name=Octopus"  # modified
    part_end = "*Solid Section, elset=Set-1, material=elastomer-new"  # modified
    part_content = contents[contents.find(part_start):contents.find(part_end)]
    
    node_start = "*Node"
    node_end = "*Element, type=C3D4H"
    node_data = part_content[part_content.find(node_start) + len(node_start):part_content.find(node_end)]
    
    node_coords = [float(x) for x in node_data.replace(',', '').split()]
    num_nodes = len(node_coords) // 4
    node_coords = np.array(node_coords).reshape((num_nodes, 4))
    return node_coords[:, 1:]

def extract_indices(inp_content, start_marker, end_marker):
    start_index = inp_content.find(start_marker)
    end_index = inp_content.find(end_marker, start_index)
    section_text = inp_content[start_index:end_index]
    lines = section_text.split('\n')
    indices = [int(num) for line in lines for num in line.split(',') if num.strip().isdigit()]
    return indices

def save_indices(file_path, indices):
    with open(file_path, 'w') as file:
        for index in indices:
            file.write(str(index) + '\n')
            
def visualize_points(points, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title)
    plt.show()

def read_npy(file_path, indices):
    data = np.load(file_path)
    return data[indices - 1, :]

def find_contact_surface_nodes(filepath):
    with open(filepath, 'r') as file:
        contents = file.read()
    start_marker = '*Nset, nset="Contact Surface"'
    end_marker = '*Elset, elset="Contact Surface"'
    start_index = contents.find(start_marker)
    end_index = contents.find(end_marker, start_index)
    contact_surface_text = contents[start_index:end_index]
    lines = contact_surface_text.split('\n')
    return [int(num) for line in lines for num in line.split(',') if num.strip().isdigit()]

def find_element_nodes(filepath):
    with open(filepath, 'r') as file:
        contents = file.read()
    # Find the part with the specified name
    part_start = "*Element, type=C3D4H" # modified
    part_end = "*Solid Section, elset=Set-1, material=elastomer-new" # modified
    part_content = contents[contents.find(part_start):contents.find(part_end)]  
    # Extract node data from the part using contents
    element_start = '*Element, type=C3D4H' # modified
    element_end = '*Nset, nset=Set-1, generate'  
    element_data = part_content[part_content.find(element_start) + len(element_start):part_content.find(element_end)] 
    node_coords = [int(x) for x in element_data.replace(',', ' ').split()]
    num_nodes = len(node_coords) // 5
    
    return np.array(node_coords).reshape((num_nodes, 5))[:, 1:]
    
def find_mesh_element(filepath):
    with open(filepath, 'r') as file:
        contents = file.read()
    # Extract node data from the part using contents
    node_start = '*Elset, elset="Contact Surface", instance=Octopus-1' # modified
    node_end = '*Nset, nset="Fixed Surface", instance=Octopus-1' # modified
    node_data = contents[contents.find(node_start) + len(node_start):contents.find(node_end)]
    element_list = []
    # Remove commas, split at whitespace, and convert to integers or floats
    for line in node_data.split('\n'):
        if not line.startswith('*'):
            elements = [float(x) for x in line.replace(',', '').split()]
            element_list.extend([int(x) if x.is_integer() else x for x in elements])        

    return element_list

def txt_to_obj(input_file, output_file): # for C3D4 mesh, there are 3 nodes within 1 face.
    vertices, faces = [], []
    with open(input_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
    with open(output_file, 'w') as obj_file:
        for vertex in vertices:
            obj_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            obj_file.write(f"f {face[0]} {face[1]} {face[2]} \n")

def read_obj(file_path):
    vertices = []
    faces = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertex = list(map(float, line.strip().split()[1:]))
                vertices.append(vertex)
            elif line.startswith('f '):
                face = [int(vertex.split('/')[0]) - 1 for vertex in line.strip().split()[1:]]
                faces.append(face)
    return np.array(vertices), np.array(faces)

def reorder_faces(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    vertices, faces = [], []
    for line in lines:
        if line.startswith('v '):
            vertices.append(list(map(float, line.strip().split()[1:])))
        elif line.startswith('f '):
            face = list(map(int, line.strip().split()[1:]))
            faces.append(face)
    
    vertices, faces = np.array(vertices), np.array(faces)
    triangle_faces = []
    
    for face in faces:
        if len(face) == 4:  
            center = np.mean(vertices[face - 1], axis=0)
            angles = np.arctan2(vertices[face - 1][:, 1] - center[1], vertices[face - 1][:, 0] - center[0])
            sorted_indices = np.argsort(angles)
            sorted_face_vertices = face[sorted_indices]
            triangle_faces.append([sorted_face_vertices[0], sorted_face_vertices[1], sorted_face_vertices[2]])
            triangle_faces.append([sorted_face_vertices[0], sorted_face_vertices[2], sorted_face_vertices[3]])
        
        elif len(face) == 3: 
            triangle_faces.append(face)
        else:
            print(f"Warning: Skipping non-triangle/quadrilateral face {face}")
    
    np.save("MESH04/triangle_faces.npy", np.array(triangle_faces))
    
    with open("MESH04/mesh_modified.obj", "w") as file:
        for vertex in vertices:
            file.write(f"v {' '.join(map(str, vertex))}\n")
        for face in triangle_faces:
            file.write(f"f {' '.join(map(str, face))}\n")
    
    return vertices, triangle_faces

    
def visualize_obj(vertices, faces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(vertices[:,0], vertices[:,1], vertices[:,2], triangles=faces, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    

def generate_mesh_txt(inp_file_path, npy_file_path, output_txt_file):

    data = np.load(npy_file_path)
    with open(output_txt_file, 'w') as txt_file:
        for element in data:
            txt_file.write(f'v {element[0]} {element[1]} {element[2]}\n')
    element_list = find_mesh_element(inp_file_path)
    element_num = np.shape(element_list)[0]
    elements = find_element_nodes(inp_file_path)
    contact_surface_nodes_list = find_contact_surface_nodes(inp_file_path)  
    mesh = [] 
    for i in range(element_num):
        current_element = elements[element_list[i]-1]
        filtered_element = [] 
        for dim in current_element:
            if dim in contact_surface_nodes_list:
                index_in_contact_surface = contact_surface_nodes_list.index(dim) + 1
                filtered_element.append(int(index_in_contact_surface)) 
                 
        formatted_element = ' '.join(['f ' + ' '.join(map(str, filtered_element))]) + '\n'
        mesh.append(formatted_element)

    with open(output_txt_file, 'a') as file:
        file.writelines(mesh)

def main():
    inp_file_path = 'MESH04/Octopus.inp'  # modified
    npy_file_path = 'MESH04/Octopus.npy'  # modified
    contact_txt_path = 'MESH04/Contact_Surface_index.txt'
    boundary_txt_path = 'MESH04/Boundary_Condition_index.txt'
    initial_npy_path = 'MESH04/initial_npy.npy'
    mesh_txt_file = "MESH04/mesh.txt"
    mesh_obj_file = "MESH04/mesh.obj"
    
    with open(inp_file_path, 'r') as inp_file:
        inp_content = inp_file.read()
    
    points = findNodes(inp_file_path)
    np.save(npy_file_path, points)
    visualize_points(points, 'All Nodes')
    
    contact_indices = extract_indices(inp_content, '*Nset, nset="Contact Surface"', '*Elset, elset="Contact Surface"')
    save_indices(contact_txt_path, contact_indices)
    
    boundary_indices = extract_indices(inp_content, '*Nset, nset="Fixed Surface"', '*Elset, elset="Fixed Surface"')
    save_indices(boundary_txt_path, boundary_indices)
    
    contact_nodes = read_npy(npy_file_path, np.array(contact_indices))
    np.save(initial_npy_path, contact_nodes)
    visualize_points(contact_nodes, 'Initial_npy')
    
    generate_mesh_txt(inp_file_path, initial_npy_path, mesh_txt_file)
    
    txt_to_obj(mesh_txt_file, mesh_obj_file)
    vertices, faces = reorder_faces(mesh_obj_file)
    vertices, faces = read_obj(mesh_obj_file)
    visualize_obj(vertices, faces)

if __name__ == "__main__":
    main()
    

