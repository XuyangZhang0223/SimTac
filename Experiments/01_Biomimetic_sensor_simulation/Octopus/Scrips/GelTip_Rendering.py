import sys
sys.path.append("..Rendering_Phong_model")

import time
import numpy as np
import os
import cv2
from Rendering_Phong_model.sim_model.model_gpu import SimulationModel
from Rendering_Phong_model.sim_model.utils.vis_img import to_panel

def generate_tactile_rgb(depths):
    #fields_size = (480, 640)
    fields_size = (120, 160)
    #sim_size = (1280, 960)
    sim_size = (640, 480)
    field = 'plane'
    #field = 'linear'
    rectify_fields = True
    __location__ = os.path.dirname(os.path.abspath(__file__))
    assets_path = os.path.join(__location__, '../Rendering_Phong_model/experimental_setup/sim_assets/')
    npy_path = os.path.join(__location__, '../Rendering_Phong_model/npy_data/')
    light_fields = SimulationModel.load_assets(assets_path, fields_size, sim_size, field, 4) # modified 
    model = SimulationModel(**{
        'ia': 1,
        'fov': 120,
        'light_sources': [
        {'field': light_fields[2], 'color': [255, 0, 0], 'id': 0.6, 'is': 0.05}, # red
        {'field': light_fields[3], 'color': [0, 255, 0], 'id': 0.6, 'is': 0.05}, # green
        {'field': light_fields[0], 'color': [0, 0, 255], 'id': 0.6, 'is': 0.05}, # blue
        {'field': light_fields[1], 'color': [255, 255, 255], 'id': 0.3, 'is': 0.05} # white
        ],
        'background_depth': cv2.resize(np.load(npy_path + 'bkg.npy').astype(np.float32), sim_size),
        'background_img': (cv2.cvtColor(cv2.imread(npy_path + '/bkg.png'), cv2.COLOR_RGB2BGR) / 255).astype(np.float32),
        'elastic_deformation': False,
        'rectify_fields': rectify_fields
    })
    print('depths shape:',depths.shape)
    tactile_rgb = [
        cv2.cvtColor(model.generate(cv2.resize(depths, sim_size)), cv2.COLOR_RGB2BGR)
    ]

    return tactile_rgb

