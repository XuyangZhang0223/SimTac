import time

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

from sim_model.model_gpu import SimulationModel
from sim_model.utils.vis_img import to_panel
from math import pi, sin, cos

#fields_size = (120, 160)
fields_size = (480, 640)
sim_size = (640, 480)
#sim_size = (1280, 960)
# sim_size = (320, 240)
field = 'linear'
#field = 'geodesic'
#field = 'plane'
# field = 'planes'
# field = 'transport'
# field = 'rtransport'
# field = 'plane2'
rectify_fields = False

__location__ = os.path.dirname(os.path.abspath(__file__))
assets_path = os.path.join(__location__, './experimental_setup/geltip/sim_assets/')
save_path = os.path.join(__location__, 'Script/')

light_fields = SimulationModel.load_assets(assets_path, fields_size, sim_size, field, 9)

scale = 0.001
# z = -0.015
z = 0
led_radius = 15.5

model = SimulationModel(**{
    'ia': 1,
    'fov': 100,
    'light_sources': [
        {'field': light_fields[3], 'color': [255, 0, 0], 'id': 0.3, 'is': 0.05},  # [108, 82, 255]
        {'field': light_fields[4], 'color': [255, 0, 0], 'id': 0.3, 'is': 0.05},  # [108, 82, 255]
        {'field': light_fields[5], 'color': [255, 0, 0], 'id': 0.3, 'is': 0.05},  # [108, 82, 255]
        {'field': light_fields[6], 'color': [0, 255, 0], 'id': 0.275, 'is': 0.05},  # [255, 130, 115]
        {'field': light_fields[7], 'color': [0, 255, 0], 'id': 0.275, 'is': 0.05},  # [255, 130, 115]
        {'field': light_fields[8], 'color': [0, 255, 0], 'id': 0.275, 'is': 0.05},  # [255, 130, 115]
        {'field': light_fields[2], 'color': [0, 0, 255], 'id': 0.35, 'is': 0.05},  # [120, 255, 153]
        {'field': light_fields[1], 'color': [0, 0, 255], 'id': 0.35, 'is': 0.05},  # [120, 255, 153]
        {'field': light_fields[0], 'color': [0, 0, 255], 'id': 0.35, 'is': 0.05},  # [120, 255, 153]
        # {'field': light_fields[1], 'color': [110, 35, 30], 'id': 0.5, 'is': 0.05},  # [108, 82, 255]
        # {'field': light_fields[2], 'color': [30, 110, 30], 'id': 0.5, 'is': 0.05},  # [255, 130, 115]
        # {'field': light_fields[0], 'color': [30, 30, 115], 'id': 0.5, 'is': 0.05},  # [120, 255, 153]
        # {'field': light_fields[1], 'color': [255, 75, 89], 'id': 0.5, 'is': 0.05}, # red
        # {'field': light_fields[2], 'color': [64, 255, 76], 'id': 0.5, 'is': 0.05}, # green
        # {'field': light_fields[0], 'color': [65, 65, 255], 'id': 0.5, 'is': 0.05}, # blue
    ],
    
    'background_depth': cv2.resize(np.load(assets_path + 'bkg.npy').astype(np.float32), sim_size),
    'background_img': (cv2.cvtColor(cv2.imread(assets_path + '/bkg.png'), cv2.COLOR_RGB2BGR) / 255).astype(np.float32),
    'elastomer_thickness': 0.003,
    'min_depth': 0.008,
    'texture_sigma': 0.00001,
    'elastic_deformation': False,
    'rectify_fields': rectify_fields
})


## deformed depth map
depths = np.load(assets_path + 'bkg.npy') + 0.5
print("depths", depths)
print(depths.shape)

tactile_rgb = cv2.cvtColor(model.generate_bkg_image(cv2.resize(depths, sim_size)), cv2.COLOR_RGB2BGR)

tactile_rgb = np.array(tactile_rgb)
tactile_rgb = np.squeeze(tactile_rgb)
img_name = assets_path + '/bkg_rendered_9leds.png'
cv2.imwrite(img_name, tactile_rgb)

# cv2.imshow('tactile_rgb',tactile_rgb)
# cv2.waitKey(-1)
