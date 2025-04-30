
from yarok import ConfigBlock, component, interface
from yarok.platforms.mjc import InterfaceMJC

import numpy as np
import cv2
import os

from time import time

from sim_model.model_gpu import SimulationModel
from sim_model.utils.camera import circle_mask

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

@interface(
    defaults={
        'frame_size': (480, 640),
        'field_size': (120, 160),
        'field_name': 'geodesic',
        'elastic_deformation': False,
        'texture_sigma': 0.000005,
        'ia': 0.8,
        'fov':100,
        'light_constants': [
            {'color': [196, 94, 255], 'id': 0.5, 'is': 0.1},  # red # [108, 82, 255]
            {'color': [154, 144, 255], 'id': 0.5, 'is': 0.1},  # green # [255, 130, 115]
            {'color': [104, 175, 255], 'id': 0.5, 'is': 0.1},  # blue  # [120, 255, 153]
        ],
    }
)
class GelTipInterfaceMJC:

    def __init__(self,
                 interface_mjc: InterfaceMJC,
                 config: ConfigBlock):
        self.interface = interface_mjc
        self.frame_size = config['frame_size']
        
        # here we cancel the mask since the RoTip does not need a mask here.
        #self.mask = circle_mask(self.frame_size)
        self.mask = None
        
        n_lights = len(config['light_constants'])

        bkg_zeros = np.zeros(self.frame_size[::-1] + (3,), dtype=np.float32)

        try:
            cloud, fields = SimulationModel.load_assets(
                os.path.join(__location__, '../../sim_model/assets/'),
                config['field_size'],
                self.frame_size,
                config['field_name'],
                n_lights
            )

            self.model = SimulationModel(**{
                'ia': config['ia'],
                'fov': config['fov'],
                'light_sources': [{
                    'field': fields[l],
                    **config['light_constants'][l]}
                    for l in range(n_lights)
                ],
                'background_depth': np.zeros(self.frame_size),
                'cloud_map': cloud,
                'background_img': bkg_zeros,  # bkg_rgb if use_bkg_rgb else
                'texture_sigma': config['texture_sigma'],
                'elastic_deformation': config['elastic_deformation']
            })
            self.last_update = 0
        except:
            print('[warning] failed to load simulation model')

    def read(self):
        t = time()
        if self.last_update > t - 1.0:
            return self.tactile
        self.last_update = t
        depth = self.read_depth()
        self.tactile = self.model.generate(depth) \
            .astype(np.uint8)
        return self.tactile

    def read_depth(self):
        return self.interface.read_camera('camera', self.frame_size, depth=True, rgb=False)


@interface()
class GelTipInterfaceHW:

    def __init__(self, config: ConfigBlock):
        self.cap = cv2.VideoCapture(config['cam_id'])
        if not self.cap.isOpened():
            raise Exception('GelTip cam ' + str(config['cam_id']) + ' not found')

        self.fake_depth = np.zeros((640, 480), np.float32)

    def read(self):
        [self.cap.read() for _ in range(10)]  # skip frames in the buffer.
        ret, frame = self.cap.read()
        return frame

    def read_depth(self):
        return self.fake_depth


@component(
    tag="geltip",
    defaults={
        'interface_mjc': GelTipInterfaceMJC,
        'interface_hw': GelTipInterfaceHW,
        'probe': lambda c: {'camera': c.read()},
        'label_color': '255 255 255'
    },
    
    template="""
        <mujoco>
            <asset>
                <material name="glass_material" rgba="1 1 1 0.1"/>
                <material name="white_elastomer" rgba="0.8 0.8 0.8 1"/>
                <material name="black_plastic" rgba="1 1 1 1"/>
                <material name="label_color" rgba="${label_color} 1.0"/>
                
                <!-- inverted mesh, for limiting the depth map-->
                <mesh name="membrane_shell" file="sim_assets/CAT_PAWS_Depth_Map.STL" scale="0.001 0.001 0.001"/>  
                
            </asset>
            <worldbody>
                <body name="geltip">
                    
                    <geom density="0.1" type="mesh" mesh="membrane_shell" material="white_elastomer" pos="0 0 0"/>
                    <camera name="camera" pos="0 0 0.008" zaxis="0 0 -1" fovy="100" />
                     
                </body>
            </worldbody>
        </mujoco>
    """
)
class GelTip:

    def __init__(self):
        """
            Geltip driver as proposed in
            https://danfergo.github.io/geltip-simulation/

        """
        pass

    def read(self):
        pass

    def read_depth(self):
        pass
