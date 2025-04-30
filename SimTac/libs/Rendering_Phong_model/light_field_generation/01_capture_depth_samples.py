## notes: set 6 geltip in Mujuco simulation environment and output the depth image captured by the set camera.
## notes: the output file is 0.npy....6.npy
## notes: The model of Geltip is input through 'from experimental_setup.geltip.geltip import GelTip'


import cv2
from yarok import Platform, PlatformMJC, ConfigBlock, Injector, component

import os
import numpy as np

from experimental_setup.geltip.geltip import GelTip
from sim_model.utils.vis_img import to_normed_rgb, to_panel


@component(
    components=[
        GelTip
    ],
    # language=xml
    template="""
        <mujoco>
            <visual>
                <!-- important for the GelTips, to ensure its camera frustum captures the close-up elastomer -->
                <map znear="0.001" zfar="50"/>
            </visual>

            <asset>
                <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2"
                         width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>    
                <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>

                <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb2="0 0 0" rgb1="1 1 1"/>
                <material name="grid" texture="grid" texrepeat="2 2" texuniform="true" reflectance=".6"/>

            </asset>
            <worldbody>
                <light directional="true" diffuse=".4 .4 .4" specular="0.1 0.1 0.1" pos="0 0 5.0" dir="0 0 -1"/>
                <light directional="true" diffuse=".6 .6 .6" specular="0.2 0.2 0.2" pos="0 0 4" dir="0 0 -1"/>

                <body name="floor">
                    <geom name="ground" type="plane" size="0 0 1" pos="0 0 0" quat="1 0 0 0" material="matplane" condim="1"/>
                </body>
    

                <body pos="0.0 0.0 0.0" xyaxes='-1 0 0 0 0 1'>
                    <geltip name="geltip1"/>
                </body>

            

            </worldbody>        
        </mujoco>
    """
)
class GelTipWorld:
    pass


class CaptureDepthSampleBehaviour:

    def __init__(self,
                 injector: Injector,
                 config: ConfigBlock,
                 pl: Platform):
        self.sensors = [injector.get('geltip' + str(i)) for i in range(1, 2)]
        self.config = config
        self.pl = pl

    # Function to capture and save the depth map
    def save_depth_frame(self, geltip, key):
        frame_path = self.config['assets_path'] + '/' + 'bkg'
        with open(frame_path + '.npy', 'wb') as f:
            depth_frame = geltip.read_depth()
            np.save(f, depth_frame)
        return depth_frame

    def on_start(self):
        self.pl.wait_seconds(5)
        # 'bkg' if i == 0 else 'depth_' +
        frames = [

            to_normed_rgb(self.save_depth_frame(g,  str(i)))
            for i, g in enumerate(self.sensors)
        ]

        #cv2.imshow('frames', to_panel(frames))
        #cv2.waitKey(-1)


if __name__ == '__main__':
    __location__ = os.path.dirname(os.path.abspath(__file__))
    Platform.create({
        'world': GelTipWorld,
        'behaviour': CaptureDepthSampleBehaviour,
        'defaults': {
            'environment': 'sim',
            'behaviour': {
                'assets_path': os.path.join(__location__, 'experimental_setup/geltip/sim_assets')
            },
            'components': {
                '/geltip1': {'label_color': '0 0 1'},
            },
            'plugins': [
                # (Cv2Inspector, {})
            ]
        },
        'environments': {
            'sim': {
                'platform': {
                    'class': PlatformMJC
                },
                'inspector': False
            },
        },
    }).run()