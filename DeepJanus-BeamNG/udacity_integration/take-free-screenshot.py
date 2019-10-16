import random

import numpy as np

from beamng_brewer import BeamNGBrewer
from beamng_waypoint import BeamNGWaypoint
from road_storage import RoadStorage
from simulation_data_collector import SimulationDataCollector
from beamng_tig_maps import maps

random.seed(42)
np.random.seed(42)

maps.install_map_if_needed()


def get_node_coords(node):
    return node[0], node[1], node[2]


def distance(p1, p2):
    return np.linalg.norm(np.subtract(get_node_coords(p1), get_node_coords(p2)))


nodes = RoadStorage().get_road_nodes_by_index(2)

brewer = BeamNGBrewer(road_nodes=nodes)
waypoint_goal = BeamNGWaypoint('waypoint_goal', nodes[40][:3])
maps.beamng_map.generated().write_items(brewer.decal_road.to_json() + '\n' + waypoint_goal.to_json())
brewer.vehicle_start_pose = brewer.road_points.vehicle_start_pose(road_point_index=0)

vehicle = brewer.setup_vehicle()
brewer.setup_scenario_camera()
beamng = brewer.beamng
steps = brewer.params.beamng_steps
sim_data_collector = SimulationDataCollector(vehicle, beamng, brewer.decal_road, brewer.params)

brewer.bring_up()
ai_aggression = None  # 1.0
sim_save_path = 'screenshot'

if ai_aggression:
    vehicle.ai_set_aggression(ai_aggression)
    vehicle.ai_drive_in_lane(True)
    vehicle.ai_set_waypoint(waypoint_goal.name)
else:
    vehicle.ai_set_mode("disabled")


def start():
    for idx in range(1000):
        sim_data_collector.collect_current_data()
        last_state = sim_data_collector.states[-1]
        if distance(last_state.pos, waypoint_goal.position) < 15.0:
            pass

        def shot(dir, h=-25):
            brewer.camera.pose.pos = tuple(last_state.pos[:2]) + (h,)
            brewer.camera.pose.rot = dir
            brewer.camera.get_rgb_image().save(f'shot{dir}_h{h}.png')

        shot((0, 0, -90), -25)
        shot((0, 0, -90), -20)
        shot((0, 0, -90), -15)
        # brewer.camera.resolution = (800,600)
        break
        beamng.step(steps)


try:
    start()
finally:
    sim_data_collector.save()
    beamng.close()
