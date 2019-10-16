from random import randint
from typing import List, Tuple

import math
from shapely.geometry import Point

from self_driving.beamng_member import BeamNGMember
from self_driving.catmull_rom import catmull_rom
from self_driving.road_bbox import RoadBoundingBox
from self_driving.road_polygon import RoadPolygon
from self_driving.beamng_road_visualizer import plot_road_polygon, plot_road_bbox

Tuple4F = Tuple[float, float, float, float]
Tuple2F = Tuple[float, float]


class RoadGenerator:
    """A class that is able to generate random roads given some parameters, such as
    the number of segments."""

    # Initial setup
    #num_control_nodes = 15
    #INITIAL_SEGMENTS_MAX_ANGLE = 90
    #NUM_INITIAL_SEGMENTS_THRESHOLD = 4
    #NUM_UNDO_ATTEMPTS = 20
    #max_angle=360
    #seg_length=50
    #num_spline_nodes=20
    #START_ANGLE = 45


    START_ANGLE = 60
    INITIAL_SEGMENTS_MAX_ANGLE = 60
    #MAX_ANGLE = 160
    MAX_ANGLE = 60

    NUM_SPLINE_NODES = 20
    NUM_INITIAL_SEGMENTS_THRESHOLD = 4
    NUM_UNDO_ATTEMPTS = 20
    SEG_LENGTH = 25

    def __init__(self, num_control_nodes=15, max_angle=MAX_ANGLE, seg_length=SEG_LENGTH,
                 num_spline_nodes=NUM_SPLINE_NODES, initial_node=(0.0, 0.0, -28.0, 8.0),
                 bbox_size=(-250, 0, 250, 500)):
        assert num_control_nodes > 1 and num_spline_nodes > 0
        assert 0 <= max_angle <= 360
        assert seg_length > 0
        assert len(initial_node) == 4 and len(bbox_size) == 4
        self.num_control_nodes = num_control_nodes
        self.num_spline_nodes = num_spline_nodes
        self.initial_node = initial_node
        self.max_angle = max_angle
        self.seg_length = seg_length
        self.road_bbox = RoadBoundingBox(bbox_size)
        assert not self.road_bbox.intersects_vertices(self._get_initial_point())
        assert self.road_bbox.intersects_sides(self._get_initial_point())

    def generate_control_nodes(self, visualise=False, attempts=NUM_UNDO_ATTEMPTS) -> List[Tuple4F]:
        while True:
            nodes = [self._get_initial_control_node(), self.initial_node]

            # i is the number of valid generated control nodes.
            i = 0

            # When attempt >= attempts and the skeleton of the road is still invalid,
            # the construction of the skeleton starts again from the beginning.
            # attempt is incremented every time the skeleton is invalid.
            attempt = 0

            while i < self.num_control_nodes:
                nodes.append(self._get_next_node(nodes[-1], self._get_next_max_angle(i)))
                road_polygon = RoadPolygon.from_nodes(nodes)

                if visualise:
                    fig = plot_road_bbox(self.road_bbox)
                    plot_road_polygon(road_polygon, title="RoadPolygon i=%s" % i, fig=fig)

                # budget is the number of iterations used to attempt to add a valid next control node
                # before also removing the previous control node.
                budget = self.num_control_nodes - i
                assert budget >= 1

                is_valid = road_polygon.is_valid()
                while not is_valid and budget > 0:
                    nodes.pop()
                    budget = budget - 1
                    attempt += 1

                    nodes.append(self._get_next_node(nodes[-1], self._get_next_max_angle(i)))
                    road_polygon = RoadPolygon.from_nodes(nodes)
                    is_valid = road_polygon.is_valid()

                    if visualise:
                        fig = plot_road_bbox(self.road_bbox)
                        plot_road_polygon(road_polygon, title="RoadPolygon i=%s" % i, fig=fig)

                if not is_valid:
                    assert budget == 0
                    nodes.pop()
                    if len(nodes) > 2:
                        nodes.pop()
                        i -= 1
                else:
                    i += 1

                assert RoadPolygon.from_nodes(nodes).is_valid()
                assert 0 <= i <= self.num_control_nodes

                if i >= 2 and self.road_bbox.intersects_boundary(road_polygon.polygons[-1]):
                    break

                if attempt >= attempts:
                    break

            if attempt < attempts:
                return nodes

    def generate(self, visualise=False) -> BeamNGMember:
        control_nodes = self.generate_control_nodes(visualise)
        sample_nodes = catmull_rom(control_nodes, self.num_spline_nodes)
        road = BeamNGMember(control_nodes, sample_nodes, self.num_spline_nodes, self.road_bbox)
        while not road.is_valid():
            control_nodes = self.generate_control_nodes(visualise)
            sample_nodes = catmull_rom(control_nodes, self.num_spline_nodes)
            road = BeamNGMember(control_nodes, sample_nodes, self.num_spline_nodes, self.road_bbox)
        return road

    def _get_initial_point(self) -> Point:
        return Point(self.initial_node[0], self.initial_node[1])

    def _get_initial_control_node(self) -> Tuple4F:
        init_max_angle = self._get_next_max_angle(0)
        x, y, z, width = self._get_next_node(self.initial_node, init_max_angle, start_angle=240)
        #x, y, z, width = self._get_next_node(self.initial_node, self.max_angle)
        while self.road_bbox.bbox.contains(Point(x, y)):
            #x, y, _, _ = self._get_next_node(self.initial_node, self.max_angle)
            x, y, _, _ = self._get_next_node(self.initial_node, init_max_angle, start_angle=240)
        return x, y, z, width

    def _get_next_node(self, initial_node: Tuple4F, max_angle, start_angle=START_ANGLE) -> Tuple4F:
        angle = randint(start_angle, max_angle + start_angle)
        x0, y0, z0, width0 = initial_node
        x1, y1 = self._get_next_xy(x0, y0, angle)
        return x1, y1, z0, width0

    def _get_next_xy(self, x0: float, y0: float, angle: float) -> Tuple2F:
        angle_rad = math.radians(angle)
        return x0 + self.seg_length * math.cos(angle_rad), y0 + self.seg_length * math.sin(angle_rad)

    def _get_next_max_angle(self, i: int, threshold=NUM_INITIAL_SEGMENTS_THRESHOLD) -> float:
        if i < threshold:
            return self.INITIAL_SEGMENTS_MAX_ANGLE
        else:
            return self.max_angle


if __name__ == "__main__":
    NODES = 10
    #MAX_ANGLE = 130
    MAX_ANGLE = 60
    NUM_SPLINE_NODES = 20
    SEG_LENGTH = 25

    road = RoadGenerator(num_control_nodes=NODES, max_angle=MAX_ANGLE, seg_length=SEG_LENGTH,
                 num_spline_nodes=NUM_SPLINE_NODES).generate(visualise=True)
    from beamng_road_visualizer import plot_road

    plot_road(road, save=True)
