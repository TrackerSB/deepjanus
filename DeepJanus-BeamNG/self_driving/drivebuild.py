from typing import Callable

from drivebuildclient.aiExchangeMessages_pb2 import SimulationID, VehicleID
from lxml.etree import _Element

from drivebuildclient.AIExchangeService import AIExchangeService


class AI:
    def __init__(self, service: AIExchangeService) -> None:
        from core.folders import folders
        from self_driving.beamng_config import BeamNGConfig
        from self_driving.nvidia_prediction import NvidiaPrediction
        from keras.models import load_model
        self._service = service
        config = BeamNGConfig()
        model_file = str(folders.trained_models_colab.joinpath(config.keras_model_file))
        model = load_model(model_file)
        self._prediction = NvidiaPrediction(model, config)

    def start(self, sid: SimulationID, vid: VehicleID, dynamic_callback: Callable[[], None]) -> None:
        from drivebuildclient.aiExchangeMessages_pb2 import SimStateResponse, DataRequest, Control
        from self_driving.simulation_data import SimulationDataRecord
        from PIL import Image
        import io
        request = DataRequest()
        request.request_ids.append("center_cam")
        simulation_data_records = []
        while True:
            sim_state = self._service.wait_for_simulator_request(sid, vid)
            dynamic_callback()
            if sim_state == SimStateResponse.SimState.RUNNING:
                # Request camera image
                byte_im = self._service.request_data(sid, vid, request).data["center_cam"].camera.color
                image = Image.open(io.BytesIO(byte_im))
                image.convert("RGB")

                # Determine last state
                # is_oob, oob_counter, max_oob_percentage, oob_distance = self._oob_monitor.get_oob_info(oob_bb=False, wrt="right")
                is_oob = False
                oob_counter = max_oob_percentage = oob_distance = None
                car_state = {"timer": None,
                             "damage": None,
                             "pos": None,
                             "dir": None,
                             "vel": None,
                             "gforces": None,
                             "gforces2": None,
                             "steering": None,
                             "steering_input": None,
                             "brake": None,
                             "brake_input": None,
                             "throttle": None,
                             "throttle_input": None,
                             "throttleFactor": None,
                             "engineThrottle": None,
                             "wheelspeed": None,
                             "vel_kmh": None
                             }
                sim_data_record = SimulationDataRecord(**car_state,
                                                       is_oob=is_oob,
                                                       oob_counter=oob_counter,
                                                       max_oob_percentage=max_oob_percentage,
                                                       oob_distance=oob_distance)
                simulation_data_records.append(sim_data_record)
                last_state = simulation_data_records[-1]
                if last_state.is_oob:
                    break

                # Compute control command
                steering_angle, throttle = self._prediction.predict(image, last_state)
                control = Control.AvCommand()
                control.steering_angle = steering_angle
                control.accelerate = throttle
                self._service.control(sid, vid, control)
            else:
                break

    @staticmethod
    def add_data_requests(ai_tag: _Element, participant_id: str) -> None:
        from lxml.etree import Element
        camera = Element("camera")
        camera.set("id", "center_cam")
        camera.set("direction", "FRONT")
        camera.set("width", "320")
        camera.set("height", "160")
        camera.set("fov", "120")
        ai_tag.append(camera)
