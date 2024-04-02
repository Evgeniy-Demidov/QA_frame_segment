from dotenv import load_dotenv
import os
import requests

load_dotenv()


FRAME_SEGMENT_GET_POLY_REPRS_NO_OWNER = os.getenv("FRAME_SEGMENT_GET_POLY_REPRS_NO_OWNER")
BASE_PATH = os.getenv("BASE_PATH")

cookies = {"session": os.getenv("SESSION_COOKIE")}

def get_frame_segment_info(frame_segment_id):
    url = FRAME_SEGMENT_GET_POLY_REPRS_NO_OWNER.format(frame_segment_id=frame_segment_id)
    response = requests.get(url, cookies=cookies)
    response.raise_for_status()
    return response.json()

def get_bbox_from_two_points(left_top, right_bottom):
    return [left_top, [left_top[0], right_bottom[1]], right_bottom, [right_bottom[0], left_top[1]]]

def get_data_from_bbox(left_top, right_bottom):
    test_geometry = {
        "geom_filter": {
            "type": "Polygon",
            "coordinates": [get_bbox_from_two_points(left_top, right_bottom)]
        },
        "geom_relation": "ST_INTERSECTS"
    }

    url = f"{BASE_PATH}/pci-api/v1/area/30?layers=viewpoints.uniform"
    response = requests.post(url, json=test_geometry, cookies=cookies)
    response.raise_for_status()
    return response.json()['features']

def update_status_frame_segment(frame_segment_id, status):
    url = f"{BASE_PATH}/pci-api/v1/frame_segment/{frame_segment_id}?status={status}"
    response = requests.patch(url, cookies=cookies)
    response.raise_for_status()
    return response.status_code