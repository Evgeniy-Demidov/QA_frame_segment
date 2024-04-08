from dotenv import load_dotenv
import os
import requests
import numpy as np
from io import BytesIO
from PIL import Image
import shapely
import shapely.wkt
from shapely.geometry import Point
import cv2
import folium
from folium.plugins import Draw
import concurrent.futures
from platform_requests import get_frame_segment_info


# Load environment variables
load_dotenv()

PCI_RATINGS: dict[str, str] = {
    'Failed': ((0, 10), '#9a1914'),
    'Serious': ((10, 25), '#ad3631'),
    'Inferior':((25, 40), '#da615d'),
    'VeryPoor':((25, 40), '#da615d'),
    'Poor':((40, 55), '#e69d59'),
    'Fair':((55, 70), '#ebca62'),
    'Satisfactory':((70, 85), '#89c458'),
    'Good':((85, 100), '#489758'),
}

cookies = {"session": os.getenv("SESSION_COOKIE")}
gbucket_images_base_path = os.getenv("GBUCKET_IMAGES_BASE_PATH")
owner_id = os.getenv("OWNER_ID")


def hex_to_rgb(hex_str):
    if hex_str.startswith("#"):
        hex_str = hex_str[1:]
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

def get_image(image_path):
    image_response = requests.get(gbucket_images_base_path + '/' + image_path)
    image = Image.open(BytesIO(image_response.content))
    return np.asarray(image)[..., ::-1]

def get_frontal_image(frame_segment_info):
    return get_image(frame_segment_info['images']['front']['img'])

def get_roi(frame_segment_info):
    return get_image(frame_segment_info['images']['front']['roi'])

def get_bev_image(frame_segment_info):
    return get_image(frame_segment_info['images']['bev']['img'])

def draw_distress(image, distress_info, frame_segment_color):
    distress_poly = shapely.wkt.loads(distress_info['reprs']['front']['poly'])
    image_with_distress = image.copy()
    if isinstance(distress_poly, shapely.geometry.GeometryCollection):
        for geom in distress_poly.geoms:
            geom_coords = get_coordinates(geom, image.shape)
            cv2.drawContours(image, [geom_coords], -1, frame_segment_color, -1)
    else:
        geom_coords = get_coordinates(distress_poly, image.shape)
        cv2.drawContours(image, [geom_coords], -1, frame_segment_color, -1)
    return cv2.addWeighted(image, alpha=0.5, src2=image_with_distress, beta=0.5, gamma=0)

def get_coordinates(geom, image_shape):
    geom_coords = shapely.get_coordinates(geom)
    geom_coords *= np.array(image_shape[:2][::-1])
    return np.array(geom_coords).astype(np.int32)

def make_image_with_distresses(image, frame_segment_info):
    if len(frame_segment_info['distresses']) == 0:
        return image
    frame_segment_rating = frame_segment_info['viewpoint']['properties']['stats']['rating']
    frame_segment_color = tuple(reversed(hex_to_rgb(PCI_RATINGS[frame_segment_rating][1])))
    image_with_distresses = image.copy()
    for distress_info in frame_segment_info['distresses']:
        image_with_distresses = draw_distress(image_with_distresses, distress_info, frame_segment_color)
    return image_with_distresses


def get_list_pipelines_from_bbox(data):
    return list(set([item['properties']['pipeline_id'] for item in data]))

def filter_data_by_pipeline(pipeline_id, data):
    return [item for item in data if item['properties']['pipeline_id'] == pipeline_id]

def get_list_coordinates(data):
    return [item['geometry']['coordinates'] for item in data]


def folium_map(data):
    coordinates = get_list_coordinates(data)   
    center = np.mean(coordinates, axis=0)
    m = folium.Map(location=[center[1], center[0]], zoom_start=14, tiles="cartodbpositron")

    for item in data:
        coordinate = item['geometry']['coordinates']
        status = item['properties']['status']
        rating = item['properties']['stats']['rating']
        pipeline_id = item['properties']['pipeline_id']
        
        # Choose color based on status
        if status == 'processed':
            color = PCI_RATINGS[rating][1]
        elif status == 'hidden':
            color = 'gray'
        else:
            color = 'white'
        
        folium.CircleMarker(
            location=[coordinate[1], coordinate[0]], 
            radius=5, 
            color=color, 
            fill=True, 
            fill_color=color, 
            popup=f'Pipeline ID: {pipeline_id}' 
        ).add_to(m)
    draw = Draw(export=True, filename='data.geojson')
    draw.add_to(m)

    return m

def filter_data_by_polygons(polygon_df, data):
    return [item for item in data if any(poly.contains(Point(item['geometry']['coordinates'])) for poly in polygon_df.geometry)]

def get_frame_segments_data(filtered_data):
    max_workers = 3  # Set a limit on the number of concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        frame_segments_data = list(executor.map(get_frame_segment_info, (item['properties']['id'] for item in filtered_data)))
    return frame_segments_data