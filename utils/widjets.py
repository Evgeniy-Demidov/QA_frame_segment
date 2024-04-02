import os
import concurrent.futures
import threading
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import folium
import pandas as pd
from IPython.display import display, clear_output
from ipywidgets import HTML, IntProgress, Output, Image, Button, HBox, VBox
from PIL import Image as PILImage

from data import get_frontal_image, make_image_with_distresses, get_roi, PCI_RATINGS
from platform_requests import update_status_frame_segment


class ImageLoader:
    def __init__(self, data: List, pipeline_id: str):
        self.data = data
        self.p = Path('../data') / pipeline_id
        os.makedirs(self.p, exist_ok=True)
        self.preloaded_images = {}  
        self.progress = IntProgress(min=0, max=len(data))  

    def preload_images(self):
        display(self.progress)  
        with concurrent.futures.ThreadPoolExecutor() as executor:
            self.preloaded_images.update(dict(executor.map(self.save_image_to_disk, enumerate(self.data[:10]))))
            threading.Thread(target=self._preload_remaining_images).start()

    def _preload_remaining_images(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            self.preloaded_images.update(dict(executor.map(self.save_image_to_disk, enumerate(self.data[10:], start=10))))

    def save_image_to_disk(self, data: Tuple[int, dict]):
        index, frame_segment_info = data
        dis_img_path = self.p / f'dis_image_{index}.jpg'
        raw_img_path = self.p / f'raw_image_{index}.jpg'
        roi_img_path = self.p / f'roi_image_{index}.jpg'

        # Check if the images already exist
        if not dis_img_path.exists() or not raw_img_path.exists() or not roi_img_path.exists():
            raw_img, distress_img, roi_img = self.load_image(frame_segment_info)
            img_dis = cv2.cvtColor(distress_img, cv2.COLOR_BGR2RGB)
            img_raw = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            img_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(dis_img_path), img_dis)
            cv2.imwrite(str(raw_img_path), img_raw)
            cv2.imwrite(str(roi_img_path), img_roi)

        self.progress.value += 1 
        return index, (dis_img_path, raw_img_path, roi_img_path)

    def load_image(self, frame_segment_info: dict) -> Tuple[PILImage.Image, PILImage.Image]:
        raw_img = get_frontal_image(frame_segment_info)
        distress_img = make_image_with_distresses(raw_img, frame_segment_info)
        roi_img = get_roi(frame_segment_info)
        return (raw_img, distress_img, roi_img)

class Pagination:
    def __init__(self, data, on_click):
        self.data = data
        self.index = 0
        self.next_button = Button(description='Next')
        self.prev_button = Button(description='Previous')
        self.on_click = on_click

        self.next_button.on_click(self.on_next_button_click)
        self.prev_button.on_click(self.on_prev_button_click)

    def on_next_button_click(self, b):
        self.index = min(self.index + 1, len(self.data) - 1)
        self.on_click()

    def on_prev_button_click(self, b):
        self.index = max(self.index - 1, 0)
        self.on_click()

    def display(self):
        return HBox([self.prev_button, self.next_button])

class ImageWidget:
    def __init__(self, pipeline_id: str, data: List, image_loader: ImageLoader):
        self.image_widget = Image()
        self.p = Path('../data') / pipeline_id
        self.data = data
        self.image_loader = image_loader

    def display(self, index: int, apply_distress: bool = True, apply_roi: bool = True):
        while index not in self.image_loader.preloaded_images:
            time.sleep(0.1) 

        dis_img_path, raw_img_path, roi_img_path = self.image_loader.preloaded_images[index]
        img_path = dis_img_path if apply_distress else raw_img_path
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to read image from file: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        if apply_roi:
            roi_img = cv2.imread(str(roi_img_path), cv2.IMREAD_GRAYSCALE)
            if roi_img is None:
                raise ValueError(f"Failed to read ROI image from file: {roi_img_path}")
            mask = cv2.bitwise_not(roi_img) 
            img = cv2.bitwise_and(img, img, mask=mask)  
        _, img_encoded = cv2.imencode('.jpg', img)
        self.image_widget.value = img_encoded.tobytes()  # Use tobytes() instead of tostring()
        return self.image_widget

        
class StatsWidget:
    def __init__(self, data):
        self.data = data
        self.stats_widget = Output()

    def get_df_stats(self, frame_segment_data):
        return pd.DataFrame(pd.Series(frame_segment_data['viewpoint']['properties']['stats'])).T

    def get_df_distresses(self, frame_segment_data):
        return pd.DataFrame(frame_segment_data['distresses'])[['type', 'severity', 'measurements']] if frame_segment_data['distresses'] else None

    def display(self, index):
        with self.stats_widget:
            clear_output(wait=True)
            display(self.get_df_stats(self.data[index]))
            display(self.get_df_distresses(self.data[index]))
        return self.stats_widget

class Mask:
    def __init__(self, pipeline_id: str, data: List, image_widget: ImageWidget, image_display: 'ImageDisplay'):
        self.p = Path('../data') / pipeline_id
        self.data = data
        self.image_widget = image_widget
        self.image_display = image_display
        self.distress_button = Button(description='Distress on')
        self.roi_button = Button(description='ROI off')  # Create the ROI button

        self.distress_button.on_click(self.on_distress_button_click)
        self.roi_button.on_click(self.on_roi_button_click)  # Set the on_click event for the ROI button

    def on_distress_button_click(self, b):
        if b.description == 'Distress on':
            self.image_display.display_data(apply_distress=False)  
            b.description = 'Distress off'
        else:
            self.image_display.display_data(apply_distress=True) 
            b.description = 'Distress on'

    def on_roi_button_click(self, b):  # Define the on_click event for the ROI button
        if b.description == 'ROI on':
            self.image_display.display_data(apply_roi=False)  
            b.description = 'ROI off'
        else:
            self.image_display.display_data(apply_roi=True) 
            b.description = 'ROI on'

    def display(self):
        return HBox([self.distress_button, self.roi_button])  # Return both buttons

class QAStatus:
    def __init__(self, data: List, image_display: 'ImageDisplay'):
        self.data = data
        self.image_display = image_display
        self.hidden_button = Button(description='Hidden')
        self.invalid_button = Button(description='Invalid')
        self.processed_button = Button(description='Processed')  # Add the "Processed" button

        self.hidden_button.on_click(self.on_hidden_button_click)
        self.invalid_button.on_click(self.on_invalid_button_click)
        self.processed_button.on_click(self.on_processed_button_click)  # Set the on_click event for the "Processed" button

    def on_hidden_button_click(self, b):
        frame_segment_id = self.data[self.image_display.index]['viewpoint']['properties']['id']
        update_status_frame_segment(frame_segment_id, 'hidden')
        self.image_display.display_data()

    def on_invalid_button_click(self, b):
        frame_segment_id = self.data[self.image_display.index]['viewpoint']['properties']['id']
        update_status_frame_segment(frame_segment_id, 'invalid')
        self.image_display.display_data()

    def on_processed_button_click(self, b):  
        frame_segment_id = self.data[self.image_display.index]['viewpoint']['properties']['id']
        update_status_frame_segment(frame_segment_id, 'processed')
        self.image_display.display_data()

    def display(self):
        return HBox([self.hidden_button, self.invalid_button, self.processed_button])  


class StatusBar:
    def __init__(self, data: List, index: int):
        self.data = data
        self.index = index
        self.current_coordinates = self.data[self.index]['viewpoint']['geometry']['coordinates']
        self.map = self.generate_map()

    def generate_map(self):
        m = folium.Map(location=[self.current_coordinates[1], self.current_coordinates[0]], zoom_start=16, tiles='cartodbpositron')
        for index, frame_segment_info in enumerate(self.data):
            coordinates = frame_segment_info['viewpoint']['geometry']['coordinates']
            rating = frame_segment_info['viewpoint']['properties']['stats']['rating']
            status = frame_segment_info['viewpoint']['properties']['status']
            if status == 'hidden':
                color = 'gray'
            elif status == 'invalid':
                color = 'red'
            else:
                color = PCI_RATINGS[rating][1]

            folium.CircleMarker(
                location=[coordinates[1], coordinates[0]],
                radius=5, 
                color=color, 
                fill=True, 
                fill_color=color,
                popup=f'Index: {index}, Status: {status}',  # Add popup
            ).add_to(m)
        return m

    def generate_current_marker(self):
        current_marker = folium.map.FeatureGroup()
        folium.CircleMarker(
            location=[self.current_coordinates[1], self.current_coordinates[0]],
            radius=5, 
            color='blue', 
            fill=True, 
            fill_color='blue',
        ).add_to(current_marker)
        self.map.add_child(current_marker)
        return current_marker

    def update_current_marker(self, index):
        self.index = index
        self.current_coordinates = self.data[self.index]['viewpoint']['geometry']['coordinates']
        self.map = self.generate_map()  # Regenerate the map
        self.current_marker = self.generate_current_marker()

    def display(self):
        self.update_current_marker(self.index)
        html_widget = HTML(self.map._repr_html_())
        html_widget.layout.height = '400px'
        html_widget.layout.width = '100%'
        return html_widget


class ImageDisplay:
    def __init__(self, data, pipeline_id):
        self.data = data
        self.p = Path('../data') / pipeline_id
        self.index = 0
        self.status_bar = StatusBar(data, self.index) 
        self.status_bar_output = Output() 

        self.image_loader = ImageLoader(data, pipeline_id)
        self.image_loader.preload_images()  # Preload images
        self.pagination = Pagination(data, self.display_data)
        self.image_widget = ImageWidget(pipeline_id, data, self.image_loader) 
        self.stats_widget = StatsWidget(data)
        self.mask = Mask(pipeline_id, data, self.image_widget, self)
        self.qa_status = QAStatus(data, self)

    def display_data(self, apply_distress=True, apply_roi=False):
        self.index = self.pagination.index 
        self.image_widget.display(self.index, apply_distress, apply_roi)
        self.stats_widget.display(self.index)
        with self.status_bar_output:
            clear_output(wait=True)  
            display(self.status_bar.display())  
        self.status_bar.update_current_marker(self.index)    

    def display(self):
        clear_output(wait=True)  
        display(VBox([
            self.status_bar_output,  
            self.image_widget.display(self.index),
            self.stats_widget.display(self.index),
            HBox([self.pagination.display(), self.mask.display()]),  
            HBox([self.qa_status.display()]) 
        ]))
        self.display_data()