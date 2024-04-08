import os
import concurrent.futures
from concurrent.futures import Future
import threading
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import folium
import pandas as pd
from IPython.display import display, clear_output
from ipywidgets import HTML, Output, Image, Button, HBox, VBox, GridBox, Layout, Dropdown
from PIL import Image as PILImage


from data import get_frontal_image, make_image_with_distresses, get_roi, PCI_RATINGS
from platform_requests import update_status_frame_segment
from collections import OrderedDict


def with_processing_widget(func):

    def enable_processing_widget(*args, **kwargs):
        self = args[0]
        self.processing_widget.layout.visibility = 'visible'
        func_results = func(*args, **kwargs)
        self.processing_widget.layout.visibility = 'hidden'
        return func_results

    return enable_processing_widget


class CacheDict(OrderedDict):
    """Dict with a limited length, ejecting LRUs as needed."""

    def __init__(self, *args, cache_len: int = 10, **kwargs):
        assert cache_len > 0
        self.cache_len = cache_len

        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().move_to_end(key)

        while len(self) > self.cache_len:
            oldkey = next(iter(self))
            super().__delitem__(oldkey)

    def __getitem__(self, key):
        val = super().__getitem__(key)
        super().move_to_end(key)

        return val


class ImageLoader:
    def __init__(self, data: List):
        self.data = data
        self.preloaded_images = CacheDict(cache_len=15)
        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.processing_widget = Image(
            value='https://media.tenor.com/On7kvXhzml4AAAAj/loading-gif.gif'.encode('utf-8'),
            format='url', width=32, height=32
        )
        self.processing_widget.layout.visibility = 'hidden'

    @with_processing_widget
    def preload_images(self, index):
        start = max(0, index - 2)
        end = min(len(self.data), index + 10)
        for img_idx in range(start, end):
            if img_idx not in self.preloaded_images:
                self.preloaded_images[img_idx] = self.executor.submit(self.load_image, self.data[img_idx])
        
    @with_processing_widget
    def get_image(self, img_index):
        if img_index not in self.preloaded_images:
            self.preload_images(img_index)
        img_data = self.preloaded_images[img_index]
        if isinstance(img_data, Future):
            self.preloaded_images[img_index] = img_data.result()
        return self.preloaded_images[img_index]
    
    def load_image(self, frame_segment_info: dict):
        raw_img = get_frontal_image(frame_segment_info)
        distress_img = make_image_with_distresses(raw_img, frame_segment_info)
        roi_img = get_roi(frame_segment_info)
        return raw_img, distress_img, roi_img


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
    def __init__(self, data: List, image_loader: ImageLoader):
        self.image_widget = Image()
        self.data = data
        self.image_loader = image_loader

    def display(self, index: int, apply_distress: bool = True, apply_roi: bool = True):
        raw_img, distress_img, roi_img = self.image_loader.get_image(index)
        img = distress_img if apply_distress else raw_img
        if apply_roi:
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
        self.processed_button = Button(description='Processed')
        self.unprocessable_button = Button(description='Unprocessable')
        self.bad_frame_button = Button(description='Bad Frame')
        self.invalid_pci_stat_button = Button(description='Invalid PCI Stat')
        self.invalid_distress_segmentation_button = Button(description='Invalid Distress Segmentation')
        self.invalid_roi_button = Button(description='Invalid ROI')

        self.image_display = image_display
        self.image_display.pagination.on_next_button_click(self.disable_all)

        self.hidden_button.on_click(self.on_hidden_button_click)
        self.invalid_button.on_click(self.on_invalid_button_click)
        self.processed_button.on_click(self.on_processed_button_click)
        self.unprocessable_button.on_click(self.on_unprocessable_button_click)
        self.bad_frame_button.on_click(self.on_bad_frame_button_click)
        self.invalid_pci_stat_button.on_click(self.on_invalid_pci_stat_button_click)
        self.invalid_distress_segmentation_button.on_click(self.on_invalid_distress_segmentation_button_click)
        self.invalid_roi_button.on_click(self.on_invalid_roi_button_click)

    def disable_all(self, b=None):
        self.unprocessable_button.disabled = True
        self.bad_frame_button.disabled = True
        self.invalid_distress_segmentation_button.disabled = True
        self.invalid_roi_button.disabled = True

    def on_hidden_button_click(self, b):
        self.update_status('hidden')

    def on_invalid_button_click(self, b):
        self.update_status('invalid')

    def on_processed_button_click(self, b):
        self.update_status('processed')

    def on_unprocessable_button_click(self, b):
        pass

    def on_bad_frame_button_click(self, b):
        pass

    def on_invalid_pci_stat_button_click(self, b):
        pass

    def on_invalid_distress_segmentation_button_click(self, b):
        pass

    def on_invalid_roi_button_click(self, b):
        pass

    def update_status(self, status):
        frame_segment_id = self.data[self.image_display.index]['viewpoint']['properties']['id']
        update_status_frame_segment(frame_segment_id, status)
        self.image_display.display_data()

    def display(self):
        self.disable_all()
        main_buttons = HBox([self.processed_button, self.hidden_button, self.invalid_button])
        unprocessable_buttons = HBox([self.unprocessable_button, self.create_button('Concrete', self.enable_unprocessable), self.create_button('Paved', self.enable_unprocessable), self.create_button('Unpaved', self.enable_unprocessable)])
        bad_frame_buttons = HBox([self.bad_frame_button, self.create_button('Blur', self.enable_bad_frame), self.create_button('Glare', self.enable_bad_frame), self.create_button('Occlusions', self.enable_bad_frame), self.create_button('Other', self.enable_bad_frame)])
        distress_segmentation_buttons = HBox([self.invalid_distress_segmentation_button, self.create_button('Low', self.enable_distress_segmentation), self.create_button('Medium', self.enable_distress_segmentation), self.create_button('High', self.enable_distress_segmentation)])
        roi_buttons = HBox([self.invalid_roi_button, self.create_button('Mask', self.enable_roi), self.create_button('Area', self.enable_roi)])
        pci_stat_button = HBox([self.invalid_pci_stat_button])
        
        return VBox([main_buttons, unprocessable_buttons, bad_frame_buttons, distress_segmentation_buttons, roi_buttons, pci_stat_button])

    def create_button(self, description, on_click):
        button = Button(description=description)
        button.on_click(self.disable_all)  # Disable all buttons when this button is clicked
        button.on_click(on_click)  # Then call the original on_click event
        return button

    def enable_unprocessable(self, b):
        self.unprocessable_button.disabled = False

    def enable_bad_frame(self, b):
        self.bad_frame_button.disabled = False

    def enable_distress_segmentation(self, b):
        self.invalid_distress_segmentation_button.disabled = False

    def enable_roi(self, b):
        self.invalid_roi_button.disabled = False

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
        html_widget.layout.height = '300px'
        html_widget.layout.width = '100%'
        return html_widget


class ImageDisplay:
    def __init__(self, data, pipeline_id):
        self.data = data
        self.index = 0
        self.status_bar = StatusBar(data, self.index) 
        self.status_bar_output = Output() 

        self.image_loader = ImageLoader(data)
        self.image_loader.preload_images(self.index)  # Preload images
        self.pagination = Pagination(data, self.display_data)
        self.image_widget = ImageWidget(data, self.image_loader) 
        self.stats_widget = StatsWidget(data)
        self.mask = Mask(pipeline_id, data, self.image_widget, self)
        self.qa_status = QAStatus(data, self)

    def display_data(self, apply_distress=True, apply_roi=False):
        self.index = self.pagination.index
        self.image_loader.preload_images(self.index)  # Preload images
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
            HBox([self.pagination.display(), self.mask.display(), self.image_loader.processing_widget]),
            HBox([self.qa_status.display()]),
            self.stats_widget.display(self.index)
        ]))
        self.display_data()