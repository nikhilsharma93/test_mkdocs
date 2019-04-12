from datetime import datetime
from functools import reduce
import os

import cv2
import numpy as np


class BlankHolder:
    pass


class GraphPlot:
    def __init__(self, path):
        self._out_path = path
        self._pipeline_dict = []
        self._base_pipeline_ht = 300
        self._base_pipeline_wd = 1400
        self._base_component_ht = 200
        self._base_component_wd = 1100
        self._base_pipeline_ht_buffer = 100
        self._base_component_arrow_ht = 150
        self._base_pipeline_arrow_ht = 300
        self._total_buffer_wd = 50
        self._total_buffer_ht = 100
        self._max_pipeline_name_len = 25
        self._max_component_name_len = 20
        self._component_bg_color = (255,245,245)
        self._pipeline_bg_color = (249,244,245)
        self._component_name_font = cv2.FONT_HERSHEY_DUPLEX
        self._pipeline_name_font = cv2.FONT_HERSHEY_COMPLEX
        self._component_text_font = cv2.FONT_HERSHEY_SIMPLEX
        self._pipeline_text_font = cv2.FONT_HERSHEY_SIMPLEX
        self._logo = cv2.imread(os.path.join(os.path.dirname(__file__), '_aux/logo.png'), -1)
        self._logo = cv2.resize(self._logo, (50,50))

    def add_pipeline(self, inp_dict):
        self._pipeline_dict.append(inp_dict)

    def plot(self):
        num_pipelines = len(self._pipeline_dict)
        # Get total ht and wd
        total_wd = self._base_pipeline_wd + 2 * self._total_buffer_wd
        # Add buffers
        total_ht = 2 * self._total_buffer_ht
        # Add arrow heights
        total_ht += (num_pipelines - 1) * self._base_pipeline_arrow_ht
        # Now get ht from each pipeline
        for pipeline_dict in self._pipeline_dict:
            new_pipeline_ht, _ = self._get_new_pipeline_dims(len(pipeline_dict['component_names']))
            total_ht += new_pipeline_ht

        img = 255 * np.ones((total_ht, total_wd, 3), dtype='uint8')
        start_y = self._total_buffer_ht
        arrow_start_x = int((total_wd/2) - 5)
        for iter_pipeline, pipeline_dict in enumerate(self._pipeline_dict):
            pipeline_img = self._new_pipeline(pipeline_dict)
            pipeline_img_ht, pipeline_img_wd = pipeline_img.shape[:2]
            start_x = int((total_wd/2 - pipeline_img_wd/2))
            img[start_y:start_y+pipeline_img_ht, start_x:start_x+pipeline_img_wd] = pipeline_img.copy()
            if iter_pipeline == num_pipelines - 1:
                break
            # Add arrow
            start_y += pipeline_img_ht
            cv2.arrowedLine(img, (arrow_start_x, start_y), (arrow_start_x, start_y + self._base_pipeline_arrow_ht),
                            (0, 0, 0), 3, cv2.LINE_AA)
            # After arrow
            start_y += self._base_pipeline_arrow_ht
        # cv2.imshow('img', img)
        # cv2.waitKey()
        cv2.imwrite(self._out_path, img)

    def _update_pipeline_base(self, img, name, run_times, indent_level):
        if len(name) > self._max_pipeline_name_len:
            name = name[:self._max_pipeline_name_len - 3] + '...'
        run_count = len(run_times)
        cv2.putText(img, 'Pipeline: ' + name, (int(0.1 * self._base_pipeline_wd), int(self._base_pipeline_ht/4)),
                    self._pipeline_name_font, 1.4, (206,0,48), 3, cv2.LINE_AA)
        stats_text = 'Total number of runs: {}. Runtime Statistics:'.format(run_count)
        cv2.putText(img, stats_text, (int(0.1*self._base_pipeline_wd), int(0.5 * self._base_pipeline_ht)),
                    self._pipeline_text_font, 1.0, (0,0,0), 2, cv2.LINE_AA)
        stats_text = 'Max = {}  Min = {}  Avg = {}'.format(max(run_times), min(run_times), list_avg(run_times))
        cv2.putText(img, stats_text, (int(0.1*self._base_pipeline_wd), int(0.7 * self._base_pipeline_ht)),
                    self._pipeline_text_font, 0.8, (0,0,0), 1, cv2.LINE_AA)
        indent_text = '[Level = ' + str(indent_level) + ']'
        cv2.putText(img, indent_text, (int(0.88*self._base_pipeline_wd), int(0.35 * self._base_pipeline_ht)),
                    self._pipeline_text_font, 0.8, (150,145,142), 1, cv2.LINE_AA)
        logo_roi = img[25:25+50, self._base_pipeline_wd-(50+65):-65]
        logo_alpha = self._logo[..., -1]/255
        logo_alpha = np.repeat(logo_alpha[:, :, np.newaxis], 3, axis=2)
        logo_roi = logo_alpha * self._logo[..., :3] + (1 - logo_alpha) * logo_roi
        img[25:25+50, self._base_pipeline_wd-(50+65):-65]= logo_roi

    def _new_component(self, name, run_times):
        if len(name) > self._max_component_name_len:
            name = name[:self._max_component_name_len - 3] + '...'
        run_count = len(run_times)
        img = np.zeros((self._base_component_ht, self._base_component_wd, 3), dtype='uint8')
        img[...] = self._pipeline_bg_color
        img = add_rounded_rectangle_border(img, border_radius_percent=0.04, line_thickness_percent=0.003,
                                           color=(0,0,0), fill_color=self._component_bg_color)
        cv2.putText(img, name, (int(0.1*self._base_component_wd), int(self._base_component_ht/4)),
                    self._component_name_font, 1.2, (266,161,0), 2, cv2.LINE_AA)
        stats_text = 'Total number of runs: {}. Runtime Statistics:'.format(run_count)
        cv2.putText(img, stats_text, (int(0.1*self._base_component_wd), int(0.5 * self._base_component_ht)),
                    self._component_text_font, 1.0, (0,0,0), 2, cv2.LINE_AA)
        stats_text = 'Max = {}  Min = {}  Avg = {}'.format(max(run_times), min(run_times), list_avg(run_times))
        cv2.putText(img, stats_text, (int(0.1*self._base_component_wd), int(0.7 * self._base_component_ht)),
                    self._component_text_font, 0.8, (0,0,0), 1, cv2.LINE_AA)
        return img

    def _new_pipeline(self, pipeline_dict):
        num_components = len(pipeline_dict['component_names'])
        total_ht, total_wd = self._get_new_pipeline_dims(num_components)
        img = 255 * np.ones((total_ht, total_wd, 3), dtype='uint8')
        img = add_rounded_rectangle_border(img, border_radius_percent=0.08, line_thickness_percent=0.006,
                                           color=(206,0,48), fill_color=self._pipeline_bg_color)
        self._update_pipeline_base(img=img, name=pipeline_dict['name'],
                                   run_times=pipeline_dict['pipeline_run_times'],
                                   indent_level=pipeline_dict['indent_level'])
        # Iterate and add components
        start_y = self._base_pipeline_ht
        arrow_start_x = int((total_wd/2) - 5)
        for iter_comp, comp_name in enumerate(pipeline_dict['component_names']):
            # Add component
            comp_img = self._new_component(name=comp_name, run_times=pipeline_dict['component_run_times'][comp_name])
            comp_img_ht, comp_img_wd = comp_img.shape[:2]
            start_x = int((total_wd/2 - comp_img_wd/2))
            img[start_y:start_y+comp_img_ht, start_x:start_x+comp_img_wd] = comp_img.copy()
            if iter_comp == num_components - 1:
                break
            # Add arrow
            start_y += comp_img_ht
            cv2.arrowedLine(img, (arrow_start_x, start_y), (arrow_start_x, start_y + self._base_component_arrow_ht),
                            (0, 0, 0), 2, cv2.LINE_AA)
            # After arrow
            start_y += self._base_component_arrow_ht
        return img

    def _get_new_pipeline_dims(self, num_components):
        total_wd = self._base_pipeline_wd
        total_ht = self._base_pipeline_ht + num_components * self._base_component_ht + \
                   (num_components - 1) * self._base_component_arrow_ht + self._base_pipeline_ht_buffer
        return total_ht, total_wd


def add_rounded_rectangle_border(img, border_radius_percent=0.08, line_thickness_percent=0.006, color=(206,0,48),
                                 fill_color=None):
    height, width, channels = img.shape

    border_radius = int(width * border_radius_percent)
    line_thickness = int(max(width, height) * line_thickness_percent)
    edge_shift = int(line_thickness/3.0)

    red = color[-1]
    green = color[1]
    blue = color[0]

    if fill_color is not None:
        # Rectangle fill
        img[edge_shift:height-line_thickness, border_radius:width - border_radius] = fill_color
        img[border_radius:height-border_radius, edge_shift:width-line_thickness] = fill_color
        # corners
        cv2.ellipse(img, (border_radius+ edge_shift, border_radius+edge_shift),
                    (border_radius, border_radius), 180, 0, 90, fill_color, -1)
        cv2.ellipse(img, (width-(border_radius+line_thickness), border_radius),
                    (border_radius, border_radius), 270, 0, 90, fill_color, -1)
        cv2.ellipse(img, (width-(border_radius+line_thickness), height-(border_radius + line_thickness)),
                    (border_radius, border_radius), 0, 0, 90, fill_color, -1)
        cv2.ellipse(img, (border_radius+edge_shift, height-(border_radius + line_thickness)),
                    (border_radius, border_radius), 90, 0, 90, fill_color, -1)

    # draw lines
    # top
    cv2.line(img, (border_radius, edge_shift),
             (width - border_radius, edge_shift), (blue, green, red), line_thickness)
    # bottom
    cv2.line(img, (border_radius, height-line_thickness),
             (width - border_radius, height-line_thickness), (blue, green, red), line_thickness)
    # left
    cv2.line(img, (edge_shift, border_radius),
             (edge_shift, height-border_radius), (blue, green, red), line_thickness)
    # right
    cv2.line(img, (width-line_thickness, border_radius),
             (width-line_thickness, height-border_radius), (blue, green, red), line_thickness)

    # corners
    cv2.ellipse(img, (border_radius+ edge_shift, border_radius+edge_shift),
                (border_radius, border_radius), 180, 0, 90, color, line_thickness)
    cv2.ellipse(img, (width-(border_radius+line_thickness), border_radius),
                (border_radius, border_radius), 270, 0, 90, color, line_thickness)
    cv2.ellipse(img, (width-(border_radius+line_thickness), height-(border_radius + line_thickness)),
                (border_radius, border_radius), 0, 0, 90, color, line_thickness)
    cv2.ellipse(img, (border_radius+edge_shift, height-(border_radius + line_thickness)),
                (border_radius, border_radius), 90, 0, 90, color, line_thickness)

    return img


def fprint(pid, indent_level, run_count, comp_name, msg):
    if comp_name is None:
        print('\n\n\n{}{} [{}][PID {}][n_time={}]'.format(get_tabs(indent_level), msg, datetime.now(), pid, run_count))
    else:
        print('{}[{}][PID {}][Comp {}][n_time={}] {}'.format(get_tabs(indent_level),
               datetime.now(), pid, comp_name, run_count, msg))


def get_tabs(indent_level):
    return '\t' * indent_level


def list_avg(lst):
    return reduce(lambda a, b: a + b, lst) / len(lst)


def repeator(*args, **kwargs):
    arg_dict = {i:i for i in args}
    return {**arg_dict, **kwargs}
