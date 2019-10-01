import torch.nn as nn
import torch
import numpy as np
from typing import List, Tuple
import torch.nn.functional as F
import time

from ..utils import box_utils
from collections import namedtuple
GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])  #


def file_open():
    output_f = open('/home/jian/Documents/adaptive_batching/yolo-9000/darknet/pytorch_test/batch_test.txt', 'a')
    return output_f


def file_write(output_f, content):
    output_f.write(content + '\n')


def file_close(output_f):
    output_f.flush()
    output_f.close()


class SSD(nn.Module):
    def __init__(self, num_classes: int, base_net: nn.ModuleList, source_layer_indexes: List[int],
                 extras: nn.ModuleList, classification_headers: nn.ModuleList,
                 regression_headers: nn.ModuleList, is_test=False, config=None, device=None):
        """Compose a SSD model using the given components.
        """
        super(SSD, self).__init__()

        self.num_classes = num_classes
        self.base_net = base_net
        self.source_layer_indexes = source_layer_indexes
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.is_test = is_test
        self.config = config

        # register layers in source_layer_indexes by adding them to a module list
        self.source_layer_add_ons = nn.ModuleList([t[1] for t in source_layer_indexes
                                                   if isinstance(t, tuple) and not isinstance(t, GraphPath)])
        # print('source_layer_add_ons: \n' + str(self.source_layer_add_ons))
        if device:
            self.device = device
        else:
            # self.device = torch.cuda.current_device()
            self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        self.device = torch.cuda.current_device()
        # print(self.device)
        # self.device = torch.device("cpu")
        if is_test:
            self.config = config
            self.priors = config.priors.to(self.device)
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        for end_layer_index in self.source_layer_indexes:
            if isinstance(end_layer_index, GraphPath):
                path = end_layer_index
                end_layer_index = end_layer_index.s0
                added_layer = None
            elif isinstance(end_layer_index, tuple):
                added_layer = end_layer_index[1]
                end_layer_index = end_layer_index[0]
                path = None
            else:
                added_layer = None
                path = None
            
            output_f = file_open()
            print('strat_layer:' + str(start_layer_index) + ',' + str(end_layer_index))
            for layer in self.base_net[start_layer_index: end_layer_index]:
                # print(x.size())
                torch.cuda.synchronize()
                t0 = time.time()
                x = layer(x)
                torch.cuda.synchronize()
                t1 = time.time()
                # print(str(layer))
                print(str(layer) + ': ' + str(t1 - t0))
                print(x[0][0][0])
                file_write(output_f, str(t1 - t0))
            file_close(output_f)

            if added_layer:
                y = added_layer(x)
            else:
                y = x
            if path:
                sub = getattr(self.base_net[end_layer_index], path.name)
                output_f = file_open()
                for layer in sub[:path.s1]:
                    # print(x.size())
                    torch.cuda.synchronize()
                    t0 = time.time()
                    x = layer(x)
                    torch.cuda.synchronize()
                    t1 = time.time()
                    # print(str(layer))
                    print(str(layer) + ': ' + str(t1 - t0))
                    print(x[0][0][0])
                    file_write(output_f, str(t1 - t0))
                y = x
                for layer in sub[path.s1:]:
                    # print(x.size())
                    torch.cuda.synchronize()
                    t0 = time.time()
                    x = layer(x)
                    torch.cuda.synchronize()
                    t1 = time.time()
                    # print(str(layer))
                    print(str(layer) + ': ' + str(t1 - t0))
                    print(x[0][0][0])
                    file_write(output_f, str(t1 - t0))
                file_close(output_f)

                end_layer_index += 1
            start_layer_index = end_layer_index
            confidence, location = self.compute_header(header_index, y)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)
        
        output_f = file_open()
        for layer in self.base_net[end_layer_index:]:
            # print(x.size())
            torch.cuda.synchronize()
            t0 = time.time()
            x = layer(x)
            torch.cuda.synchronize()
            t1 = time.time()
            # print(str(layer))
            print(str(layer) + ': ' + str(t1 - t0))
            print(x[0][0][0])
            file_write(output_f, str(t1 - t0))
        file_close(output_f)
        
        # tmp_file = open('/home/jian/workspace_cx/pytorch_test/tmp_info.txt', 'w')
        # tmp_file.write('Start ModuleList\n')
        # tmp_file.flush()
        # tmp_file.close()
        for layer in self.extras:
            x = layer(x)
            print(str(layer))
            confidence, location = self.compute_header(header_index, x)
            # print('between')
            header_index += 1
            confidences.append(confidence)
            locations.append(location)
        # print('len of conf: ' + str(len(confidences)))

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        
        # if self.is_test:
        #     confidences = F.softmax(confidences, dim=2)
        #     boxes = box_utils.convert_locations_to_boxes(
        #         locations, self.priors, self.config.center_variance, self.config.size_variance
        #     )
        #     boxes = box_utils.center_form_to_corner_form(boxes)
        #     return confidences, boxes
        # else:
        #     return confidences, locations
        return confidences, locations

    def compute_header(self, i, x):
        tmp_file = open('../../../tmp_info.txt', 'w')
        tmp_file.write('Start ModuleList\n')
        tmp_file.flush()
        tmp_file.close()

        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)

        tmp_file = open('../../../tmp_info.txt', 'w')
        tmp_file.flush()
        tmp_file.close()

        return confidence, location

    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if not (k.startswith("classification_headers") or k.startswith("regression_headers"))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


class MatchPrior(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
        return locations, labels


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
