import numpy as np
import math
from copy import deepcopy
from collections import defaultdict
from lxml.etree import XMLParser, parse, ElementTree, Element, SubElement
from lxml import etree
from io import BytesIO
from pprint import pprint
import os
import sys
import time
import yaml
# sys.path.append(os.getcwd())
# from mujoco_py import load_model_from_path, MjSim, MjViewer
from collections import defaultdict, deque
import gym
import numpy as np
import random
import pprint
import xml.etree.ElementTree as ET
import re
import networkx as nx
import subprocess
import json
import itertools

import multiprocessing
# from threading import Lock

file_lock = multiprocessing.Lock()



global MAX_CHILD_ALLOWED
MAX_CHILD_ALLOWED=100



def parse_vec(string):
    return np.fromstring(string, sep=' ')

def parse_fromto(string):
    fromto = np.fromstring(string, sep=' ')
    return fromto[:3], fromto[3:]

def normalize_range(value, lb, ub):
    return (value - lb) / (ub - lb) * 2 - 1

def denormalize_range(value, lb, ub):
    return (value + 1) * 0.5 * (ub - lb) + lb

def vec_to_polar(v):
    phi = math.atan2(v[1], v[0])
    theta = math.acos(v[2])
    return np.array([theta, phi])

def polar_to_vec(p):
    v = np.zeros(3)
    v[0] = math.sin(p[0]) * math.cos(p[1])
    v[1] = math.sin(p[0]) * math.sin(p[1])
    v[2] = math.cos(p[0])
    return v

def transform_body(angle_deg, body):
    
    magnitude = np.linalg.norm(body)
    normalized_vector = body / magnitude
    
    angle_rad = np.deg2rad(angle_deg)
    rotation_matrix = np.array([
                        [np.cos(angle_rad), -np.sin(angle_rad), 0],
                        [np.sin(angle_rad), np.cos(angle_rad), 0],
                        [0, 0, 1]
                        ])
    # rotated_vector = np.dot(rotation_matrix, body)
    
    rotated_vector = np.dot(rotation_matrix, normalized_vector)
    final_vector = rotated_vector * magnitude
    # print('ratated vector', final_vector)
    x, y, z = final_vector.ravel()
    return x,y,z

class Joint:

    def __init__(self, node, body):
        self.node = node
        self.body = body
        self.cfg = body.cfg
        self.local_coord = body.local_coord
        
        self.name = node.attrib['name']
        self.type = node.attrib['type']
        if self.type == 'hinge':
            self.range = np.deg2rad(parse_vec(node.attrib.get('range', "-360 360")))
        actu_node = body.tree.getroot().find("actuator").find(f'motor[@joint="{self.name}"]')
        if actu_node is not None:
            self.actuator = Actuator(actu_node, self)
        else:
            self.actuator = None
        self.parse_param_specs()
        self.param_inited = False
        # tunable parameters
        self.pos = parse_vec(node.attrib['pos'])

        # if self.type == 'hinge':
        #     self.axis = vec_to_polar(parse_vec(node.attrib['axis']))
        if self.local_coord:
            self.pos += body.pos
                
    def __repr__(self):
        return 'joint_' + self.name  

    def parse_param_specs(self):
        self.param_specs =  deepcopy(self.cfg['joint_params'])
        for name, specs in self.param_specs.items():
            if 'lb' in specs and isinstance(specs['lb'], list):
                specs['lb'] = np.array(specs['lb'])
            if 'ub' in specs and isinstance(specs['ub'], list):
                specs['ub'] = np.array(specs['ub'])

    def sync_node(self, new_pos):
        if new_pos is not None:
            pos = new_pos
        else:
            pos = self.pos

        self.name = self.body.name + '_joint'
        self.node.attrib['name'] = self.name
        self.node.attrib['pos'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in pos])
        # if self.type == 'hinge':
        #     axis_vec = polar_to_vec(self.axis)
        #     self.node.attrib['axis'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in axis_vec])
        if self.actuator is not None:
            self.actuator.sync_node()
        
    def get_params(self, param_list, get_name=False, pad_zeros=False):
        if 'axis' in self.param_specs:
            if self.type == 'hinge':
                if get_name:
                    param_list += ['axis_theta', 'axis_phi']
                else:
                    axis = normalize_range(self.axis, np.array([0, -2 * np.pi]), np.array([np.pi, 2 * np.pi]))
                    param_list.append(axis)
            elif pad_zeros:
                param_list.append(np.zeros(2))

        if self.actuator is not None:
            self.actuator.get_params(param_list, get_name)
        elif pad_zeros:
            param_list.append(np.zeros(1))


        if not get_name:
            self.param_inited = True

    def set_params(self, params, pad_zeros=False):
        if 'axis' in self.param_specs:
            if self.type == 'hinge':
                self.axis = denormalize_range(params[:2], np.array([0, -2 * np.pi]), np.array([np.pi, 2 * np.pi]))
                params = params[2:]
            elif pad_zeros:
                params = params[2:]

        if self.actuator is not None:
            params = self.actuator.set_params(params)
        elif pad_zeros:
            params = params[1:]
        return params


class Geom:

    def __init__(self, node, body):
        self.node = node
        self.body = body
        self.cfg = body.cfg
        self.local_coord = body.local_coord
        self.name = node.attrib.get('name', '')
        self.type = node.attrib['type']
        self.parse_param_specs()
        self.param_inited = False
        # tunable parameters
        self.size = parse_vec(node.attrib['size'])
        if self.type == 'capsule':
            self.start, self.end = parse_fromto(node.attrib['fromto'])
            # print("In geom constructor the start & end before local_coordinate influence are", self.start, self.end)
            if self.local_coord:
                self.start += body.pos
                self.end += body.pos
            # print("In geom constructor the start & end after local_coordinate influence are", self.start, self.end)
            # print("-------------------------------------------------------")
            if body.bone_start is None:
                self.bone_start = self.start.copy()
                body.bone_start = self.bone_start.copy()
            else:
                self.bone_start = body.bone_start.copy()
            self.ext_start = np.linalg.norm(self.bone_start - self.start)
        # print("self.start & self.end at the end of Geom constructor", self.start, self.end)

    def __repr__(self):
        return 'geom_' + self.name

    def parse_param_specs(self):
        self.param_specs = deepcopy(self.cfg['geom_params'])
        for name, specs in self.param_specs.items():
            if 'lb' in specs and isinstance(specs['lb'], list):
                specs['lb'] = np.array(specs['lb'])
            if 'ub' in specs and isinstance(specs['ub'], list):
                specs['ub'] = np.array(specs['ub'])

    def update_start(self):
        if self.type == 'capsule':
            vec = self.bone_start - self.end
            # print("vec, self.bone_start, self.end", vec, self.bone_start, self.end)
            self.start = self.bone_start + vec * (self.ext_start / np.linalg.norm(vec))

    def sync_node(self, body_start=None, body_end=None):
        # self.node.attrib['name'] = self.name
        self.node.attrib.pop('name', None)
        self.node.attrib['size'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in self.size])
        # print("self.type to check if it is capsule", self.type)
        # print("body_start", body_start)
        if self.type == 'capsule':
            # print("**********")
            # print("self.start", self.start)
            # print("self.body.pos", self.body.pos)
            # print("self.end", self.end)
            # print("**********")
            start = self.start - self.body.pos if self.local_coord else self.start
            end = self.end - self.body.pos if self.local_coord else self.end
            if body_start is not None:
                # print("body start is not none, i.e. received new body_start and body_end")
                self.node.attrib['fromto'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in np.concatenate([body_start, body_end])])
            else:
                # print("body start is none, i.e. didnot receive new body_start and body_end")
                self.node.attrib['fromto'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in np.concatenate([start, end])])

            # print("self.node.attrib['fromto']", self.node.attrib['fromto'])


    def get_params(self, param_list, get_name=False, pad_zeros=False):
        if 'size' in self.param_specs:
            if get_name:
                param_list.append('size')
            else:
                if self.type == 'capsule':
                    if not self.param_inited and self.param_specs['size'].get('rel', False):
                        self.param_specs['size']['lb'] += self.size
                        self.param_specs['size']['ub'] += self.size
                        self.param_specs['size']['lb'] = max(self.param_specs['size']['lb'], self.param_specs['size'].get('min', -np.inf))
                        self.param_specs['size']['ub'] = min(self.param_specs['size']['ub'], self.param_specs['size'].get('max', np.inf))
                    size = normalize_range(self.size, self.param_specs['size']['lb'], self.param_specs['size']['ub'])
                    param_list.append(size.flatten())
                elif pad_zeros:
                    param_list.append(np.zeros(1))
        if 'ext_start' in self.param_specs:
            if get_name:
                param_list.append('ext_start')
            else:
                if self.type == 'capsule':
                    if not self.param_inited and self.param_specs['ext_start'].get('rel', False):
                        self.param_specs['ext_start']['lb'] += self.ext_start
                        self.param_specs['ext_start']['ub'] += self.ext_start
                        self.param_specs['ext_start']['lb'] = max(self.param_specs['ext_start']['lb'], self.param_specs['ext_start'].get('min', -np.inf))
                        self.param_specs['ext_start']['ub'] = min(self.param_specs['ext_start']['ub'], self.param_specs['ext_start'].get('max', np.inf))
                    ext_start = normalize_range(self.ext_start, self.param_specs['ext_start']['lb'], self.param_specs['ext_start']['ub'])
                    param_list.append(ext_start.flatten())
                elif pad_zeros:
                    param_list.append(np.zeros(1))

        if not get_name:
            self.param_inited = True

    def set_params(self, params, pad_zeros=False):
        if 'size' in self.param_specs:
            if self.type == 'capsule':
                self.size = denormalize_range(params[[0]], self.param_specs['size']['lb'], self.param_specs['size']['ub'])
                params = params[1:]
            elif pad_zeros:
                params = params[1:]
        if 'ext_start' in self.param_specs:
            if self.type == 'capsule':
                self.ext_start = denormalize_range(params[[0]], self.param_specs['ext_start']['lb'], self.param_specs['ext_start']['ub'])
                params = params[1:]
            elif pad_zeros:
                params = params[1:]
        return params


class Actuator:

    def __init__(self, node, joint):
        self.node = node
        self.joint = joint
        self.cfg = joint.cfg
        self.joint_name = node.attrib['joint']
        self.name = self.joint_name
        self.parse_param_specs()
        self.param_inited = False
        # tunable parameters
        self.gear = float(node.attrib['gear'])

    def parse_param_specs(self):
        self.param_specs =  deepcopy(self.cfg['actuator_params'])
        for name, specs in self.param_specs.items():
            if 'lb' in specs and isinstance(specs['lb'], list):
                specs['lb'] = np.array(specs['lb'])
            if 'ub' in specs and isinstance(specs['ub'], list):
                specs['ub'] = np.array(specs['ub'])

    def sync_node(self):
        self.node.attrib['gear'] = f'{self.gear:.6f}'.rstrip('0').rstrip('.')
        self.name = self.joint.name
        self.node.attrib['name'] = self.name
        self.node.attrib['joint'] = self.joint.name

    def get_params(self, param_list, get_name=False):
        if 'gear' in self.param_specs:
            if get_name:
                param_list.append('gear')
            else:
                if not self.param_inited and self.param_specs['gear'].get('rel', False):
                    self.param_specs['gear']['lb'] += self.gear
                    self.param_specs['gear']['ub'] += self.gear
                    self.param_specs['gear']['lb'] = max(self.param_specs['gear']['lb'], self.param_specs['gear'].get('min', -np.inf))
                    self.param_specs['gear']['ub'] = min(self.param_specs['gear']['ub'], self.param_specs['gear'].get('max', np.inf))
                gear = normalize_range(self.gear, self.param_specs['gear']['lb'], self.param_specs['gear']['ub'])
                param_list.append(np.array([gear]))

        if not get_name:
            self.param_inited = True

    def set_params(self, params):
        if 'gear' in self.param_specs:
            self.gear = denormalize_range(params[0].item(), self.param_specs['gear']['lb'], self.param_specs['gear']['ub'])
            params = params[1:]
        return params


class Body:

    def __init__(self, node, parent_body, robot, cfg):
        self.node = node
        self.parent = parent_body
        if parent_body is not None:
            parent_body.child.append(self)
            parent_body.cind += 1
            self.depth = parent_body.depth + 1
        else:
            self.depth = 0
        self.robot = robot
        self.cfg = cfg
        self.tree = robot.tree
        self.local_coord = robot.local_coord
        # print(node.attrib)
        # print("self.parent.name", self.parent.name)
        # print(self.parent.name + f'_child{len(self.parent.child)}')
        self.name = node.attrib['name'] if 'name' in node.attrib else self.parent.name + f'_child{len(self.parent.child)}'
        # print("self.name", self.name)
        self.child = []
        self.cind = 0
        self.pos = parse_vec(node.attrib['pos'])
        if self.local_coord and parent_body is not None:
            self.pos += parent_body.pos

        if cfg.get('init_root_from_geom', False):
            self.bone_start = None if parent_body is None else self.pos.copy()
        else:
            self.bone_start = self.pos.copy()
        self.joints = [Joint(x, self) for x in node.findall('joint[@type="hinge"]')] + [Joint(x, self) for x in node.findall('joint[@type="free"]')]
        self.geoms = [Geom(x, self) for x in node.findall('geom[@type="capsule"]')] + [Geom(x, self) for x in node.findall('geom[@type="sphere"]')]
        self.parse_param_specs()
        self.param_inited = False
        # parameters
        self.bone_end = None
        self.bone_offset = None

    def __repr__(self):
        # print("Name", self.name)
        return 'body_' + self.name

    def parse_param_specs(self):
        self.param_specs = deepcopy(self.cfg['body_params'])
        for name, specs in self.param_specs.items():
            if 'lb' in specs and isinstance(specs['lb'], list):
                specs['lb'] = np.array(specs['lb'])
            if 'ub' in specs and isinstance(specs['ub'], list):
                specs['ub'] = np.array(specs['ub'])
            if name == 'bone_ang':
                specs['lb'] = np.deg2rad(specs['lb'])
                specs['ub'] = np.deg2rad(specs['ub'])

    def reindex(self):
        if self.parent is None:
            self.name = '0'
        else:
            ind = self.parent.child.index(self) + 1
            pname = '' if self.parent.name == '0' else self.parent.name
            self.name = str(ind) + pname

    def init(self):
        if len(self.child) > 0:
            bone_ends = [x.bone_start for x in self.child]
        else:
            bone_ends = [x.end for x in self.geoms]
        if len(bone_ends) > 0:
            self.bone_end = np.mean(np.stack(bone_ends), axis=0)
            self.bone_offset = self.bone_end - self.bone_start

    def get_actuator_name(self):
        for joint in self.joints:
            if joint.actuator is not None:
                return joint.actuator.name
        return None

    def get_joint_range(self):
        assert len(self.joints) == 1
        return self.joints[0].range

    def get_axis(self):   # Kishan
        assert len(self.joints) == 1
        return self.joints[0].axis

    def sync_node(self, body_start=None, body_end=None):

        # setting all bodies inertial frame to initialize at (0,0,0). 
        # pos = np.array([0,0,0])
        # print("pos value in sync node of Body class", pos, type(pos))

        self.node.attrib['name'] = self.name
        # print("self.node.attrib['name']", self.node.attrib['name'])
        # self.node.attrib['pos'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in pos]) # converting array to string
        # print("Before joint and geom sync ::: self.node.attrib['pos']", self.node.attrib['pos'])
        
        for joint in self.joints:
            joint.sync_node(body_start)
        for geom in self.geoms:
            geom.sync_node(body_start=None, body_end=None)
        
        # print("After joint and geom sync :::  fromto", geom.sync_node.attrib['fromto'])       

    def sync_geom(self):
        for geom in self.geoms:
            geom.bone_start = self.bone_start.copy()
            geom.end = self.bone_end.copy()
            geom.update_start()

    def sync_joint(self):
        if self.parent is not None:
            for joint in self.joints:
                joint.pos = self.pos.copy()

    def rebuild(self):
        if self.parent is not None:
            self.bone_start = self.parent.bone_end.copy()
            self.pos = self.bone_start.copy()
        if self.bone_offset is not None:
            self.bone_end = self.bone_start + self.bone_offset
        if self.parent is None and self.cfg.get('no_root_offset', False):
            self.bone_end = self.bone_start
        self.sync_geom()
        self.sync_joint()

    def get_params(self, param_list, get_name=False, pad_zeros=False, demap_params=False):
        if self.bone_offset is not None and 'offset' in self.param_specs:
            if get_name:
                if self.param_specs['offset']['type'] == 'xz':
                    param_list += ['offset_x', 'offset_z']
                elif self.param_specs['offset']['type'] == 'xy':
                    param_list += ['offset_x', 'offset_y']
                else:
                    param_list += ['offset_x', 'offset_y', 'offset_z']
            else:
                if self.param_specs['offset']['type'] == 'xz':
                    offset = self.bone_offset[[0, 2]]
                elif self.param_specs['offset']['type'] == 'xy':
                    offset = self.bone_offset[[0, 1]]
                else:
                    offset = self.bone_offset
                if not self.param_inited and self.param_specs['offset'].get('rel', False):
                    self.param_specs['offset']['lb'] += offset
                    self.param_specs['offset']['ub'] += offset
                    self.param_specs['offset']['lb'] = np.maximum(self.param_specs['offset']['lb'], self.param_specs['offset'].get('min', np.full_like(offset, -np.inf)))
                    self.param_specs['offset']['ub'] = np.minimum(self.param_specs['offset']['ub'], self.param_specs['offset'].get('max', np.full_like(offset, np.inf)))
                offset = normalize_range(offset, self.param_specs['offset']['lb'], self.param_specs['offset']['ub'])
                param_list.append(offset.flatten())

        if self.bone_offset is not None and 'bone_len' in self.param_specs:
            if get_name:
                param_list += ['bone_len']
            else:
                bone_len = np.linalg.norm(self.bone_offset)
                if not self.param_inited and self.param_specs['bone_len'].get('rel', False):
                    self.param_specs['bone_len']['lb'] += bone_len
                    self.param_specs['bone_len']['ub'] += bone_len
                    self.param_specs['bone_len']['lb'] = max(self.param_specs['bone_len']['lb'], self.param_specs['bone_len'].get('min',-np.inf))
                    self.param_specs['bone_len']['ub'] = min(self.param_specs['bone_len']['ub'], self.param_specs['bone_len'].get('max', np.inf))
                bone_len = normalize_range(bone_len, self.param_specs['bone_len']['lb'], self.param_specs['bone_len']['ub'])
                param_list.append(np.array([bone_len]))

        if self.bone_offset is not None and 'bone_ang' in self.param_specs:
            if get_name:
                param_list += ['bone_ang']
            else:
                bone_ang = math.atan2(self.bone_offset[2], self.bone_offset[0])
                if not self.param_inited and self.param_specs['bone_ang'].get('rel', False):
                    self.param_specs['bone_ang']['lb'] += bone_ang
                    self.param_specs['bone_ang']['ub'] += bone_ang
                    self.param_specs['bone_ang']['lb'] = max(self.param_specs['bone_ang']['lb'], self.param_specs['bone_ang'].get('min',-np.inf))
                    self.param_specs['bone_ang']['ub'] = min(self.param_specs['bone_ang']['ub'], self.param_specs['bone_ang'].get('max', np.inf))
                bone_ang = normalize_range(bone_ang, self.param_specs['bone_ang']['lb'], self.param_specs['bone_ang']['ub'])
                param_list.append(np.array([bone_ang]))

        for joint in self.joints:
            joint.get_params(param_list, get_name, pad_zeros)
        if pad_zeros and len(self.joints) == 0:
            param_list.append(np.zeros(1))

        for geom in self.geoms:
            geom.get_params(param_list, get_name, pad_zeros)
        if not get_name:
            self.param_inited = True

        if demap_params and not get_name:
            params = self.robot.demap_params(np.concatenate(param_list))
            return params

    def set_params(self, params, pad_zeros=False, map_params=False):
        if map_params:
            params = self.robot.map_params(params)

        if self.bone_offset is not None and 'offset' in self.param_specs:
            if self.param_specs['offset']['type'] in {'xz', 'xy'}:
                offset = denormalize_range(params[:2], self.param_specs['offset']['lb'], self.param_specs['offset']['ub'])
                if np.all(offset == 0.0):
                    offset[0] += 1e-8
                if self.param_specs['offset']['type'] == 'xz':
                    self.bone_offset[[0, 2]] = offset
                elif self.param_specs['offset']['type'] == 'xy':
                    self.bone_offset[[0, 1]] = offset
                params = params[2:]
            else:
                offset = denormalize_range(params[:3], self.param_specs['offset']['lb'], self.param_specs['offset']['ub'])
                if np.all(offset == 0.0):
                    offset[0] += 1e-8
                self.bone_offset[:] = offset
                params = params[3:]

        if self.bone_offset is not None and 'bone_len' in self.param_specs:
            bone_len = denormalize_range(params[0].item(), self.param_specs['bone_len']['lb'], self.param_specs['bone_len']['ub'])
            bone_len = max(bone_len, 1e-4)
            params = params[1:]
        else:
            bone_len = np.linalg.norm(self.bone_offset)

        if self.bone_offset is not None and 'bone_ang' in self.param_specs:
            bone_ang = denormalize_range(params[0].item(), self.param_specs['bone_ang']['lb'], self.param_specs['bone_ang']['ub'])
            params = params[1:]
        else:
            bone_ang = math.atan2(self.bone_offset[2], self.bone_offset[0])

        if 'bone_len' in self.param_specs or 'bone_ang' in self.param_specs:
            self.bone_offset = np.array([bone_len * math.cos(bone_ang), 0, bone_len * math.sin(bone_ang)])

        for joint in self.joints:
            params = joint.set_params(params, pad_zeros)
        for geom in self.geoms:
            params = geom.set_params(params, pad_zeros)
        # rebuild bone, geom, joint
        self.rebuild()
        return params


class Robot:

    def __init__(self, cfg, xml, is_xml_str=False):
        self.bodies = []
        self.cfg = cfg
        self.param_mapping = cfg.get('param_mapping', 'clip')
        self.tree = None    # xml tree
        self.load_from_xml(xml, is_xml_str)
        self.init_bodies()
        # self.param_names = self.get_params(get_name=True)
        # self.init_params = self.get_params()

    def load_from_xml(self, xml, is_xml_str=False):
        parser = XMLParser(remove_blank_text=True)
        self.tree = parse(BytesIO(xml) if is_xml_str else xml, parser=parser)
        self.local_coord = self.tree.getroot().find('.//compiler').attrib['coordinate'] == 'local'
        self.local_coord = None
        root = self.tree.getroot().find('worldbody').find('body')
        self.add_body(root, None)

    def add_body(self, body_node, parent_body):
        body = Body(body_node, parent_body, self, self.cfg)
        
        self.bodies.append(body)
        # print("add body", self.bodies[0].geoms[0].end)

        for body_node_c in body_node.findall('body'):
            self.add_body(body_node_c, body)

    def init_bodies(self, body_start=None, body_end=None):
        for body in self.bodies:
            body.init()
        self.sync_node(body_start, body_end)

    def sync_node(self, body_start=None, body_end=None):
        for body in self.bodies:
            body.reindex()
            body.sync_node(body_start, body_end)
        

    def add_child_to_body(self, parent_body):
        if parent_body == self.bodies[0]:
            body2clone = parent_body.child[0]
        else:
            body2clone = parent_body
        child_node = deepcopy(body2clone.node)
        for bnode in child_node.findall('body'):
            child_node.remove(bnode)
        child_body = Body(child_node, parent_body, self, self.cfg)
        actu_node = parent_body.tree.getroot().find("actuator")
        for joint in child_body.joints:
            new_actu_node = deepcopy(actu_node.find(f'motor[@joint="{joint.name}"]'))
            actu_node.append(new_actu_node)
            joint.actuator = Actuator(new_actu_node, joint)
        child_body.bone_offset = parent_body.bone_offset.copy()
        child_body.param_specs = deepcopy(parent_body.param_specs)
        child_body.param_inited = True
        # print(vars(child_body))
        # print("_____________________")
        child_body.rebuild()
        child_body.sync_node()
        # print(vars(child_body))
        parent_body.node.append(child_node)
        self.bodies.append(child_body)
        self.sync_node()
        
    def remove_body(self, body):
        body.node.getparent().remove(body.node)
        body.parent.child.remove(body)
        self.bodies.remove(body)
        actu_node = body.tree.getroot().find("actuator")
        for joint in body.joints:
            actu_node.remove(joint.actuator.node)
        del body
        self.sync_node()


    def redesign_bodies_with_init_design(self, body_list, body, attach_loc=None, geom_type=None, parent_body=None, length_ratio=None, size_factor=None, gear_factor=None):
        #modify the created robot bodies based on the required specification

        body_start = parent_body.geoms[0].start 
        body_end = parent_body.geoms[0].end
        # print("parent_body body_start", body_start, type(body_start))
        # print("parent_body body_end", body_end)
        #take the length of parent body
        body_start_array = np.array(body_start) 
        body_end_array = np.array(body_end)
        sign = np.sign(body_end_array)
        # print("sign", sign)
        # print("length ratio is", length_ratio)
        length = np.abs(np.abs(body_end_array) - np.abs(body_start_array)) * length_ratio if length_ratio is not None else np.abs(np.abs(body_end_array) - np.abs(body_start_array))         # The final length is a result of (default length * length ratio) - Apr23 Kishan
        # print("length unsigned", length, type(length))
        length_new = length * sign
        # print("length signed", length_new, length, sign, type(length_new))
        # print("BEFORE REDESIGN ::: BODY START & BODY END and type of BODY END", body.geoms[0].start, body.geoms[0].end , type(body.geoms[0].end))
        # print("spot", spot)
        # attach_loc = None
        
        if attach_loc is None:
            pass
        if attach_loc == 'spot_0':
            angle = 0
            new_x, new_y, new_z = transform_body(angle, length_new) 
            body_end = [new_x, new_y, new_z]
        elif attach_loc == 'spot_1':
            angle = 30
            new_x, new_y, new_z = transform_body(angle, length_new) 
            body_end = [new_x, new_y, new_z]
            # print("body_end when spot_2 is selected", body_end)
        elif attach_loc == 'spot_2':
            angle = 60
            new_x, new_y, new_z = transform_body(angle, length_new) 
            body_end = [new_x, new_y, new_z]
            # print("body_end when spot_3 is selected", body_end)
        elif attach_loc == 'spot_3':
            angle = 90
            new_x, new_y, new_z = transform_body(angle, length_new) 
            body_end = [new_x, new_y, new_z]
        
        # print("attach loc", attach_loc)
        # print("bodu_end", body_end)
        # print("length_new", length_new)
        # print("Parent_body just before fail", parent_body)
        # print("TEST body.joints.mode print", body.joints[0].node, len(body.joints))
        # print("body_list", body_list)
        # Using the parent's fromto values to set connnection point between parent and current body (i.e the child)    
        if parent_body is None:
            # print("parent_body", parent_body)
            # if body.node.attrib['name'] == '1':
            #     body.node.attrib['pos'] = '0.2 0.2 0'
            #     body.joints[0].node.attrib['pos'] = '0 0 0'
            # elif body.node.attrib['name'] == '0':
            #     body.node.attrib['pos'] = '0 0 0.75'
            #     body.joints[0].node.attrib['pos'] = '0 0 0'
            # # exit()
            # print("Parent body is none")
            pass
        else:
            # body_fromto = body_list[-1].geoms[0].node.attrib['fromto']
            fromto_end = parent_body.geoms[0].end
            # body_start = fromto_end
            # print("parent_fromto_end", fromto_end)
            child_body_pos = fromto_end
            body.node.attrib['pos'] =  ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in child_body_pos])

            length_new = body_end
        
            sign_for_axis = np.sign(body_end)

            # print("sign_for_axis", sign_for_axis, type(sign_for_axis))

            if sign_for_axis[0] == sign_for_axis[1]:
                final_axis_sign = np.array([-1, 1, 0])
            else:
                final_axis_sign = np.array([-1, -1, 0])

            fixed_axis_values = [0.707, 0.707, 0]

            final_axis_values = fixed_axis_values * final_axis_sign

            body.joints[0].node.attrib['axis'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in final_axis_values])

            # print("body.node.attrib['pos']", body.node.attrib['pos'])
            # body_end = np.array(body_end)
            # print("body_end", body_end)
            # length = np.array(length_new)
            # body.geoms[0].end = length_new
            #concatenate body_start and body_end values and write to geom['fromto]
            # print("before")
            # print("body_start", body_start)
            # print("body_end", body_end)
            # pprint(vars(body.geoms[0]))
            # body.geoms[0].start = body_start
            fixed_pos_values = [0.0, 0.0, 0.0] 
            body.joints[0].node.attrib['pos'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in fixed_pos_values])

            # if attach_loc == ''
            # new_axis = 
            # body.joints[0].node.attrib['axis'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in new_axis])
            drgftSergf

            fixed_size = 0.08
            body.geoms[0].node.attrib['size'] = str(fixed_size * size_factor)

            body.geoms[0].start = fixed_pos_values
            body.geoms[0].end = length_new
            # pprint(vars(body.geoms[0]))
            body.geoms[0].node.attrib['fromto'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in np.concatenate([fixed_pos_values, length_new])])
            # print(body.geoms[0].node.attrib['fromto'])
            # print("AFTER REDESIGN ::: BODY START & BODY END and type of BODY END", body.geoms[0].start, body.geoms[0].end, type(body.geoms[0].end))
            fixed_gear = 150
          
            body.joints[0].actuator.node.attrib['gear'] = str(int(fixed_gear * gear_factor))

        body_list.append(body)
        self.bodies = body_list


    # def redesign_bodies(self, body_list, body, attach_loc=None, geom_type=None, parent_body=None, length_ratio=None):
    #     #modify the created robot bodies based on the required specification
    #     # print(vars(body.geoms[0])
        
    #     #these are base parent values and will be same for any body consider for redesign
        
    #     # print("body", body)
    #     # print("parent_body", parent_body)

    #     if body.name != '0':
    #         # body is not of the same type as body_0
    #         body_start = body.geoms[0].start 
    #         body_end = body.geoms[0].end
    #         length = (body_end - body_start) * length_ratio if length_ratio is not None else body_end - body_start         # The final length is a result of (default length * length ratio) - Apr23 Kishan
    #     # print("length", length, type(length))

    #     # print("BEFORE REDESIGN ::: BODY START & BODY END and type of BODY END", body.geoms[0].start, body.geoms[0].end , type(body.geoms[0].end))
    #     if attach_loc is None:
    #         pass
    #     if attach_loc == 'spot_0':
    #         angle = 0
    #         new_x, new_y, new_z = transform_body(angle, length) 
    #         body_end = [new_x, new_y, new_z]
    #     elif attach_loc == 'spot_1':
    #         angle = 30
    #         new_x, new_y, new_z = transform_body(angle, length) 
    #         body_end = [new_x, new_y, new_z]
    #         # print("body_end when spot_2 is selected", body_end)
    #     elif attach_loc == 'spot_2':
    #         angle = 60
    #         new_x, new_y, new_z = transform_body(angle, length) 
    #         body_end = [new_x, new_y, new_z]
    #         # print("body_end when spot_3 is selected", body_end)
    #     elif attach_loc == 'spot_3':
    #         angle = 90
    #         new_x, new_y, new_z = transform_body(angle, length) 
    #         body_end = [new_x, new_y, new_z]

    #     # print("body_list", body_list)
    #     # Using the parent's fromto values to set connnection point between parent and current body (i.e the child)    
    #     if parent_body is None:
    #         if body.node.attrib['name'] == '1':
    #             body.node.attrib['pos'] = '0.2 0.2 0'
    #             body.joints[0].node.attrib['pos'] = '0 0 0'
    #         elif body.node.attrib['name'] == '0':
    #             body.node.attrib['pos'] = '0 0 0.75'
    #             body.joints[0].node.attrib['pos'] = '0 0 0'
    #     else:
    #         # body_fromto = body_list[-1].geoms[0].node.attrib['fromto']
    #         _, fromto_end = parse_fromto(parent_body.geoms[0].node.attrib['fromto'])
    #         # body_start = fromto_end
    #         # print("parent_fromto_end", fromto_end)
    #         child_body_pos = fromto_end
    #         body.node.attrib['pos'] =  ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in child_body_pos])

    #         body_end = np.array(body_end)
    #         body.geoms[0].end = body_end
    #         #concatenate body_start and body_end values and write to geom['fromto]
    #         # print("before")
    #         # print("body_start", body_start)
    #         # print("body_end", body_end)
    #         # pprint(vars(body.geoms[0]))
    #         # body.geoms[0].start = body_start
    #         fixed_values = [0, 0, 0] 
    #         body.joints[0].node.attrib['pos'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in fixed_values])
    #         body.geoms[0].end = body_end
    #         # print("after")
    #         # pprint(vars(body.geoms[0]))
    #         body.geoms[0].node.attrib['fromto'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in np.concatenate([fixed_values, body_end])])
    #         # print(body.geoms[0].node.attrib['fromto'])
    #         # print("AFTER REDESIGN ::: BODY START & BODY END and type of BODY END", body.geoms[0].start, body.geoms[0].end, type(body.geoms[0].end))
    #     body_list.append(body)
    #     self.bodies = body_list

        # print("BODY_start and type of BODY_start", body_start, type(body_start))
        # print("BODY_END and type of BODY_END", body_end, type(body_end))
        # print("starting sync node where final fromto is being written from")
        
        # self.sync_node(body_start, body_end) - final fix is to remove sync_node and write to xml fromto directly. 
        
    def write_xml(self, fname):
        self.tree.write(fname, pretty_print=True)

    def export_xml_string(self):
        return etree.tostring(self.tree, pretty_print=True)

    def demap_params(self, params):
        # if not np.all((params <= 1.0) & (params >= -1.0)):
        #     print(f'param out of bounds: {params}')
        params = np.clip(params, -1.0, 1.0)
        if self.param_mapping == 'sin':
            params = np.arcsin(params) / (0.5 * np.pi)
        return params

    def get_params(self, get_name=False):
        param_list = []
        for body in self.bodies:
            body.get_params(param_list, get_name)

        if not get_name:
            params = np.concatenate(param_list)
            params = self.demap_params(params)
        else:
            params = np.array(param_list)
        return params

    def map_params(self, params):
        if self.param_mapping == 'clip':
            params = np.clip(params, -1.0, 1.0)
        elif self.param_mapping == 'sin':
            params = np.sin(params * (0.5 * np.pi))
        return params

    def set_params(self, params):
        # clip params to range
        params = self.map_params(params)
        for body in self.bodies:
            params = body.set_params(params)
        assert(len(params) == 0)    # all parameters need to be consumed!
        self.sync_node()

    def rebuild(self):
        for body in self.bodies:
            body.rebuild()
            body.sync_node()

    def get_gnn_edges(self):
        edges = []
        for i, body in enumerate(self.bodies):
            if body.parent is not None:
                j = self.bodies.index(body.parent)
                edges.append([i, j])
                edges.append([j, i])
        edges = np.stack(edges, axis=1)
        return edges

def run_bfs(nodes, edges):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)  # For undirected graph
    start = 0 # node to start bfs from 
    visited = set()  # To keep track of visited nodes
    queue = deque()   # Initialize a queue for BFS
    result = []       # To store the BFS traversal result

    queue.append(start)  # Start from the initial node
    visited.add(start)   # Mark it as visited

    while queue:
        node = queue.popleft()  # Dequeue the front node
        result.append(node)     # Add it to the result

        # Visit all adjacent nodes of the current node
        for neighbor in graph[node]:
            if neighbor not in visited:
                queue.append(neighbor)  # Enqueue unvisited neighbors
                visited.add(neighbor)   # Mark them as visited

    return result

def find_parent_in_graph(node, edges):
    # This function takes in a graph node and returns the parent of that node based on the edges provided.
    #converting list of tuples to list of lists
    list_of_edges = [list(t) for t in edges]
    if node == 0:
        parent_node = None
    for i in range(1,MAX_CHILD_ALLOWED):
        if node == i:
            for index, _edge in enumerate(list_of_edges):
                if len(_edge) > 1 and _edge[1] == i:
                    parent_node = _edge[0]
    return parent_node


# def graph_to_robot(graph, xml_dir, log_dir):
#     # print("log_dir", log_dir)
#     # print("xml_dir", xml_dir)
#     '''
#     This function creates a xml_robot from given graph and saves this .xml file in xml_dir
#     '''
#     # model_name = 'base_robot'
#     cfg_path = 'assets/base.yml'
#     # model = load_model_from_path(f'assets/mujoco_envs/{model_name}.xml')
#     cfg_path = str(os.getcwd()) +"/"+ cfg_path
#     yml = yaml.safe_load(open(cfg_path, 'r'))
#     cfg = yml['robot']
#     xml_robot = Robot(cfg, xml=f'{xml_dir}/env.xml')
#     root = xml_robot.tree.getroot().find('worldbody').find('body')

#     # print("graph", type(graph))
#     # print("graph nodes", graph.nodes)
#     # print("graph edges", graph.edges)
#     nodes = graph.nodes
#     edges = graph.edges
#     # if hasattr(graph, 'node_categories'):
#     node_cat_id = graph.node_categories    # each node as each body - body is length-original(1x) or half length(0.5x) etc
#     # print("Only one type of node-body provided", graph.node_categories)

#     # elif hasattr(graph, 'edge_categories'):
#     edge_cat_id = graph.edge_categories   # describes connection slots and contains src and dest info
#     # else:
#         # print("No edge attributes provided")
    

#     _edge_cat_id = [val for j, val in enumerate(edge_cat_id) if j % 2 ==0]  # captures source slot info only
#     _edge_cat_id.append(0)   # Padding to make the last robot body has a default slot 0 positioning

#     # ****** Below code without _edge_cat_id modification is saved in Documents of my Macbook******

#     # print("_edge_cat_id", _edge_cat_id)

#     # geom_types = ['capsule']
#     # location_space = ['spot_0', 'spot_1', 'spot_2']
#     # length_space = [0.25, 0.5, 0.75, 1]

#     # cat_map = {}  # Initialize an empty dictionary to store the combinations
#     # # Loop through each combination of geom_type, location, and length
#     # for geom_type in geom_types:
#     #     for location in location_space:
#     #         for length in length_space:
#     #             # Create a unique key for each combination
#     #             key = len(cat_map)
#     #             # Add the combination to the dictionary
#     #             cat_map[key] = {'geom_type': geom_type, 'location': location, 'length': length}

#     '''
#     The edge_map and node_map are fixed length maps comprising of all the variations possible 
#     in number of slots (edge_map) and type of node (node_map)
#     '''
#     edge_map = {
#                 0: {'geom_type':'capsule', 'location':'spot_0'},
#                 1: {'geom_type':'capsule', 'location':'spot_1'},
#                 2: {'geom_type':'capsule', 'location':'spot_2'},
#                 3: {'geom_type':'capsule', 'location':'spot_3'},
#                 }

#     node_map = {0: {'length_ratio':1, 'resource':30_000},
#                 1: {'length_ratio':1, 'resource':15_000},
#                 2: {'length_ratio':1, 'resource':7_500},
#                 3: {'length_ratio':0.75, 'resource':30_000},
#                 4: {'length_ratio':0.75, 'resource':15_000},
#                 5: {'length_ratio':0.75, 'resource':7_500},
#                 6: {'length_ratio':1.25, 'resource':30_000},
#                 7: {'length_ratio':1.25, 'resource':15_000},
#                 8: {'length_ratio':1.25, 'resource':7_500}
#                 }

#     # Create an adjacency list from the given nodes and edges
#     graph = {}
#     for node in nodes:
#         graph[node] = []

#     for edge in edges:
#         parent, child = edge
#         graph[parent].append(child)

#     ind = 0  # removing first element
#     # xml_robot.bodies -> [a,b,c] then _bodies -> {1:a, 2:b, 3:c}
#     # print("xml_robot.bodies", xml_robot.bodies)
    
#     _bodies_list = xml_robot.bodies
#     # _bodies_list.pop(ind)
#     # print("xml_robot.bodies", xml_robot.bodies)
#     _bodies = dict([(order, robot_body) for order, robot_body in enumerate(_bodies_list)])
#     # print("_bodies initialized", _bodies)
    
#     # print("_bodies dictionary", _bodies)

#     bfs_list = run_bfs(nodes, edges)
#     bfs_child_list =  bfs_list.copy()
#     bfs_child_list.pop(ind)
    
#     # print("//////////////////")
#     # print("bfs_list", bfs_list)
    
#     for node in bfs_child_list:
#         # print("current node", node)
#         parent_node = find_parent_in_graph(node, edges)
#         # print("parent_node", parent_node)
#         if parent_node == None:
#             pass
#         else:
#             # print("_bodies current state", _bodies)
#             parent_body_node = _bodies[bfs_list.index(parent_node)]
#             # print("parent_body found", parent_body_node)
#             xml_robot.add_child_to_body(parent_body_node)
#             # print("after adding child", xml_robot.bodies)
#             _bodies_list = xml_robot.bodies.copy()
#             _bodies = dict([(order, robot_body) for order, robot_body in enumerate(_bodies_list)])
    
#     # print("_bodies after bfs thing", _bodies)

#     # mapping nodes and cat_id

#     node_cat_map = dict(zip(nodes, node_cat_id))
#     edge_cat_map = dict(zip(nodes, _edge_cat_id))

#     # print("node_cat_map", node_cat_map)
#     # mapping nodes and cat_id
#     body_node_map = dict(zip(xml_robot.bodies, bfs_list))
#     # print("body_node_map", body_node_map)
#     ordered_catid = list(node_cat_map.values())

#     body_cat_map = dict(zip(xml_robot.bodies, ordered_catid))
#     # print("body_cat_map", body_cat_map) 

#     # print("before redesigning the created children")
#     # for i in range(len(xml_robot.bodies)):
#     #     print("xml_robot.node.attrib['pos']", i, xml_robot.bodies[i].node.attrib['pos'])
#     # print("\n")
#     # print("********************************************************************")
#     # print("\n")
#     # print("\n")
#     # fromto_list = []
#     body_list = []
#     total_resource = 0
#     for node in nodes:
#         edge_cat_key = edge_cat_map[node]
#         child_edge_mods = edge_map[edge_cat_key]  #to identify what modifications are needed for the default body

#         node_cat_key = node_cat_map[node]
#         child_node_mods = node_map[node_cat_key]

#         parent_node = find_parent_in_graph(node, edges)
#         # print("parent_node again", parent_node)
#         if parent_node == None:
#             parent_body = None
#             # print("Found this (node,edges) with not parent_node", node, edges)
#             attach_loc = None
#             geom_type = None
#             length = None
#             resource = 0
#         else:
#             # print("_bodies current state", _bodies)
#             parent_body = _bodies[bfs_list.index(parent_node)]
#             attach_loc = child_edge_mods['location']
#             # geom_type = child_mods['geom_type']
#             length = child_node_mods['length_ratio']
#             resource = child_node_mods['resource']
#             # print("parent_body", parent_body)
#         for key, value in body_node_map.items():
#             if value == node:
#                 body = key
#         total_resource+= resource
#         # print("body_node_map", body_node_map)
#         # print("----------- Redesigning Started for body ------------", body)

#         # print("body, parent_body", body, parent_body)

#         xml_robot.redesign_bodies(body_list, body, attach_loc, geom_type, parent_body, length)
#         # print("xml_robot.bodies[-1].geoms[0].node.attrib['fromto]", xml_robot.bodies[-1].geoms[0].node.attrib['fromto'], type(xml_robot.bodies[-1].geoms[0].node.attrib['fromto']))
#         # fromto_list.append(xml_robot.bodies[-1].geoms[0].node.attrib['fromto'])
#         # print("----------- Redesigning Ended for body ------------", body)

#     os.makedirs('out', exist_ok=True)

#     # print("total_resource", total_resource)
#     # print(xml_robot.bodies)
    
#     postfix = int(time.time()) + random.randint(0,10000)
#     xml_robot.write_xml(os.path.join(os.path.abspath(log_dir), f'robot_{len(nodes)}_{total_resource}_{postfix}.xml'))
#     return os.path.join(os.path.abspath(log_dir), f'robot_{len(nodes)}_{total_resource}_{postfix}.xml')


def graph_to_robot_with_init_design(graph, xml_dir, log_dir):
    # print("log_dir", log_dir)
    # print("xml_dir", xml_dir)
    '''
    This function creates a xml_robot from given graph and saves this .xml file in xml_dir
    '''
    # model_name = 'base_robot'
    cfg_path = 'assets/base.yml'
    # model = load_model_from_path(f'assets/mujoco_envs/{model_name}.xml')
    cfg_path = str(os.getcwd()) +"/"+ cfg_path
    yml = yaml.safe_load(open(cfg_path, 'r'))
    cfg = yml['robot']
    xml_robot = Robot(cfg, xml=f'{xml_dir}/env.xml')
    root = xml_robot.tree.getroot().find('worldbody').find('body')

    nodes = graph.nodes
    edges = graph.edges
    # if hasattr(graph, 'node_categories'):
    node_cat_id = graph.node_categories    # each node as each body - body is length-original(1x) or half length(0.5x) etc
    # elif hasattr(graph, 'edge_categories'):
    edge_cat_id = graph.edge_categories   # describes connection slots and contains src and dest info

    _edge_cat_id = [val for j, val in enumerate(edge_cat_id) if j % 2 ==0]  # captures source slot info only
    _edge_cat_id.append(0)   # Padding to make the last robot body has a default slot 0 positioning

    '''
    The edge_map and node_map are fixed length maps comprising of all the variations possible 
    in number of slots (edge_map) and type of node (node_map)
    '''
    edge_map = {
                0: {'geom_type':'capsule', 'location':'spot_0'},
                1: {'geom_type':'capsule', 'location':'spot_1'},
                2: {'geom_type':'capsule', 'location':'spot_2'},
                3: {'geom_type':'capsule', 'location':'spot_3'},
                }

    node_map = {0: {'length_ratio':1, 'resource':30_000},
                1: {'length_ratio':1, 'resource':15_000},
                2: {'length_ratio':1, 'resource':7_500},
                3: {'length_ratio':0.75, 'resource':30_000},
                4: {'length_ratio':0.75, 'resource':15_000},
                5: {'length_ratio':0.75, 'resource':7_500},
                6: {'length_ratio':1.25, 'resource':30_000},
                7: {'length_ratio':1.25, 'resource':15_000},
                8: {'length_ratio':1.25, 'resource':7_500}
                }
    
        # Allowed values for length_ratio, size, and gear
    values1 = [1, 0.9, 0.8, 1.1, 1.2]  # For length_ratio, size, gear
    values2 = [1, 0.75, 0.5, 1.25, 1.5]  # For resource

    # Generate all possible combinations of (length_ratio, size, gear) and (resource)
    combinations = list(itertools.product(values1, values1, values1, values2))

    # Create the node_map with unique combinations
    node_map = {
        i: {
            'length_ratio': comb[0],
            'size': comb[1],
            'gear': comb[2],
            'resource': comb[3] * 200_000  # Use comb[3] from values2
        }
        for i, comb in enumerate(combinations)
    }



    # Create an adjacency list from the given nodes and edges
    graph = {}
    for node in nodes:
        graph[node] = []

    for edge in edges:
        parent, child = edge
        graph[parent].append(child)

    # Extract the initial list of body objects
    init_bodies_list = xml_robot.bodies
    init_bodies_count = len(init_bodies_list)

    # Find all 'body' elements in the XML tree
    bodies = xml_robot.tree.xpath('//body')
    leaf_bodies = [body for body in bodies if len(body.xpath('./body')) == 0]
    leaf_body_names = [body.attrib['name'] for body in leaf_bodies]
    # Filter the initial bodies list to find those that match leaf body names, which will then be parents for later bodies
    init_parent_bodies = [robot_body for robot_body in init_bodies_list if robot_body.name in leaf_body_names]

    _bodies = dict([(order, robot_body) for order, robot_body in enumerate(init_bodies_list)])

    bfs_list = run_bfs(nodes, edges)
    
    # print("//////////////////")
    # print("bfs_list", bfs_list)
    
    for node in bfs_list:
        parent_node = find_parent_in_graph(node, edges)
        # print("node", node)
        if parent_node == None:
            for parent_body in init_parent_bodies:
                parent_body_node = parent_body
                xml_robot.add_child_to_body(parent_body_node)
            _bodies_list = xml_robot.bodies
            # print("_bodies_list from node 0", _bodies_list)
            _bodies = dict([(order, robot_body) for order, robot_body in enumerate(_bodies_list)])
            # print("_bodies in translation", _bodies)
        else:
            # print("------------------------------------")
            child_bodies_list = xml_robot.bodies[init_bodies_count:] #contains only child bodies or gfn-first nodes ie. 0 [1111, 1111]
            # print("child_bodies_list", list(enumerate(child_bodies_list)), len(child_bodies_list))
            start_index = (bfs_list.index(parent_node)) * len(init_parent_bodies)
            # print(f"start_index for node {node}", start_index)
            # Slice the list 'child_bodies_list' to only get required number of elements into 'to_be_parents_bodies_list'
            to_be_parents_bodies_list = child_bodies_list[start_index:start_index + len(init_parent_bodies)]
            # print("to_be_parents_bodies_list", to_be_parents_bodies_list)
            for child_to_parent_body in to_be_parents_bodies_list:
                # print("child_body00000", child_body)
                for x in _bodies_list:
                    # print("x.name",x.name)
                    # print("child_body.name", child_body.name)
                    if child_to_parent_body.name == x.name:
                        parent_body_node = x
                        # print("parent_body_node", parent_body_node)
                        xml_robot.add_child_to_body(parent_body_node)
            # print("Updated _bodies_list", xml_robot.bodies)
            _bodies_list = xml_robot.bodies.copy()
            _bodies = dict([(order, robot_body) for order, robot_body in enumerate(_bodies_list)])
    # print("_bodies after bfs thing, but have not been influenced by the bfs_list", _bodies)
    # print("xml_robot.bodies", xml_robot.bodies, len(xml_robot.bodies))
    # mapping nodes and cat_id
    node_cat_map = dict(zip(nodes, node_cat_id))
    edge_cat_map = dict(zip(nodes, _edge_cat_id))
    
    # print("bfs_list", bfs_list)
    # print("len(init_parent_bodies)", len(init_parent_bodies))
    body_node_map = {}
    # Loop through the range from 0 to len(init_parent_bodies) to create and merge the dictionaries
    for i in range(len(init_parent_bodies)):
        # print("i", i)
        temp_body_node_map = dict(zip(xml_robot.bodies[init_bodies_count + i::len(init_parent_bodies)], bfs_list))
        # print("temp_body_node_map", temp_body_node_map)
        body_node_map.update(temp_body_node_map)

    body_cat_map = {body: node_cat_map[node] for body, node in body_node_map.items()}

    #BODIES CREATED ARE REDESIGNED HERE

    body_list = []
    total_resource = 0
    current_bodies_list = xml_robot.bodies
    bodies_to_redesign = [x for x in xml_robot.bodies[init_bodies_count:]]
    # print("BODIES TO REDESING", bodies_to_redesign)
    # print("body_node_map", body_node_map)
    inverse_body_node_map = {v: k for k, v in body_node_map.items()}
    for each_body in bodies_to_redesign:
        node = body_node_map[each_body]
        edge_cat_key = edge_cat_map[node]
        child_edge_mods = edge_map[edge_cat_key]  #to identify what modifications are needed for the default body

        node_cat_key = node_cat_map[node]
        child_node_mods = node_map[node_cat_key]

        current_child_body = each_body
        parent_body = each_body.parent
        attach_loc = child_edge_mods['location']
        # print("attach_loc", attach_loc)
        geom_type = None
        length = child_node_mods['length_ratio']
        size = child_node_mods['size']
        gear = child_node_mods['gear']
        resource = child_node_mods['resource']
        # print("current_child_body", current_child_body)
        # print("parent body", parent_body)
        # print("curr_res", resource)
        xml_robot.redesign_bodies_with_init_design(body_list, current_child_body, attach_loc, geom_type, parent_body, length, size, gear)

        total_resource+= resource/len(init_parent_bodies)

        # print("total resource", total_resource)

    # total_resource = 1.5 * total_resource         #for walled terrain only
    total_resource = int(total_resource) 


    # print("final total resource", total_resource)

    # print("Resources given:", total_resource)

    os.makedirs('out', exist_ok=True)
    
    postfix = int(time.time()) + random.randint(0,10000)
    xml_robot.write_xml(os.path.join(os.path.abspath(log_dir), f'robot_{len(nodes)}_{total_resource}_{postfix}.xml'))
    return os.path.join(os.path.abspath(log_dir), f'robot_{len(nodes)}_{total_resource}_{postfix}.xml')

def run(args2):
    args2 = [str(x) for x in args2]
    try:
        subprocess.check_call(args2)
        return 0
    except subprocess.CalledProcessError as e:
        print("This is the issue", e)
        return e.returncode


def simulate(env_id, robot_path, timesteps):

    # file_lock.acquire()

    # print("robot_path", robot_path)

    # Find the index of 'train' or 'valid' or 'final' in robot_path
    # Eg. robot_path = '/home/knagiredla/gflownet/src/gflownet/logs/run_exp_ant_flat@400ktimesteps_150gfniterations_3_1708761623/xmlrobots/./robot_1708767829.xml'
    index_train = robot_path.find('train')
    # print("index_train", index_train)
    index_valid = robot_path.find('valid')
    # print("index_valid", index_valid)
    index_final = robot_path.find('final')
    # print("index_final", index_final)

    #select the one that actually has the path instead of nothing
    index_robots = max(index_train, index_valid, index_final)
    # print("index_robots", index_robots)
    # Remove the part of the string starting from 'xmlrobots'
    if index_robots != -1:
        perf_path = robot_path[:index_robots + len('train') or len('valid') or len('final')]
        # print("perf_path", perf_path)

    # if task == 'hopper':
    # env_id = 'Ant-v3'
    network = 'mlp'
    total_timesteps = timesteps
    ctrl_cost_weight = 0.5
    # xml_file_path = robot_path

    # print("*****************************PPO start*****************************")

    # with file_lock:
    #   print("locked")

    py_file = ["python", "/home/knagiredla/.conda/envs/gfn_sb3/lib/python3.8/site-packages/gflownet_old/envs/run_ppo_sb3.py"]
    args = py_file 
    args += ["--env_id", env_id]
    args += ["--total_timesteps", int(total_timesteps)]
    args += ["--network", network]
    args += ["--perf_log_path", perf_path]
    args += ["--xml_file_path", robot_path]
    args += ["--ctrl_cost_weight", ctrl_cost_weight]
    run(args)
    subprocess.run("pwd", shell=True)

    with open(os.path.join(perf_path, 'rews.json')) as f:
        content = f.read()

    # Modify the text to add '[' at the start, remove ',' at the end, and add a ']'
    modified_content = '[' + content.rstrip(',') + ']'

    # Save the modified content back to a JSON file
    with open(os.path.join(perf_path, 'rew_modified.json'), 'w') as file:
        file.write(modified_content)

    # Load the modified JSON file to verify
    with open(os.path.join(perf_path, 'rew_modified.json'), 'r') as file:
        modified_data = json.load(file)
    
    for list_elem in modified_data:
        if robot_path in list_elem:
            eprewmean = list_elem[robot_path]
    
    # print("*****************************PPO end*****************************")
    # print("eprew mean", eprewmean)
    # file_lock.release()
    return eprewmean


def get_children(node):
    # Find all numbers in the string using regular expression
    digits = [int(i) for i in node.split() if i.isdigit()]
    digits = [int(d) for d in str(digits[0])]
    # print("numbers", numbers, type(numbers), len(numbers))

    first_digit = None
    other_digits = []

    if digits:
        # Extract the first number
        first_number = digits[0]
        
        # Extract the remaining numbers
        other_digits = digits[1:]
        other_digits = ''.join(map(str, other_digits))

    #the idea here is the first digit is the child number and the remaining digits refer to the parent
    child = first_digit
    parent = other_digits

    return (parent, node)

def get_attribs(root):
    attribs = []
    node_cats = []

    # reading from xml
    for elem in root.iter("geom"):
        geom_name = elem.get("name")
        attribs.append(geom_name)

    for item in attribs:
        if item is not None:
            node_cats.append(item[-1])

    return node_cats


# def robot_to_graph(robot_xml_path):

#     robot_model = load_model_from_path(robot_xml_path)
#     # print("robot_xml_path", robot_xml_path)    
#     # tree = ET.parse(robot_xml_path)
#     tree = ET.parse('assets/base_robot_modified.xml')
#     root = tree.getroot()
   
#     # Create a NetworkX graph
#     robot_graph = nx.Graph()
#     attributes = get_attribs(root)
#     #After removing world_body geoms out
#     node_attr = attributes[1:]
#     # print("attr_list", node_attr)
#     robot_nodes = []
#     # Iterate through the XML to add nodes for bodies
#     for body_elem in root.iter('body'):
#         body_name = body_elem.get('name')
#         robot_nodes.append(body_name)
#         # robot_graph.add_node(body_name)
#     # print("robot_nodes", robot_nodes
#     # print("robot_graph", robot_graph)

#     for node, attrib in zip(robot_nodes, node_attr):
#         robot_graph.add_node(robot_nodes.index(node), attr=attrib)

#     edge_list = [(0,1)]
#     for index, node in enumerate(robot_nodes):
#         if node is not None and node != '0' and  node!='1':
#             (parent, child) = get_children(node)
#             edge_list.append((robot_nodes.index(parent), robot_nodes.index(child)))
#     # print("edge_list", edge_list)
#     robot_graph.add_edges_from(edge_list)

    # print("robot_graph", robot_graph, type(robot_graph))
    # print(robot_graph.nodes[3])
    # print(robot_graph.edges)
    # print(robot_graph.graph)
    # print(robot_graph.nodes, type(robot_graph.nodes))


