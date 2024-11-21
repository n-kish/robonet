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
import multiprocessing
import itertools
import random
import subprocess
from collections import defaultdict, deque


file_lock = multiprocessing.Lock()
MAX_CHILD_ALLOWED = 100

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
    rotated_vector = np.dot(rotation_matrix, normalized_vector)
    final_vector = rotated_vector * magnitude
    x, y, z = final_vector.ravel()
    return x,y,z

class Joint:

    def __init__(self, node, body):
        self.node = node
        self.body = body
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
        self.param_inited = False
        self.pos = parse_vec(node.attrib['pos'])
        if self.local_coord:
            self.pos += body.pos
    
    def __repr__(self):
        return 'joint_' + self.name  

    def parse_param_specs(self):
        pass

    def sync_node(self, new_pos):
        if new_pos is not None:
            pos = new_pos
        else:
            pos = self.pos
        self.name = self.body.name + '_joint'
        self.node.attrib['name'] = self.name
        self.node.attrib['pos'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in pos])
        if self.actuator is not None:
            self.actuator.sync_node()
        
class Geom:

    def __init__(self, node, body):
        self.node = node
        self.body = body
        self.local_coord = body.local_coord
        self.name = node.attrib.get('name', '')
        self.type = node.attrib['type']
        self.param_inited = False
        self.size = parse_vec(node.attrib['size'])
        if self.type == 'capsule':
            self.start, self.end = parse_fromto(node.attrib['fromto'])
            if self.local_coord:
                self.start += body.pos
                self.end += body.pos
            if body.bone_start is None:
                self.bone_start = self.start.copy()
                body.bone_start = self.bone_start.copy()
            else:
                self.bone_start = body.bone_start.copy()
            self.ext_start = np.linalg.norm(self.bone_start - self.start)

    def __repr__(self):
        return 'geom_' + self.name

    def parse_param_specs(self):
        pass

    def update_start(self):
        if self.type == 'capsule':
            vec = self.bone_start - self.end
            self.start = self.bone_start + vec * (self.ext_start / np.linalg.norm(vec))

    def sync_node(self, body_start=None, body_end=None):
        self.node.attrib.pop('name', None)
        self.node.attrib['size'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in self.size])
        if self.type == 'capsule':
            start = self.start - self.body.pos if self.local_coord else self.start
            end = self.end - self.body.pos if self.local_coord else self.end
            if body_start is not None:
                # print("body start is not none, i.e. received new body_start and body_end")
                self.node.attrib['fromto'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in np.concatenate([body_start, body_end])])
            else:
                # print("body start is none, i.e. didnot receive new body_start and body_end")
                self.node.attrib['fromto'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in np.concatenate([start, end])])


class Actuator:

    def __init__(self, node, joint):
        self.node = node
        self.joint = joint
        # self.cfg = joint.cfg
        self.joint_name = node.attrib['joint']
        self.name = self.joint_name
        # self.parse_param_specs()
        self.param_inited = False
        # tunable parameters
        self.gear = float(node.attrib['gear'])

    def sync_node(self):
        self.node.attrib['gear'] = f'{self.gear:.6f}'.rstrip('0').rstrip('.')
        self.name = self.joint.name
        self.node.attrib['name'] = self.name
        self.node.attrib['joint'] = self.joint.name


class Body:

    def __init__(self, node, parent_body, robot):
        self.node = node
        self.parent = parent_body
        if parent_body is not None:
            parent_body.child.append(self)
            parent_body.cind += 1
            self.depth = parent_body.depth + 1
        else:
            self.depth = 0
        self.robot = robot
        self.tree = robot.tree
        self.local_coord = robot.local_coord
        self.name = node.attrib['name'] if 'name' in node.attrib else self.parent.name + f'_child{len(self.parent.child)}'
        # print("body's self.name", self.name)
        self.child = []
        self.cind = 0
        self.pos = parse_vec(node.attrib['pos'])
        # print("pos from body class before changes", self.pos)

        if self.local_coord and parent_body is not None:
            self.pos += parent_body.pos
        
        self.bone_start = self.pos.copy()
        self.joints = [Joint(x, self) for x in node.findall('joint[@type="hinge"]')] + [Joint(x, self) for x in node.findall('joint[@type="free"]')]
        self.geoms = [Geom(x, self) for x in node.findall('geom[@type="capsule"]')] + [Geom(x, self) for x in node.findall('geom[@type="sphere"]')]
        # self.parse_param_specs()
        self.param_inited = False
        # parameters
        self.bone_end = None
        self.bone_offset = None

    def __repr__(self):
        # print("Name", self.name)
        return 'body_' + self.name
    
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
         
        for joint in self.joints:
            joint.sync_node(body_start)
        for geom in self.geoms:
            geom.sync_node(body_start=None, body_end=None)
        

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
        # if self.parent is None and self.cfg.get('no_root_offset', False):
        #     self.bone_end = self.bone_start
        self.sync_geom()
        self.sync_joint()

class Robot:

    def __init__(self, xml, is_xml_str=False):
        self.bodies = []
        # self.cfg = cfg
        # self.param_mapping = cfg.get('param_mapping', 'clip')
        self.tree = None    # xml tree
        self.load_from_xml(xml, is_xml_str)
        self.init_bodies()
        # self.param_names = self.get_params(get_name=True)
        # self.init_params = self.get_params()

    def load_from_xml(self, xml, is_xml_str=False):
        parser = XMLParser(remove_blank_text=True)
        self.tree = parse(BytesIO(xml) if is_xml_str else xml, parser=parser)
        # self.local_coord = self.tree.getroot().find('.//compiler').attrib['coordinate'] == 'local'
        self.local_coord = None
        root = self.tree.getroot().find('worldbody').find('body')
        self.add_body(root, None)

    def add_body(self, body_node, parent_body):
        body = Body(body_node, parent_body, self)
        # print("body created", body)
        self.bodies.append(body)
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
        child_body = Body(child_node, parent_body, self)
        actu_node = parent_body.tree.getroot().find("actuator")
        for joint in child_body.joints:
            new_actu_node = deepcopy(actu_node.find(f'motor[@joint="{joint.name}"]'))
            actu_node.append(new_actu_node)
            joint.actuator = Actuator(new_actu_node, joint)
        child_body.bone_offset = parent_body.bone_offset.copy()
        # child_body.param_specs = deepcopy(parent_body.param_specs)
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
        
        if parent_body is None:
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


            fixed_size = 0.08
            body.geoms[0].node.attrib['size'] = str(fixed_size * size_factor)

            body.geoms[0].start = fixed_pos_values
            body.geoms[0].end = length_new
            # pprint(vars(body.geoms[0]))
            body.geoms[0].node.attrib['fromto'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in np.concatenate([fixed_pos_values, length_new])])
            # print(body.geoms[0].node.attrib['fromto'])
            # print("AFTER REDESIGN ::: BODY START & BODY END and type of BODY END", body.geoms[0].start, body.geoms[0].end, type(body.geoms[0].end))
            fixed_gear = 150
          
            # body.joints[0].actuator.node.attrib['gear'] = str(int(fixed_gear * gear_factor))

        body_list.append(body)
        self.bodies = body_list


    def redesign_bodies(self, body_list, body, attach_loc=None, geom_type=None, parent_body=None, length_ratio=None):
        #modify the created robot bodies based on the required specification
        # print(vars(body.geoms[0])
        
        #these are base parent values and will be same for any body consider for redesign
        
        # print("body", body)
        
        # print("parent_body", parent_body)

        if body.name != '0':
            # body is not of the same type as body_0
            body_start = body.geoms[0].start 
            body_end = body.geoms[0].end
            length = (body_end - body_start) * length_ratio if length_ratio is not None else body_end - body_start         # The final length is a result of (default length * length ratio) - Apr23 Kishan
        # print("length", length, type(length))

        # print("BEFORE REDESIGN ::: BODY START & BODY END and type of BODY END", body.geoms[0].start, body.geoms[0].end , type(body.geoms[0].end))
        if attach_loc is None:
            pass
        if attach_loc == 'spot_0':
            angle = 0
            new_x, new_y, new_z = transform_body(angle, length) 
            body_end = [new_x, new_y, new_z]
        elif attach_loc == 'spot_1':
            angle = 30
            new_x, new_y, new_z = transform_body(angle, length) 
            body_end = [new_x, new_y, new_z]
            # print("body_end when spot_2 is selected", body_end)
        elif attach_loc == 'spot_2':
            angle = 60
            new_x, new_y, new_z = transform_body(angle, length) 
            body_end = [new_x, new_y, new_z]
            # print("body_end when spot_3 is selected", body_end)
        elif attach_loc == 'spot_3':
            angle = 90
            new_x, new_y, new_z = transform_body(angle, length) 
            body_end = [new_x, new_y, new_z]
        # print("Parent_body just before fail", parent_body)
        # print("TEST body.joints.mode print", body.joints[0].node, len(body.joints))
        # print("body_list", body_list)
        # Using the parent's fromto values to set connnection point between parent and current body (i.e the child)    
        if parent_body is None:
            # print("parent_body", parent_body)
            if body.node.attrib['name'] == '1':
                body.node.attrib['pos'] = '0.2 0.2 0'
                body.joints[0].node.attrib['pos'] = '0 0 0'
            elif body.node.attrib['name'] == '0':
                body.node.attrib['pos'] = '0 0 0.75'
                body.joints[0].node.attrib['pos'] = '0 0 0'
            # exit()
        else:
            # body_fromto = body_list[-1].geoms[0].node.attrib['fromto']
            _, fromto_end = parse_fromto(parent_body.geoms[0].node.attrib['fromto'])
            # body_start = fromto_end
            # print("parent_fromto_end", fromto_end)
            child_body_pos = fromto_end
            body.node.attrib['pos'] =  ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in child_body_pos])

            body_end = np.array(body_end)
            body.geoms[0].end = body_end
            #concatenate body_start and body_end values and write to geom['fromto]
            # print("before")
            # print("body_start", body_start)
            # print("body_end", body_end)
            # pprint(vars(body.geoms[0]))
            # body.geoms[0].start = body_start
            fixed_values = [0, 0, 0] 
            body.joints[0].node.attrib['pos'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in fixed_values])
            body.geoms[0].end = body_end
            # print("after")
            # pprint(vars(body.geoms[0]))
            body.geoms[0].node.attrib['fromto'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in np.concatenate([fixed_values, body_end])])
            # print(body.geoms[0].node.attrib['fromto'])
            # print("AFTER REDESIGN ::: BODY START & BODY END and type of BODY END", body.geoms[0].start, body.geoms[0].end, type(body.geoms[0].end))
        body_list.append(body)
        self.bodies = body_list
        
    def write_xml(self, fname):
        self.tree.write(fname, pretty_print=True)

    def export_xml_string(self):
        return etree.tostring(self.tree, pretty_print=True)

    def demap_params(self, params):
        # if not np.all((params <= 1.0) & (params >= -1.0)):
        #     print(f'param out of bounds: {params}')
        params = np.clip(params, -1.0, 1.0)
        # if self.param_mapping == 'sin':
        #     params = np.arcsin(params) / (0.5 * np.pi)
        return params

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

def graph_to_robot_with_init_design(graph, xml_dir, log_dir, exp_method, min_resource, flag):
    '''
    This function creates a xml_robot from given graph and saves this .xml file in xml_dir
    '''

    xml_robot = Robot(xml=f'{xml_dir}/env.xml')
    root = xml_robot.tree.getroot().find('worldbody').find('body')

    nodes = graph.nodes
    edges = graph.edges
    node_cat_id = graph.node_categories    # each node as each body - body is length-original(1x) or half length(0.5x) etc
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
                3: {'geom_type':'capsule', 'location':'spot_3'}
                }

    # node_map = {0: {'length_ratio':0.5},
    #              1: {'length_ratio':0.25},
    #              2: {'length_ratio':0.75}
    #              }

    # Allowed values for length_ratio, size, and gear

    if exp_method in ["CA", "GSCA", "linearscaling"]:
        values1 = [1, 0.9, 0.8, 1.1, 1.2]  # For length_ratio, size, gear
        if exp_method in ["linearscaling"]:
            values2 = [1]  # For resource
        else:
            values2 = [1, 1.25, 1.5]


        # values1 = [1, 0.9, 0.8, 1.1, 1.2]  # For length_ratio, size, gear
        # values2 = [1, 0.75, 0.5, 1.25, 1.5]  # For resource

        # Generate all possible combinations of (length_ratio, size, gear) and (resource)
        combinations = list(itertools.product(values1, values1, values2))

        # Create the node_map with unique combinations
        node_map = {
            i: {
                'length_ratio': comb[0],
                'size': comb[1],
                # 'gear': comb[2],
                'resource': comb[2]  # Use comb[3] from values2
            }
            for i, comb in enumerate(combinations)
        }
    else:    
        values = [1, 0.9, 0.8, 1.1, 1.2]
        # Generate all possible combinations of (length_ratio, size, gear)
        combinations = list(itertools.product(values, repeat=2))

        # Create the node_map with unique combinations
        # node_map = {i: {'length_ratio': comb[0], 'size': comb[1], 'gear': comb[2]} for i, comb in enumerate(combinations)}
        node_map = {i: {'length_ratio': comb[0], 'size': comb[1]} for i, comb in enumerate(combinations)}


    # Create an adjacency list from the given nodes and edges
    graph = {}
    for node in nodes:
        graph[node] = []

    for edge in edges:
        parent, child = edge
        graph[parent].append(child)

    # print("graph adj list", graph)

     
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
    
    '''
    21/08:
    There is a possibilitythat 'bfs_child_list' can be empty after the first node is popped. In that case, we
    may directly create the xml file to be put into the simulator, without redesigning any parts through the
    redesign.bodies method. - Changed bfs_child_list to bfs_list
    '''
    
    # Eg. bfs_list = [0,1,4,2,3]
    for node in bfs_list:
        parent_node = find_parent_in_graph(node, edges)
        if parent_node == None:
            for parent_body in init_parent_bodies:
                parent_body_node = parent_body
                xml_robot.add_child_to_body(parent_body_node)
            _bodies_list = xml_robot.bodies
            _bodies = dict([(order, robot_body) for order, robot_body in enumerate(_bodies_list)])
            # print("_bodies in translation", _bodies)
        else:
            child_bodies_list = xml_robot.bodies[init_bodies_count:] #contains only child bodies or gfn-first nodes ie. 0 [1111, 1111]
            start_index = (bfs_list.index(parent_node)) * len(init_parent_bodies)
            # Slice the list 'child_bodies_list' to only get required number of elements into 'to_be_parents_bodies_list'
            to_be_parents_bodies_list = child_bodies_list[start_index:start_index + len(init_parent_bodies)]

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
    
    # mapping nodes and cat_id
    node_cat_map = dict(zip(nodes, node_cat_id))
    edge_cat_map = dict(zip(nodes, _edge_cat_id))
    
    body_node_map = {}
    # Loop through the range from 0 to 3 to create and merge the dictionaries
    for i in range(len(init_parent_bodies)):
        temp_body_node_map = dict(zip(xml_robot.bodies[init_bodies_count + i::len(init_parent_bodies)], bfs_list))
        body_node_map.update(temp_body_node_map)

    body_cat_map = {body: node_cat_map[node] for body, node in body_node_map.items()}


    #BODIES CREATED ARE REDESIGNED HERE

    body_list = []
    total_resource = 0
    current_bodies_list = xml_robot.bodies
    bodies_to_redesign = [x for x in xml_robot.bodies[init_bodies_count:]]
    # print("BODIES TO REDESING", bodies_to_redesign)

    for each_body in bodies_to_redesign:
        # print("each_body", each_body)
        node = body_node_map[each_body]
        edge_cat_key = edge_cat_map[node]
        child_edge_mods = edge_map[edge_cat_key]  #to identify what modifications are needed for the default body

        node_cat_key = node_cat_map[node]
        child_node_mods = node_map[node_cat_key]

        child = each_body
        parent_body = each_body.parent
        # print("parent_body", parent_body)
        attach_loc = child_edge_mods['location']
        # print("attach_loc", attach_loc)
        geom_type = None
        length = child_node_mods['length_ratio']
        size = child_node_mods['size']
        # gear = child_node_mods['gear']
        xml_robot.redesign_bodies_with_init_design(body_list, child, attach_loc, geom_type, parent_body, length, size)

        if exp_method in ["CA", "GSCA", "linearscaling"]:
            resource_factor = child_node_mods['resource']
            total_resource+= resource_factor*int(min_resource)                           #/len(init_parent_bodies)


    os.makedirs('out', exist_ok=True)
    postfix = int(time.time()) + random.randint(0,10000)
    
    # if exp_method in ["CA", "GSCA"]:
    #     if flag == "sampler_eval":
    #         # print("entered flag smapler_eval")
    #         xml_robot.write_xml(os.path.join(os.path.abspath(log_dir), f'robot_{len(nodes)}_{str(total_resource)}_{postfix}.xml'))
    #     return os.path.join(os.path.abspath(log_dir), f'robot_{len(nodes)}_{str(total_resource)}_{postfix}.xml')
    # else:
    # xml_robot.write_xml(os.path.join(os.path.abspath(log_dir), f'robot_{len(nodes)}_{postfix}.xml'))
    # return os.path.join(os.path.abspath(log_dir), f'robot_{len(nodes)}_{postfix}.xml')

    xml_robot.write_xml(os.path.join(os.path.abspath(log_dir), f'robot_{len(nodes)}_{str(total_resource)}_{postfix}.xml'))
    return os.path.join(os.path.abspath(log_dir), f'robot_{len(nodes)}_{str(total_resource)}_{postfix}.xml')


def run(args2):
    args2 = [str(x) for x in args2]
    try:
        subprocess.check_call(args2)
        return 0
    except subprocess.CalledProcessError as e:
        print("This is the issue", e)
        return e.returncode

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


