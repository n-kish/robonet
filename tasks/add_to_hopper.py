import xml.etree.ElementTree as ET

# Example graph data
bodies_to_add = [
    {"name": "body2", "pos": "0 0 0.5", "length": 0.3},
    {"name": "body3", "pos": "0 0 0.8", "length": 0.4},
]

# Load the existing XML
tree = ET.parse('assets/base_hopper_flat.xml')
root = tree.getroot()

# Find the worldbody element
worldbody = root.find('worldbody')

# Function to create a new body element
def create_body_element(body):
    body_elem = ET.Element('body', name=body['name'], pos=body['pos'])
    joint_elem = ET.Element('joint', axis="0 0 1", name=f"{body['name']}_joint", pos="0 0 0", type="hinge")
    geom_elem = ET.Element('geom', friction="0.9", name=f"{body['name']}_geom", size=f"0.05 {body['length']}", type="capsule")
    
    body_elem.append(joint_elem)
    body_elem.append(geom_elem)
    
    return body_elem

# Add new bodies to the worldbody
for body in bodies_to_add:
    new_body = create_body_element(body)
    worldbody.append(new_body)

# Save the modified XML
tree.write('assets/modified_hopper_flat.xml')