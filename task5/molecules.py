import xml.etree.cElementTree as ET
import os

def load_graphs():
    graphs = []
    for filename in os.listdir('gxl'):
        graphs.append(parse_xml('gxl/'+filename))
    return graphs

def parse_xml(filename):
    tree = ET.ElementTree(file=filename)
    xml_graph = list(tree.getroot())[0] #index 0 cause only one child graph
    graph = {"filename": filename, "edgemode" : xml_graph.attrib.get('edgemode'), "nodes": []}
    nodes = []
    for xml_node in list(xml_graph):
        if xml_node.tag == 'node':
            #Add node to graph
            node = {"id": xml_node.attrib.get('id'), "symbol": str(xml_node[0][0].text), "edges": []}
            nodes.append(node)
        elif xml_node.tag == 'edge':
            #Add neighbors to node (bidirectional)
            add_edge_to_node(nodes, xml_node.attrib.get('from'), xml_node.attrib.get('to'), xml_node[0][0].text)
            add_edge_to_node(nodes, xml_node.attrib.get('to'), xml_node.attrib.get('from'), xml_node[0][0].text)
    #Add nodes to graph
    graph['nodes'] = nodes
    return graph

def add_edge_to_node(nodes, nodefrom, nodeto, cost):
    for node in nodes:
        if node['id'] == nodefrom:
            node['edges'].append({"from":nodefrom, "to": nodeto, "cost": cost})

def print_graph_info(graph):
    print("Graph from file : "+ graph['filename'])
    print("Nb nodes : " + str(len(graph['nodes'])))
    for x in graph['nodes']:
        print_node_info(x)

def print_node_info(node):
    print("NodeID: %s Symbol: %s"%(node['id'],node['symbol']))   
    print("Edges : ")
    for x in node['edges']:
        print("from " + x['from'] + " to " + x['to'] + " --> cost " + x['cost'])    


graphs = load_graphs()
print(len(graphs[300]['nodes']))