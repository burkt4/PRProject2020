import xml.etree.cElementTree as ET

class Graph:

    def __init__(self, nodes=[], edgemode="undirected"):
        self.nodes = nodes
        self.edgemode = edgemode

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge_to_node(self, nodefrom, nodeto, cost):
        for x in self.nodes:
            print(x.id)
            print(nodefrom)
            if x.id == nodefrom:
                x.edges.append({"from":nodefrom, "to": nodeto, "cost": cost})

    def find_node_by_id(id):
        for x in self.nodes:
            if x.id == id:
                return x
        return None

    def print_graph_info(self):
        print("Nb nodes : " + str(len(self.nodes)) + "\nEdgemode : " + self.edgemode)
        for x in self.nodes:
            x.printData()

class Node:

    def __init__(self, _id, symbol):
        self.id = _id
        self.symbol = symbol
        self.edges = []
    
    def print_node_info(self):
        print("NodeID: %s \nSymbol: %s"%(self.id,self.symbol))   
        print("Edges : ")
        for x in self.edges:
            print("from " + x['from'] + " to " + x['to'] + " --> cost " + x['cost'])    

graph = None

def parse_xml(file_name):
    tree = ET.ElementTree(file=file_name)
    xml_graph = list(tree.getroot())[0] #index 0 cause only one child graph
    global graph
    graph = Graph(edgemode=xml_graph.attrib.get('edgemode'))
    nodes = []
    for xml_node in list(xml_graph):
        if xml_node.tag == 'node':
            #Add node to graph
            node = Node((xml_node.attrib.get('id')), str(xml_node[0][0].text))
            graph.add_node(node)
        elif xml_node.tag == 'edge':
            #add neighbors to node (bidirectional)
            graph.add_edge_to_node(xml_node.attrib.get('from'), xml_node.attrib.get('to'), xml_node[0][0].text)
            graph.add_edge_to_node(xml_node.attrib.get('to'), xml_node.attrib.get('from'), xml_node[0][0].text)

parse_xml("task5/gxl/16.gxl")
graph.print_graph_info()
