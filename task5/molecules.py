import xml.etree.cElementTree as ET
import os
from scipy.optimize import linear_sum_assignment
import numpy as np

def load_graphs(target=None):
    graphs = []
    if target is None:
        for filename in os.listdir('gxl'):
            graphs.append(parse_xml('gxl/'+filename))
    else:
        for filename in target:
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

def diracMatrix(g1, g2, c_n, c_e):
    n = len(g1['nodes'])
    m = len(g2['nodes'])
    matrix = np.full((n+m, n+m), fill_value=np.Infinity)
    for i, node1 in enumerate(g1['nodes']):
        for j, node2 in enumerate(g2['nodes']):
            # Substitutions
            matrix[i][j] = cost_edit_node(node1, node2, c_n, c_e)
            # Dummy assignements
            matrix[n+j][m+i] = cost_edit_node(None, None, c_n, c_e)
    # Deletions
    for i, node1 in enumerate(g1['nodes']):
        # Diagonal only
        matrix[i][m+i] = cost_edit_node(node1, None, c_n, c_e)
    # Insertions
    for j, node2 in enumerate(g2['nodes']):
        # Diagonal only
        matrix[n+j][j] = cost_edit_node(None, node2, c_n, c_e)


    return matrix

def cost_edit_node(n1, n2, c_n, c_e):
    #dummy assignement
    cost = 0
    if n1 is not None:
        # substitution
        if n2 is not None:
            if n1['symbol'] != n2['symbol']:
                cost = 2*c_n
                # TODO : edge assignement cost
                cost += abs(len(n1['edges']) - len(n2['edges']))*c_e
        # deletion
        else:
            cost = c_n
            cost += len(n1['edges'])*c_e
    # insertion
    elif n2 is not None:
        cost = c_n
        cost += len(n2['edges'])*c_e
    return cost

def ged(g1, g2, c_n, c_e):
    cost = diracMatrix(g1, g2, c_n, c_e)
    row_ind, col_ind = linear_sum_assignment(cost)
    return cost[row_ind, col_ind].sum()

# source: https://stackoverflow.com/questions/1518522/find-the-most-common-element-in-a-list
def most_common(lst):
    return max(set(lst), key=lst.count)




#---------------------------
# Program
#---------------------------


# 1. Data Load
with open('train.txt', 'r') as f:
    lines = f.readlines()
    files = []
    classes = []
    for line in lines:
        line = line.split()
        files.append(line[0]+'.gxl')
        classes.append(line[1])
train_graphs = load_graphs(files)
train_classes = classes

with open('valid.txt', 'r') as f:
    lines = f.readlines()
    files = []
    classes = []
    for line in lines:
        line = line.split()
        files.append(line[0]+'.gxl')
        classes.append(line[1])
val_graphs = load_graphs(files)
val_classes = classes

# 2. parameters
c_n = 1
c_e = 1
K = 3

# 3. for each mol. in valid.txt compute distance with every mol. in train.txt
sizeV = len(val_graphs)
sizeT = len(train_graphs)
distances = np.zeros((sizeV, sizeT))
for i in range(sizeV):
    for j in range(sizeT):
        distances[i][j] = ged(val_graphs[i], train_graphs[j], c_n, c_e)


# 4. Compute accuracy
correct = 0
total = distances.shape[0]
for i in range(distances.shape[0]):
    votes = [train_classes[i] for i in np.argsort(distances[i])[:K]]
    predict = most_common(votes)
    if predict == val_classes[i]:
        correct += 1
print(correct/total)
print(correct)
print(total)


# 5. graph edit optimization
# split train set in 2 subset
training = train_graphs[:sizeT//3]
validation = train_graphs[sizeT//3:]
classesT = train_classes[:sizeT//3]
classesV = train_classes[sizeT//3:]
for c_n in [1, 2, 3]:
    for c_e in [0, 1, 2, 3]:
        # distances matrix
        sizeV = len(validation)
        sizeT = len(training)
        distances = np.zeros((sizeV, sizeT))
        for i in range(sizeV):
            for j in range(sizeT):
                distances[i][j] = ged(validation[i], training[j], c_n, c_e)
        # K optimization
        for K in [1, 3, 5, 7]:
            # Compute accuray
            correct = 0
            total = distances.shape[0]
            for i in range(distances.shape[0]):
                votes = [classesT[i] for i in np.argsort(distances[i])[:K]]
                predict = most_common(votes)
                if predict == classesV[i]:
                    correct += 1
            print('c_n=' +str(c_n)+', c_e= '+str(c_e)+', K= '+str(K))
            print(correct/total)
