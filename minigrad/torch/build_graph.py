# This Module Exports the build_graph function which constructs the computational graph for backpropagation. 
from graphviz import Digraph
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io



def trace(root):
  nodes, edges = set(), set()
  def build(v):
      nodes.add(v)
      if v._ctx is not None:
        for child in v._ctx[2] if v._ctx[2] else []:
            edges.add((id(child), id(v)))
            build(child)
  build(root)
  return nodes,edges

def draw_dot(root):
  dot = Digraph(format='png',graph_attr={'rankdir':'LR'})
  nodes, edges = trace(root)
  
  # Create a mapping of node IDs to nodes for edge drawing
  id_to_node = {id(n): n for n in nodes}
  
  for n in nodes:
    uid = str(id(n))
    dot.node(name=uid, label="{data %s | grad %s}" % (str(n.data.shape), str(n.grad.shape) if n.grad is not None else "None"),shape='record')
    if n._ctx:
      dot.node(name=uid+n._ctx[0].__name__,label=n._ctx[0].label,shape='oval')
      dot.edge(uid+n._ctx[0].__name__,uid)
  
  # Draw edges using the ID mapping
  for n1_id, n2_id in edges:
    n1 = id_to_node[n1_id]
    n2 = id_to_node[n2_id]
    dot.edge(str(n1_id), str(n2_id)+n2._ctx[0].__name__)
  
  # Render to a PNG in memory
  png_data = dot.pipe(format='png')
  
  # Display using matplotlib
  plt.figure(figsize=(10, 8))
  plt.imshow(mpimg.imread(io.BytesIO(png_data)))
  plt.axis('off')
  plt.show()
  
  return dot