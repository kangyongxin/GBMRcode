import networkx as nx
import matplotlib.pyplot as plt 
import numpy as np

G = nx.petersen_graph()
G.edges[1,2]['label']="u"
G.add_node('a')
print(G.edges[1,2]['label'])
nx.draw(G)
plt.subplot(121)
nx.draw(G, with_labels=True,font_weight='bold')
# plt.subplot(122)
# nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
plt.show()
# import matplotlib.pyplot as plt
# import networkx as nx

# G = nx.house_graph()
# # explicitly set positions
# pos ={}
# pos[0]=(0,0)
# pos[1]=(1,0)
# pos[2]=(0,1)
# pos[3]=(1,1)
# pos[4]=(0.5,2.0)
# #pos = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1), 4: (1, 2.0)}
# print(list(G.neighbors(1)))
# nx.draw_networkx_nodes(G, pos, node_size=200, nodelist=[4])
# nx.draw_networkx_nodes(G, pos, node_size=300, nodelist=[0, 1, 2, 3], node_color="b")
# nx.draw_networkx_edges(G, pos, alpha=0.5, width=6)
# plt.axis("off")
# plt.show()
