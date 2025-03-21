import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# === Load Graph Data ===
df = pd.read_csv('merged_file.csv', index_col=0)
G = nx.DiGraph()

# Build Graph
for source in df.index:
    for target in df.columns:
        weight = df.loc[source, target]
        if weight > 0:
            G.add_edge(source, target, weight=weight)

# === Step 1: Compute Average Shortest Path to 'happy' and 'stflife' ===
central_nodes = ['happy', 'stflife']
all_nodes = list(G.nodes())
node_avg_dist = {}

for node in all_nodes:
    if node in central_nodes:
        continue
    try:
        # Shortest path lengths from node to central_nodes
        dists = []
        for central in central_nodes:
            if nx.has_path(G, node, central):
                dist = nx.shortest_path_length(G, source=node, target=central, weight='weight')
                dists.append(dist)
        if dists:
            avg_dist = sum(dists) / len(dists)
            node_avg_dist[node] = avg_dist
    except:
        continue  # skip nodes with no path

# === Step 2: Get Top 10 Closest Nodes ===
sorted_by_closeness = sorted(node_avg_dist.items(), key=lambda x: x[1])
top_10_nodes = [node for node, _ in sorted_by_closeness[:10]]

print("\nTop 10 Nodes Closest to 'happy' and 'stflife':")
for node, dist in sorted_by_closeness[:10]:
    print(f"{node}: Avg Distance = {dist:.4f}")

# === Step 3: Load Categories ===
var_df = pd.read_csv('alldata.csv', header=None, encoding='latin1')
var_to_cat = dict(zip(var_df.iloc[:, 2], var_df.iloc[:, 1]))

# Map Categories to Colors
unique_categories = list(set(var_to_cat.get(node, 'Unknown') for node in top_10_nodes + central_nodes))
color_list = list(mcolors.TABLEAU_COLORS.values())
category_color_map = dict(zip(unique_categories, color_list[:len(unique_categories)]))

# Assign node colors
nodes_to_plot = top_10_nodes + central_nodes
node_colors = []
for node in nodes_to_plot:
    category = var_to_cat.get(node, 'Unknown')
    color = category_color_map.get(category, 'gray')
    node_colors.append(color)

# === Step 4: Create Subgraph and Plot ===
subG = G.subgraph(nodes_to_plot)
plt.figure(figsize=(12, 9))
pos = nx.spring_layout(subG, seed=42)

# Draw nodes
nx.draw(subG, pos, with_labels=True, node_color=node_colors, node_size=1800,
        font_size=10, edge_color='gray')

# Legend
for cat, color in category_color_map.items():
    plt.scatter([], [], color=color, label=cat)
plt.legend(title="Categories", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.title("Top 10 Nodes Closest to 'happy' and 'stflife' (by Avg Shortest Path)", fontsize=14)
plt.tight_layout()
plt.show()
