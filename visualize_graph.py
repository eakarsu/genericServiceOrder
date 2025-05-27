from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
import matplotlib.pyplot as plt
import networkx as nx

# Your existing UniversalServiceBot code here...

def visualize_graph_complete(bot, sector="food_delivery"):
    """Complete visualization of the state graph"""
    
    # Create the graph
    graph = bot.create_graph_for_sector(sector)
    
    # Method 1: Get LangGraph representations
    print("=== LANGGRAPH NATIVE OUTPUTS ===")
    
    try:
        mermaid_output = graph.get_graph().draw_mermaid()
        print("Mermaid Diagram:")
        print(mermaid_output)
        print("\n" + "="*50 + "\n")
    except Exception as e:
        print(f"Mermaid generation error: {e}")
    
    try:
        ascii_output = graph.get_graph().draw_ascii()
        print("ASCII Representation:")
        print(ascii_output)
        print("\n" + "="*50 + "\n")
    except Exception as e:
        print(f"ASCII generation error: {e}")
    
    # Method 2: Extract and visualize with NetworkX
    print("=== NETWORKX VISUALIZATION ===")
    
    # Get graph data
    graph_data = graph.get_graph()
    
    # Extract nodes and edges
    nodes = list(graph_data.nodes.keys())
    edges = [(edge.source, edge.target) for edge in graph_data.edges]
    
    print(f"Nodes: {nodes}")
    print(f"Edges: {edges}")
    
    # Create NetworkX graph
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    # Plot with matplotlib
    plt.figure(figsize=(14, 10))
    
    # Create layout
    pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
    
    # Color nodes by type
    node_colors = []
    for node in nodes:
        if node in ['finalization', 'END']:
            node_colors.append('lightgreen')
        elif node in ['cancellation']:
            node_colors.append('lightcoral') 
        elif node in ['modification']:
            node_colors.append('yellow')
        elif node in ['START']:
            node_colors.append('lightgray')
        else:
            node_colors.append('lightblue')
    
    # Draw the graph
    nx.draw(G, pos, 
            with_labels=True, 
            node_color=node_colors,
            node_size=2000,
            font_size=8,
            font_weight='bold',
            arrows=True,
            edge_color='gray',
            arrowsize=20,
            arrowstyle='->')
    
    plt.title(f"Universal Service Bot - {sector.title()} Sector\nState Transition Diagram", 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Method 3: Save Mermaid to file for external rendering
    try:
        with open(f"{sector}_state_diagram.mmd", "w") as f:
            f.write(mermaid_output)
        print(f"Mermaid diagram saved to {sector}_state_diagram.mmd")
        print("You can render this at: https://mermaid.live/")
    except:
        print("Could not save Mermaid file")

# Usage
def main_with_visualization():
    """Main function with complete visualization"""
    bot = UniversalServiceBot()
    
    # Test different sectors
    sectors = ["food_delivery", "healthcare", "auto_repair"]
    
    for sector in sectors:
        print(f"\n{'='*60}")
        print(f"VISUALIZING: {sector.upper()}")
        print(f"{'='*60}")
        
        visualize_graph_complete(bot, sector)

