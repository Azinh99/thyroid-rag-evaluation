import os
import glob
import streamlit as st
from main import create_kg_from_multiple_txt_files, get_nodes_and_rels
from llm_df import chat_with_llm

st.set_page_config(page_title="Knowledge Graph + LLM Demo", layout="wide")
st.title("Knowledge Graph + LLM Demo")
st.write("Streamlit is running!")

# Section 1: Build KG from TXT files
st.subheader("Build Knowledge Graph from TXT files")
if st.button("Process all TXT files in data/"):
    txt_files = glob.glob(os.path.join("data", "*.txt"))
    if txt_files:
        create_kg_from_multiple_txt_files(txt_files, output_folder="data")
        st.success(f"Processed {len(txt_files)} TXT files and inserted into Neo4j!")
    else:
        st.warning("No TXT files found in data/ folder.")

# Section 2: Display Graph
st.subheader("Graph Visualization")
limit = st.slider("Number of relationships to display", min_value=10, max_value=200, value=50, step=10)
graph_data = get_nodes_and_rels(limit=limit)

if graph_data["nodes"]:
    try:
        from streamlit_agraph import agraph, Node, Edge, Config
        nodes = [Node(id=str(n.get("id")), label=str(n.get("name", "")), size=20) for n in graph_data["nodes"]]
        edges = [Edge(source=str(r["start"]), target=str(r["end"]), label=str(r.get("type", ""))) for r in graph_data["relationships"]]

        config = Config(width=900, height=600, directed=True,
                        nodeHighlightBehavior=True, highlightColor="#F39C12")
        agraph(nodes=nodes, edges=edges, config=config)
    except ImportError:
        st.warning("Please install `streamlit-agraph` (pip install streamlit-agraph) to see graph visualization.")
else:
    st.info("ℹ️ No graph data found in Neo4j yet. Try processing TXT files first.")

# Section 3: Chat with LLM
st.subheader("Chat with LLM")
user_input = st.text_input("Your message")
if user_input:
    try:
        response = chat_with_llm(user_input)
        st.write(response if response else "No response from LLM.")
    except Exception as e:
        st.error(f"LLM Error: {e}")