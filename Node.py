import streamlit as st
import json
import uuid
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum

# --------------- Configuration and Styles ---------------
st.set_page_config(
    page_title="🤖 Advanced LangChain Agent Builder",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px 16px 10px 16px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.18);
    }
    .component-card {
        background: white;
        color: #212121;
        padding: 12px;
        border-radius: 9px;
        margin-bottom: 8px;
        border-left: 4px solid #4CAF50;
        box-shadow: 0 2px 4px rgba(50,50,93,.1);
    }
    .execution-log {
        background: #282a36;
        color: #50fa7b;
        padding: 13px;
        border-radius: 8px;
        font-family: 'Fira Mono', monospace;
        font-size: 0.9em;
        margin-bottom: 8px;
    }
    .agent-stats {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
        color: white;
        padding: 15px 10px 8px 10px;
        border-radius: 8px;
        text-align: center;
        margin: 8px 0 12px 0;
    }
</style>
""", unsafe_allow_html=True)

# --------------- Data Classes and Enums ---------------
@dataclass
class NodeConfig:
    id: str
    type: str
    name: str
    x: float
    y: float
    properties: Dict[str, Any]

@dataclass
class Connection:
    id: str
    from_node: str
    to_node: str
    from_port: str
    to_port: str
    condition: Optional[str] = None

@dataclass
class AgentFlow:
    nodes: List[NodeConfig]
    connections: List[Connection]
    metadata: Dict[str, Any]

class NodeType(Enum):
    INPUT = "input"
    LLM = "llm"
    TOOL = "tool"
    MEMORY = "memory"
    ROUTER = "router"
    OUTPUT = "output"

# --------------- Session State Initialization ---------------
def initialize_session_state():
    defaults = {
        'agent_flow': AgentFlow([], [], {}),
        'canvas_data': {'nodes': [], 'connections': []},
        'execution_log': [],
        'agent_running': False,
        'api_status': 'disconnected',
        'conversation_history': [],
        'current_agent_config': None,
        'saved_agents': {},
        'execution_stats': {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'avg_response_time': 0.0
        }
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# --------------- Component Library ---------------
components_config = {
    "Input":     {"color": "#4CAF50", "icon": "📝", "description": "User input capture"},
    "LLM":       {"color": "#2196F3", "icon": "🤖", "description": "Language model processing"},
    "Prompt":    {"color": "#9C27B0", "icon": "📋", "description": "Prompt template engine"},
    "Tool":      {"color": "#FF9800", "icon": "🔧", "description": "External tools and APIs"},
    "Memory":    {"color": "#607D8B", "icon": "🧠", "description": "Conversation memory"},
    "Router":    {"color": "#F44336", "icon": "🔀", "description": "Decision routing logic"},
    "Output":    {"color": "#8BC34A", "icon": "📤", "description": "Final output formatting"}
}

# --------------- Sidebar UI Section ---------------
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 🔐 Agent Simulation Controls")
    st.text_input("OpenAI API Key (Mocked)", type="password", help="Not used, simulation only")
    st.markdown('<span class="status-indicator status-disconnected"></span>API Not Connected', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 🧩 Component Library")
    for comp_name, comp_info in components_config.items():
        st.markdown(f"""
            <div class="component-card">
                <strong>{comp_info['icon']} {comp_name}</strong><br>
                <small>{comp_info['description']}</small>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Canvas Controls")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.canvas_data = {'nodes': [], 'connections': []}
            st.session_state.agent_flow = AgentFlow([], [], {})
            st.rerun()
    with col2:
        agent_name = st.text_input("Agent Name", value="MyAgent")
        if st.button("💾 Save", use_container_width=True):
            st.session_state.saved_agents[agent_name] = st.session_state.agent_flow
            st.success("Agent saved!")
    if st.session_state.saved_agents:
        selected_agent = st.selectbox("Load Saved Agent", [""] + list(st.session_state.saved_agents.keys()))
        if selected_agent and st.button("📂 Load Agent"):
            st.session_state.agent_flow = st.session_state.saved_agents[selected_agent]
            st.success("Loaded agent!")
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 📝 Execution Log")
    for log_entry in reversed(st.session_state.execution_log[-10:]):
        st.markdown(f'<div class="execution-log">{log_entry}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --------------- Main Header and Tabs ---------------
st.markdown('<div class="main-header"><h1>🤖 Advanced LangChain Agent Builder</h1></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🎨 Visual Builder", "⚙️ Configuration", "🚀 Execution", "📊 Analytics"])

with tab1:
    st.subheader("🎨 Visual Agent Flow")
    st.write("Below is a summary of your current agent configuration (nodes and connections):")
    nodes_df = pd.DataFrame([asdict(n) for n in st.session_state.agent_flow.nodes])
    conns_df = pd.DataFrame([asdict(c) for c in st.session_state.agent_flow.connections])
    st.write("**Nodes**:")
    st.dataframe(nodes_df if not nodes_df.empty else pd.DataFrame([{'info':'No nodes yet'}]))
    st.write("**Connections**:")
    st.dataframe(conns_df if not conns_df.empty else pd.DataFrame([{'info':'No connections yet'}]))

with tab2:
    st.subheader("⚙️ Full Agent Builder Configuration")
    st.json(json.loads(json.dumps(asdict(st.session_state.agent_flow))), expanded=False)

with tab3:
    st.subheader("🚀 Run Agent Simulation")
    input_prompt = st.text_area("User Prompt", "What is the capital of France?", key="prompt_input")
    if st.button("▶️ Simulate Run"):
        from random import choice
        simulated_response = choice([
            "The capital of France is Paris.",
            "Paris is the capital city of France.",
            "France's capital is Paris."
        ])
        st.session_state.execution_stats['total_runs'] += 1
        st.session_state.execution_stats['successful_runs'] += 1
        st.session_state.execution_log.append(
            f"Prompt: {input_prompt}\nAgent (simulated): {simulated_response}"
        )
        st.success(simulated_response)
        st.balloons()
    st.markdown("Agent runs are currently simulated. Connect real APIs in code for production.")

    stats = st.session_state.execution_stats
    st.markdown(f"""
    <div class="agent-stats">
        <strong>Agent Statistics</strong><br>
        Runs: {stats['total_runs']}<br>
        Success: {stats['successful_runs']}<br>
        Failed: {stats['failed_runs']}<br>
        Avg Time: {stats['avg_response_time']:.2f}s
    </div>
    """, unsafe_allow_html=True)

with tab4:
    st.subheader("📊 Run Analytics and History")
    stats = st.session_state.execution_stats
    chart_data = pd.DataFrame({
        "Runs": [stats['total_runs']],
        "Successes": [stats['successful_runs']],
        "Failures": [stats['failed_runs']]
    })
    st.bar_chart(chart_data.transpose())
    st.write("**Last 10 log entries:**")
    for log_entry in reversed(st.session_state.execution_log[-10:]):
        st.markdown(f'<div class="execution-log">{log_entry}</div>', unsafe_allow_html=True)

# --------------- Example: Add Node/Connection Directly (for demo) ---------------
with st.expander("🧪 For Demo: Add Demo Node/Connection"):
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Add Demo Node"):
            new_id = str(uuid.uuid4())
            demo_node = NodeConfig(id=new_id, type="Input", name=f"Input_{new_id[:4]}", x=np.random.rand()*400, y=np.random.rand()*400, properties={})
            st.session_state.agent_flow.nodes.append(demo_node)
            st.session_state.execution_log.append(f"Added demo node: {demo_node.name}")
            st.rerun()
    with col_b:
        if st.button("Add Demo Connection"):
            if len(st.session_state.agent_flow.nodes) >= 2:
                conn_id = str(uuid.uuid4())
                c = Connection(
                    id=conn_id,
                    from_node=st.session_state.agent_flow.nodes.id,
                    to_node=st.session_state.agent_flow.nodes[-1].id,
                    from_port="output",
                    to_port="input",
                )
                st.session_state.agent_flow.connections.append(c)
                st.session_state.execution_log.append(f"Added demo connection from {c.from_node[:4]} to {c.to_node[:4]}")
                st.rerun()
            else:
                st.warning("Add at least 2 nodes first.")

# --------------- Footer ---------------
st.markdown(
    "<small>© 2025 Advanced LangChain Agent Builder. For educational and prototype use.</small>",
    unsafe_allow_html=True
)
