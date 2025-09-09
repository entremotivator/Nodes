import streamlit as st
import json
import openai
from typing import Dict, List, Any, Optional
import uuid
import re
import requests
import time
from datetime import datetime
import base64
import io
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum

# Configure Streamlit page
st.set_page_config(
    page_title="Advanced LangChain Agent Builder",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 30px;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .sidebar-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .component-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 4px solid #4CAF50;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .component-card:hover {
        transform: translateY(-2px);
    }
    .json-output {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
        font-family: 'Courier New', monospace;
    }
    .execution-log {
        background: #1e1e1e;
        color: #00ff00;
        padding: 15px;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        max-height: 400px;
        overflow-y: auto;
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-connected { background: #4CAF50; }
    .status-disconnected { background: #f44336; }
    .status-warning { background: #ff9800; }
    .agent-stats {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    .tool-result {
        background: #e8f5e8;
        border: 1px solid #4CAF50;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .error-message {
        background: #ffebee;
        border: 1px solid #f44336;
        color: #d32f2f;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Data classes for better structure
@dataclass
class NodeConfig:
    id: str
    type: str
    name: str
    x: float
    y: float
    properties: Dict[str, Any]
    inputs: List[str]
    outputs: List[str]

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
    PROMPT = "prompt"
    PARSER = "parser"
    VALIDATOR = "validator"
    WEBHOOK = "webhook"

# Initialize session state with comprehensive structure
def initialize_session_state():
    defaults = {
        'agent_flow': AgentFlow([], [], {}),
        'selected_node': None,
        'canvas_data': {'nodes': [], 'connections': []},
        'execution_log': [],
        'agent_running': False,
        'openai_client': None,
        'api_status': 'disconnected',
        'conversation_history': [],
        'current_agent_config': None,
        'saved_agents': {},
        'execution_stats': {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'avg_response_time': 0
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Sidebar configuration
st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
st.sidebar.markdown("### 🔐 OpenAI Configuration")

# OpenAI API Key input
api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Enter your OpenAI API key to enable real agent execution"
)

if api_key:
    try:
        openai.api_key = api_key
        client = openai.OpenAI(api_key=api_key)
        # Test the API key
        client.models.list()
        st.session_state.openai_client = client
        st.session_state.api_status = 'connected'
        st.sidebar.markdown(
            '<span class="status-indicator status-connected"></span>API Connected',
            unsafe_allow_html=True
        )
    except Exception as e:
        st.session_state.api_status = 'error'
        st.sidebar.markdown(
            '<span class="status-indicator status-disconnected"></span>API Error',
            unsafe_allow_html=True
        )
        st.sidebar.error(f"API Error: {str(e)[:50]}...")
else:
    st.sidebar.markdown(
        '<span class="status-indicator status-disconnected"></span>API Not Connected',
        unsafe_allow_html=True
    )

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Advanced component library
st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
st.sidebar.markdown("### 🧩 Component Library")

components_config = {
    "Input": {
        "color": "#4CAF50",
        "icon": "📝",
        "description": "User input capture",
        "properties": {
            "input_type": ["text", "file", "voice", "structured"],
            "validation": "",
            "placeholder": "Enter your message...",
            "required": True
        }
    },
    "LLM": {
        "color": "#2196F3",
        "icon": "🤖",
        "description": "Language model processing",
        "properties": {
            "model": ["gpt-4o", "gpt-4", "gpt-3.5-turbo", "gpt-4o-mini"],
            "temperature": 0.7,
            "max_tokens": 2000,
            "system_prompt": "You are a helpful assistant.",
            "stream": True,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
    },
    "Prompt": {
        "color": "#9C27B0",
        "icon": "📋",
        "description": "Prompt template engine",
        "properties": {
            "template": "Answer the question: {question}",
            "variables": ["question"],
            "template_type": ["simple", "chat", "few_shot"],
            "examples": []
        }
    },
    "Tool": {
        "color": "#FF9800",
        "icon": "🔧",
        "description": "External tools and APIs",
        "properties": {
            "tool_type": ["web_search", "calculator", "file_reader", "api_call", "database", "python_code"],
            "endpoint": "",
            "method": "GET",
            "headers": {},
            "parameters": {},
            "timeout": 30,
            "retry_count": 3
        }
    },
    "Memory": {
        "color": "#607D8B",
        "icon": "🧠",
        "description": "Conversation memory",
        "properties": {
            "memory_type": ["conversation_buffer", "conversation_summary", "vector_store", "entity_memory"],
            "max_tokens": 1000,
            "return_messages": True,
            "summary_template": "Summarize the conversation so far."
        }
    },
    "Router": {
        "color": "#F44336",
        "icon": "🔀",
        "description": "Decision routing logic",
        "properties": {
            "routing_type": ["semantic", "keyword", "model_based", "rule_based"],
            "conditions": {},
            "default_route": "",
            "confidence_threshold": 0.8
        }
    },
    "Parser": {
        "color": "#795548",
        "icon": "🔍",
        "description": "Output parsing and formatting",
        "properties": {
            "parser_type": ["json", "xml", "regex", "structured", "pydantic"],
            "schema": {},
            "format_template": "",
            "error_handling": "strict"
        }
    },
    "Validator": {
        "color": "#E91E63",
        "icon": "✅",
        "description": "Input/output validation",
        "properties": {
            "validation_rules": {},
            "error_message": "Validation failed",
            "strict_mode": True,
            "auto_fix": False
        }
    },
    "Webhook": {
        "color": "#00BCD4",
        "icon": "🌐",
        "description": "External webhook integration",
        "properties": {
            "url": "",
            "method": "POST",
            "headers": {},
            "auth_type": ["none", "bearer", "basic", "api_key"],
            "timeout": 30
        }
    },
    "Output": {
        "color": "#8BC34A",
        "icon": "📤",
        "description": "Final output formatting",
        "properties": {
            "format": ["text", "json", "html", "markdown"],
            "template": "",
            "post_process": False
        }
    }
}

# Display component cards
for comp_name, comp_info in components_config.items():
    with st.sidebar.container():
        st.markdown(f"""
        <div class="component-card">
            <strong>{comp_info['icon']} {comp_name}</strong><br>
            <small>{comp_info['description']}</small>
        </div>
        """, unsafe_allow_html=True)

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Canvas controls
st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
st.sidebar.markdown("### ⚙️ Canvas Controls")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.canvas_data = {'nodes': [], 'connections': []}
        st.session_state.agent_flow = AgentFlow([], [], {})
        st.rerun()

with col2:
    if st.button("💾 Save", use_container_width=True):
        agent_name = st.text_input("Agent Name", "my_agent")
        if agent_name:
            st.session_state.saved_agents[agent_name] = st.session_state.agent_flow
            st.success("Agent saved!")

# Load saved agents
if st.session_state.saved_agents:
    selected_agent = st.sidebar.selectbox(
        "Load Saved Agent",
        [""] + list(st.session_state.saved_agents.keys())
    )
    if selected_agent and st.sidebar.button("📂 Load Agent"):
        st.session_state.agent_flow = st.session_state.saved_agents[selected_agent]
        st.rerun()

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Execution controls
if st.session_state.openai_client:
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.markdown("### 🚀 Agent Execution")
    
    # Agent stats
    stats = st.session_state.execution_stats
    st.sidebar.markdown(f"""
    <div class="agent-stats">
        <strong>Agent Statistics</strong><br>
        Runs: {stats['total_runs']}<br>
        Success: {stats['successful_runs']}<br>
        Failed: {stats['failed_runs']}<br>
        Avg Time: {stats['avg_response_time']:.2f}s
    </div>
    """, unsafe_allow_html=True)
    
    # Test input
    test_input = st.sidebar.text_area(
        "Test Input",
        "What is the capital of France?",
        height=80
    )
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("▶️ Run", use_container_width=True):
            if st.session_state.canvas_data['nodes']:
                st.session_state.agent_running = True
                execute_agent_flow(test_input)
            else:
                st.error("No agent flow to execute!")
    
    with col2:
        if st.button("⏹️ Stop", use_container_width=True):
            st.session_state.agent_running = False
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🤖 Advanced LangChain Agent Builder</h1>', unsafe_allow_html=True)

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["🎨 Visual Builder", "⚙️ Configuration", "🚀 Execution", "📊 Analytics"])

with tab1:
    st.markdown("### 🎨 Visual Canvas")
    
    # Enhanced HTML and JavaScript for Fabric.js canvas
    canvas_html = f"""
    <div style="display: flex; gap: 20px;">
        <div style="flex: 1;">
            <div id="canvas-container" style="border: 2px solid #ddd; border-radius: 15px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); position: relative;">
                <canvas id="fabric-canvas" width="1000" height="700"></canvas>
                <div id="minimap" style="position: absolute; top: 10px; right: 10px; width: 150px; height: 100px; border: 1px solid #ccc; background: rgba(255,255,255,0.8); border-radius: 5px;"></div>
            </div>
            <div style="margin-top: 10px; text-align: center;">
                <button id="add-input" onclick="addComponent('Input')" style="margin: 5px; padding: 8px 15px; background: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer;">📝 Input</button>
                <button id="add-llm" onclick="addComponent('LLM')" style="margin: 5px; padding: 8px 15px; background: #2196F3; color: white; border: none; border-radius: 5px; cursor: pointer;">🤖 LLM</button>
                <button id="add-tool" onclick="addComponent('Tool')" style="margin: 5px; padding: 8px 15px; background: #FF9800; color: white; border: none; border-radius: 5px; cursor: pointer;">🔧 Tool</button>
                <button id="add-memory" onclick="addComponent('Memory')" style="margin: 5px; padding: 8px 15px; background: #607D8B; color: white; border: none; border-radius: 5px; cursor: pointer;">🧠 Memory</button>
                <button id="add-router" onclick="addComponent('Router')" style="margin: 5px; padding: 8px 15px; background: #F44336; color: white; border: none; border-radius: 5px; cursor: pointer;">🔀 Router</button>
                <button id="add-output" onclick="addComponent('Output')" style="margin: 5px; padding: 8px 15px; background: #8BC34A; color: white; border: none; border-radius: 5px; cursor: pointer;">📤 Output</button>
            </div>
        </div>
        <div id="node-properties" style="width: 300px; background: white; border: 1px solid #ddd; border-radius: 10px; padding: 20px; max-height: 700px; overflow-y: auto;">
            <h3>Node Properties</h3>
            <div id="properties-content">Select a node to edit its properties</div>
        </div>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/fabric.js/5.3.0/fabric.min.js"></script>
    
    <script>
    // Enhanced Fabric.js canvas with professional features
    const canvas = new fabric.Canvas('fabric-canvas', {{
        backgroundColor: '#f8f9fa',
        selection: true,
        preserveObjectStacking: true
    }});
    
    // Component templates with enhanced styling
    const componentTemplates = {json.dumps(components_config)};
    
    // Enhanced state management
    let canvasState = {json.dumps(st.session_state.canvas_data)};
    let selectedNode = null;
    let connectionMode = false;
    let connectionStart = null;
    let nodeCounter = 1;
    let gridSnap = 20;
    let zoomLevel = 1;
    
    // Grid background
    function drawGrid() {{
        const gridSize = 30;
        const canvasWidth = canvas.width;
        const canvasHeight = canvas.height;
        
        // Vertical lines
        for (let i = 0; i <= canvasWidth; i += gridSize) {{
            const line = new fabric.Line([i, 0, i, canvasHeight], {{
                stroke: '#e0e0e0',
                strokeWidth: 0.5,
                selectable: false,
                evented: false,
                excludeFromExport: true
            }});
            canvas.add(line);
            canvas.sendToBack(line);
        }}
        
        // Horizontal lines
        for (let i = 0; i <= canvasHeight; i += gridSize) {{
            const line = new fabric.Line([0, i, canvasWidth, i], {{
                stroke: '#e0e0e0',
                strokeWidth: 0.5,
                selectable: false,
                evented: false,
                excludeFromExport: true
            }});
            canvas.add(line);
            canvas.sendToBack(line);
        }}
    }}
    
    // Enhanced node creation with animations and gradients
    function createNode(type, x = null, y = null) {{
        const nodeId = type.toLowerCase() + '_' + nodeCounter++;
        const template = componentTemplates[type];
        
        // Calculate position
        const posX = x || Math.random() * 800 + 100;
        const posY = y || Math.random() * 500 + 100;
        
        // Snap to grid
        const snapX = Math.round(posX / gridSnap) * gridSnap;
        const snapY = Math.round(posY / gridSnap) * gridSnap;
        
        // Create gradient background
        const gradient = new fabric.Gradient({{
            type: 'linear',
            coords: {{ x1: 0, y1: 0, x2: 0, y2: 1 }},
            colorStops: [
                {{ offset: 0, color: template.color }},
                {{ offset: 1, color: fabric.Color.fromHex(template.color).darker().toHex() }}
            ]
        }});
        
        // Main rectangle with rounded corners
        const rect = new fabric.Rect({{
            width: 140,
            height: 90,
            fill: gradient,
            stroke: '#333',
            strokeWidth: 2,
            rx: 15,
            ry: 15,
            shadow: new fabric.Shadow({{
                color: 'rgba(0,0,0,0.3)',
                blur: 10,
                offsetX: 2,
                offsetY: 2
            }})
        }});
        
        // Icon
        const icon = new fabric.Text(template.icon, {{
            fontSize: 24,
            fontFamily: 'Arial',
            fill: 'white',
            textAlign: 'center',
            top: -35,
            left: 0,
            originX: 'center',
            originY: 'center'
        }});
        
        // Title text
        const title = new fabric.Text(type, {{
            fontSize: 14,
            fontWeight: 'bold',
            fontFamily: 'Arial',
            fill: 'white',
            textAlign: 'center',
            top: -5,
            left: 0,
            originX: 'center',
            originY: 'center'
        }});
        
        // Status indicator
        const statusIndicator = new fabric.Circle({{
            radius: 6,
            fill: '#4CAF50',
            top: -40,
            left: 60,
            originX: 'center',
            originY: 'center'
        }});
        
        // Input/Output ports
        const inputPort = new fabric.Circle({{
            radius: 8,
            fill: '#fff',
            stroke: '#333',
            strokeWidth: 2,
            top: 0,
            left: -70,
            originX: 'center',
            originY: 'center',
            portType: 'input'
        }});
        
        const outputPort = new fabric.Circle({{
            radius: 8,
            fill: '#fff',
            stroke: '#333',
            strokeWidth: 2,
            top: 0,
            left: 70,
            originX: 'center',
            originY: 'center',
            portType: 'output'
        }});
        
        // Create group
        const group = new fabric.Group([rect, icon, title, statusIndicator, inputPort, outputPort], {{
            left: snapX,
            top: snapY,
            selectable: true,
            hasControls: true,
            hasBorders: true,
            lockScalingX: true,
            lockScalingY: true,
            nodeId: nodeId,
            nodeType: type,
            borderColor: template.color,
            cornerColor: template.color,
            cornerStyle: 'circle'
        }});
        
        // Enhanced event listeners
        group.on('mousedown', function(e) {{
            if (connectionMode) {{
                handleConnectionMode(group);
            }} else {{
                selectNode(group);
            }}
        }});
        
        group.on('moving', function(e) {{
            // Snap to grid while moving
            const snapX = Math.round(group.left / gridSnap) * gridSnap;
            const snapY = Math.round(group.top / gridSnap) * gridSnap;
            group.set({{ left: snapX, top: snapY }});
            updateConnections();
        }});
        
        group.on('mouseenter', function(e) {{
            group.set({{
                shadow: new fabric.Shadow({{
                    color: 'rgba(0,0,0,0.5)',
                    blur: 15,
                    offsetX: 3,
                    offsetY: 3
                }})
            }});
            canvas.renderAll();
        }});
        
        group.on('mouseleave', function(e) {{
            if (group !== selectedNode) {{
                group.set({{
                    shadow: new fabric.Shadow({{
                        color: 'rgba(0,0,0,0.3)',
                        blur: 10,
                        offsetX: 2,
                        offsetY: 2
                    }})
                }});
                canvas.renderAll();
            }}
        }});
        
        // Add to canvas with animation
        canvas.add(group);
        
        // Entrance animation
        group.set({{ opacity: 0, scaleX: 0.1, scaleY: 0.1 }});
        group.animate({{ opacity: 1, scaleX: 1, scaleY: 1 }}, {{
            duration: 300,
            easing: fabric.util.ease.easeOutBack
        }});
        
        // Add to state
        canvasState.nodes.push({{
            id: nodeId,
            type: type,
            x: snapX,
            y: snapY,
            name: type + '_' + nodeId.split('_')[1],
            properties: template.properties
        }});
        
        updateStreamlitState();
        return group;
    }}
    
    // Connection handling
    function handleConnectionMode(group) {{
        if (!connectionStart) {{
            connectionStart = group;
            group.set({{ strokeWidth: 4, stroke: '#ff0000' }});
            canvas.renderAll();
        }} else if (connectionStart !== group) {{
            createConnection(connectionStart, group);
            connectionStart.set({{ strokeWidth: 2, stroke: '#333' }});
            connectionStart = null;
            connectionMode = false;
            canvas.defaultCursor = 'default';
        }}
    }}
    
    // Enhanced connection creation with bezier curves
    function createConnection(startNode, endNode) {{
        const startPos = startNode.getCenterPoint();
        const endPos = endNode.getCenterPoint();
        
        // Calculate control points for bezier curve
        const dx = endPos.x - startPos.x;
        const controlOffset = Math.abs(dx) * 0.5;
        
        const pathString = `M ${{startPos.x}} ${{startPos.y}} C ${{startPos.x + controlOffset}} ${{startPos.y}} ${{endPos.x - controlOffset}} ${{endPos.y}} ${{endPos.x}} ${{endPos.y}}`;
        
        const path = new fabric.Path(pathString, {{
            fill: '',
            stroke: '#333',
            strokeWidth: 3,
            selectable: false,
            evented: false,
            shadow: new fabric.Shadow({{
                color: 'rgba(0,0,0,0.2)',
                blur: 5,
                offsetX: 1,
                offsetY: 1
            }}),
            connectionId: 'conn_' + Date.now(),
            fromNode: startNode.nodeId,
            toNode: endNode.nodeId
        }});
        
        // Add animated arrow
        const angle = Math.atan2(endPos.y - startPos.y, endPos.x - startPos.x);
        const arrowHead = new fabric.Triangle({{
            left: endPos.x - 15,
            top: endPos.y - 5,
            width: 15,
            height: 10,
            fill: '#333',
            selectable: false,
            evented: false,
            angle: (angle * 180 / Math.PI) + 90,
            shadow: new fabric.Shadow({{
                color: 'rgba(0,0,0,0.2)',
                blur: 3,
                offsetX: 1,
                offsetY: 1
            }})
        }});
        
        canvas.add(path);
        canvas.add(arrowHead);
        canvas.sendToBack(path);
        canvas.sendToBack(arrowHead);
        
        // Add to state
        canvasState.connections.push({{
            id: 'conn_' + Date.now(),
            from: startNode.nodeId,
            to: endNode.nodeId,
            fromPort: 'output',
            toPort: 'input'
        }});
        
        updateStreamlitState();
    }}
    
    // Update connections when nodes move
    function updateConnections() {{
        // This would update connection paths when nodes are moved
        // Implementation would involve recalculating bezier curves
    }}
    
    // Node selection with property panel update
    function selectNode(group) {{
        if (selectedNode) {{
            selectedNode.set({{ strokeWidth: 2 }});
        }}
        
        selectedNode = group;
        group.set({{ strokeWidth: 4 }});
        canvas.renderAll();
        
        // Update properties panel
        updatePropertiesPanel(group);
        
        // Notify Streamlit
        window.parent.postMessage({{
            type: 'node_selected',
            nodeId: group.nodeId,
            nodeType: group.nodeType
        }}, '*');
    }}
    
    // Properties panel update
    function updatePropertiesPanel(group) {{
        const nodeData = canvasState.nodes.find(n => n.id === group.nodeId);
        if (!nodeData) return;
        
        const template = componentTemplates[group.nodeType];
        const propertiesHtml = `
            <h3>${{template.icon}} ${{group.nodeType}}</h3>
            <div style="margin-bottom: 10px;">
                <label><strong>Name:</strong></label>
                <input type="text" value="${{nodeData.name}}" onchange="updateNodeProperty('name', this.value)" style="width: 100%; padding: 5px; margin-top: 5px; border: 1px solid #ddd; border-radius: 3px;">
            </div>
            ${{generatePropertyInputs(template.properties, nodeData.properties)}}
            <button onclick="deleteNode()" style="width: 100%; padding: 8px; background: #f44336; color: white; border: none; border-radius: 5px; margin-top: 10px; cursor: pointer;">🗑️ Delete Node</button>
        `;
        
        document.getElementById('properties-content').innerHTML = propertiesHtml;
    }}
    
    // Generate property inputs dynamically
    function generatePropertyInputs(templateProps, currentProps) {{
        let html = '';
        for (const [key, defaultValue] of Object.entries(templateProps)) {{
            const currentValue = currentProps[key] || defaultValue;
