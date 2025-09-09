import streamlit as st
import json
from typing import Dict, List, Any
import uuid

# Configure Streamlit page
st.set_page_config(
    page_title="LangChain Visual Agent Builder",
    page_icon="🔗",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #2E86C1;
    margin-bottom: 30px;
}
.sidebar-section {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.json-output {
    background-color: #f1f3f4;
    padding: 15px;
    border-radius: 5px;
    border-left: 4px solid #4CAF50;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'canvas_data' not in st.session_state:
    st.session_state.canvas_data = {
        'nodes': [],
        'connections': []
    }

if 'selected_node' not in st.session_state:
    st.session_state.selected_node = None

# Main header
st.markdown('<h1 class="main-header">🔗 LangChain Visual Agent Builder</h1>', unsafe_allow_html=True)

# Sidebar for components and controls
st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
st.sidebar.markdown("### 🧩 Available Components")

# Component library
components = {
    "Input": {
        "color": "#4CAF50",
        "icon": "📝",
        "description": "User input node"
    },
    "LLM": {
        "color": "#2196F3", 
        "icon": "🤖",
        "description": "Language model"
    },
    "Tool": {
        "color": "#FF9800",
        "icon": "🔧",
        "description": "External tool"
    },
    "Memory": {
        "color": "#9C27B0",
        "icon": "🧠",
        "description": "Conversation memory"
    },
    "Router": {
        "color": "#F44336",
        "icon": "🔀",
        "description": "Decision router"
    },
    "Output": {
        "color": "#795548",
        "icon": "📤",
        "description": "Final output"
    }
}

for comp_name, comp_info in components.items():
    st.sidebar.markdown(f"""
    **{comp_info['icon']} {comp_name}**  
    {comp_info['description']}
    """)

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Canvas controls
st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
st.sidebar.markdown("### ⚙️ Canvas Controls")

if st.sidebar.button("🗑️ Clear Canvas"):
    st.session_state.canvas_data = {'nodes': [], 'connections': []}
    st.rerun()

if st.sidebar.button("📋 Export JSON"):
    st.session_state.show_json = True

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Node properties editor
if st.session_state.selected_node:
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.markdown("### 🎛️ Node Properties")
    
    # Find the selected node
    selected_node = None
    for node in st.session_state.canvas_data['nodes']:
        if node['id'] == st.session_state.selected_node:
            selected_node = node
            break
    
    if selected_node:
        st.sidebar.markdown(f"**Editing:** {selected_node['type']}")
        
        # Common properties
        selected_node['name'] = st.sidebar.text_input("Node Name", selected_node.get('name', ''))
        
        # Type-specific properties
        if selected_node['type'] == 'LLM':
            selected_node['model'] = st.sidebar.selectbox(
                "Model", 
                ["gpt-3.5-turbo", "gpt-4", "claude-3-sonnet", "llama-2"],
                index=0
            )
            selected_node['temperature'] = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
            
        elif selected_node['type'] == 'Tool':
            selected_node['tool_type'] = st.sidebar.selectbox(
                "Tool Type",
                ["web_search", "calculator", "file_reader", "api_call"],
                index=0
            )
            selected_node['parameters'] = st.sidebar.text_area("Parameters (JSON)", "{}")
            
        elif selected_node['type'] == 'Memory':
            selected_node['memory_type'] = st.sidebar.selectbox(
                "Memory Type",
                ["conversation_buffer", "conversation_summary", "vector_store"],
                index=0
            )
            
        elif selected_node['type'] == 'Router':
            selected_node['routing_logic'] = st.sidebar.text_area(
                "Routing Logic", 
                "Define routing conditions..."
            )
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Main canvas area
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 🎨 Canvas")
    
    # HTML and JavaScript for Fabric.js canvas
    canvas_html = f"""
    <div id="canvas-container" style="border: 2px solid #ddd; border-radius: 10px; background: #fff;">
        <canvas id="fabric-canvas" width="800" height="600"></canvas>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/fabric.js/5.3.0/fabric.min.js"></script>
    
    <script>
    // Initialize Fabric.js canvas
    const canvas = new fabric.Canvas('fabric-canvas', {{
        backgroundColor: '#f8f9fa',
        selection: false
    }});
    
    // Component templates
    const componentTemplates = {json.dumps(components)};
    
    // Current canvas state
    let canvasState = {json.dumps(st.session_state.canvas_data)};
    let selectedNodeId = null;
    let connectionMode = false;
    let connectionStart = null;
    
    // Create node function
    function createNode(type, x, y) {{
        const nodeId = 'node_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        const template = componentTemplates[type];
        
        // Create node group
        const rect = new fabric.Rect({{
            width: 120,
            height: 80,
            fill: template.color,
            stroke: '#333',
            strokeWidth: 2,
            rx: 10,
            ry: 10
        }});
        
        const text = new fabric.Text(template.icon + ' ' + type, {{
            fontSize: 14,
            fontFamily: 'Arial',
            fill: 'white',
            textAlign: 'center'
        }});
        
        const group = new fabric.Group([rect, text], {{
            left: x - 60,
            top: y - 40,
            selectable: true,
            hasControls: false,
            hasBorders: true,
            nodeId: nodeId,
            nodeType: type
        }});
        
        // Add event listeners
        group.on('mousedown', function(e) {{
            if (connectionMode) {{
                if (!connectionStart) {{
                    connectionStart = group;
                    group.set('stroke', '#ff0000');
                    canvas.renderAll();
                }} else if (connectionStart !== group) {{
                    // Create connection
                    createConnection(connectionStart, group);
                    connectionStart.set('stroke', template.color);
                    connectionStart = null;
                    connectionMode = false;
                    canvas.renderAll();
                }}
            }} else {{
                selectedNodeId = nodeId;
                // Send selection to Streamlit
                window.parent.postMessage({{
                    type: 'node_selected',
                    nodeId: nodeId
                }}, '*');
            }}
        }});
        
        canvas.add(group);
        
        // Add to canvas state
        canvasState.nodes.push({{
            id: nodeId,
            type: type,
            x: x,
            y: y,
            name: type + '_' + nodeId.slice(-4)
        }});
        
        updateStreamlitState();
        return group;
    }}
    
    // Create connection function
    function createConnection(startNode, endNode) {{
        const startPos = startNode.getCenterPoint();
        const endPos = endNode.getCenterPoint();
        
        const line = new fabric.Line([
            startPos.x, startPos.y,
            endPos.x, endPos.y
        ], {{
            stroke: '#333',
            strokeWidth: 3,
            selectable: false,
            evented: false,
            strokeDashArray: [5, 5]
        }});
        
        // Add arrow head
        const angle = Math.atan2(endPos.y - startPos.y, endPos.x - startPos.x);
        const headLen = 15;
        
        const arrowHead = new fabric.Triangle({{
            left: endPos.x - headLen * Math.cos(angle - Math.PI/6),
            top: endPos.y - headLen * Math.sin(angle - Math.PI/6),
            width: 10,
            height: 10,
            fill: '#333',
            selectable: false,
            evented: false,
            angle: (angle * 180 / Math.PI) + 90
        }});
        
        canvas.add(line);
        canvas.add(arrowHead);
        
        // Add to canvas state
        canvasState.connections.push({{
            from: startNode.nodeId,
            to: endNode.nodeId,
            id: 'conn_' + Date.now()
        }});
        
        updateStreamlitState();
    }}
    
    // Update Streamlit state
    function updateStreamlitState() {{
        window.parent.postMessage({{
            type: 'canvas_updated',
            data: canvasState
        }}, '*');
    }}
    
    // Handle double-click to add components
    canvas.on('mouse:dblclick', function(e) {{
        if (!connectionMode) {{
            const pointer = canvas.getPointer(e.e);
            // Show component selector (simplified - in real app you'd have a modal)
            const type = prompt('Enter component type (Input, LLM, Tool, Memory, Router, Output):');
            if (type && componentTemplates[type]) {{
                createNode(type, pointer.x, pointer.y);
            }}
        }}
    }});
    
    // Toggle connection mode with 'c' key
    document.addEventListener('keydown', function(e) {{
        if (e.key === 'c' || e.key === 'C') {{
            connectionMode = !connectionMode;
            canvas.defaultCursor = connectionMode ? 'crosshair' : 'default';
            console.log('Connection mode:', connectionMode);
        }}
    }});
    
    // Render existing nodes and connections
    function renderExistingState() {{
        // Clear canvas first
        canvas.clear();
        canvas.backgroundColor = '#f8f9fa';
        
        // Render nodes
        canvasState.nodes.forEach(node => {{
            createNode(node.type, node.x, node.y);
        }});
        
        // Note: Connections would need more complex state management
        // for proper persistence across refreshes
    }}
    
    // Listen for messages from Streamlit
    window.addEventListener('message', function(e) {{
        if (e.data.type === 'clear_canvas') {{
            canvas.clear();
            canvasState = {{nodes: [], connections: []}};
            canvas.backgroundColor = '#f8f9fa';
        }}
    }});
    
    // Instructions overlay
    const instructions = new fabric.Text(
        'Double-click to add components\\nPress "C" to toggle connection mode\\nDrag to move components',
        {{
            left: 10,
            top: 10,
            fontSize: 12,
            fill: '#666',
            selectable: false,
            evented: false
        }}
    );
    canvas.add(instructions);
    
    renderExistingState();
    </script>
    """
    
    # Display the canvas
    st.components.v1.html(canvas_html, height=650)

with col2:
    st.markdown("### 📊 Agent Flow")
    
    # Display current nodes
    if st.session_state.canvas_data['nodes']:
        st.markdown("**Nodes:**")
        for node in st.session_state.canvas_data['nodes']:
            node_emoji = components.get(node['type'], {}).get('icon', '📦')
            st.write(f"{node_emoji} {node.get('name', node['type'])}")
    
    # Display connections
    if st.session_state.canvas_data['connections']:
        st.markdown("**Connections:**")
        for conn in st.session_state.canvas_data['connections']:
            st.write(f"🔗 {conn['from']} → {conn['to']}")
    
    if not st.session_state.canvas_data['nodes']:
        st.info("Double-click on canvas to add components")

# JSON Export section
if st.session_state.get('show_json', False):
    st.markdown("### 📄 Generated JSON Configuration")
    
    # Generate LangChain-like configuration
    agent_config = {
        "agent_type": "custom_visual_agent",
        "created_at": "2024-01-01T00:00:00Z",
        "nodes": {},
        "flow": []
    }
    
    # Process nodes
    for node in st.session_state.canvas_data['nodes']:
        agent_config["nodes"][node['id']] = {
            "type": node['type'].lower(),
            "name": node.get('name', node['type']),
            "position": {"x": node['x'], "y": node['y']},
            "config": {}
        }
        
        # Add type-specific config
        if 'model' in node:
            agent_config["nodes"][node['id']]["config"]["model"] = node['model']
        if 'temperature' in node:
            agent_config["nodes"][node['id']]["config"]["temperature"] = node['temperature']
        if 'tool_type' in node:
            agent_config["nodes"][node['id']]["config"]["tool_type"] = node['tool_type']
    
    # Process connections
    for conn in st.session_state.canvas_data['connections']:
        agent_config["flow"].append({
            "from": conn['from'],
            "to": conn['to'],
            "condition": "default"
        })
    
    st.markdown('<div class="json-output">', unsafe_allow_html=True)
    st.json(agent_config)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Download button for JSON
    json_str = json.dumps(agent_config, indent=2)
    st.download_button(
        label="💾 Download Agent Configuration",
        data=json_str,
        file_name="langchain_agent_config.json",
        mime="application/json"
    )
    
    if st.button("❌ Hide JSON"):
        st.session_state.show_json = False
        st.rerun()

# Instructions
st.markdown("""
---
### 📚 How to Use:

1. **Add Components**: Double-click on the canvas and enter a component type (Input, LLM, Tool, Memory, Router, Output)
2. **Connect Components**: Press 'C' to toggle connection mode, then click two nodes to connect them
3. **Edit Properties**: Click a node to select it and edit its properties in the sidebar
4. **Export Configuration**: Click "Export JSON" to see the generated LangChain configuration
5. **Clear Canvas**: Use the "Clear Canvas" button to start over

**Available Components:**
- **Input**: Starting point for user queries
- **LLM**: Language model processing
- **Tool**: External tools and APIs
- **Memory**: Conversation history and context
- **Router**: Decision-making and flow control
- **Output**: Final response formatting
""")

# Handle JavaScript messages (this would need to be implemented with proper Streamlit-JS communication)
# For now, this is a placeholder for the concept
st.markdown("""
<script>
// This would handle communication between the Fabric.js canvas and Streamlit
window.addEventListener('message', function(event) {
    if (event.data.type === 'node_selected') {
        // Update Streamlit session state with selected node
        console.log('Node selected:', event.data.nodeId);
    }
    if (event.data.type === 'canvas_updated') {
        // Update Streamlit session state with canvas data
        console.log('Canvas updated:', event.data.data);
    }
});
</script>
""", unsafe_allow_html=True)
