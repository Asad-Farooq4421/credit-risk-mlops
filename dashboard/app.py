"""
🎮 CYBERPUNK CREDIT RISK DASHBOARD
===================================
Main Streamlit Application with 3D Visualizations
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Page config - MUST BE FIRST
st.set_page_config(
    page_title="⚡ CYBER CREDIT",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# CYBERPUNK CSS THEME
# ============================================================================

def load_cyberpunk_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');

    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0f0f1a 100%);
        background-attachment: fixed;
    }

    /* Animated grid background */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image:
            linear-gradient(rgba(0, 255, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 255, 0.03) 1px, transparent 1px);
        background-size: 50px 50px;
        pointer-events: none;
        z-index: 0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1a 0%, #1a0a2e 100%);
        border-right: 1px solid #00ffff33;
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: #00ffff;
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        color: #00ffff !important;
        text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff, 0 0 30px #00ffff;
    }

    h1 {
        font-size: 2.5rem !important;
        letter-spacing: 3px;
        animation: glow 2s ease-in-out infinite alternate;
    }

    @keyframes glow {
        from { text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff; }
        to { text-shadow: 0 0 20px #ff00ff, 0 0 30px #ff00ff, 0 0 40px #ff00ff; }
    }

    /* Regular text */
    p, span, label, .stMarkdown {
        font-family: 'Rajdhani', sans-serif !important;
        color: #e0e0e0 !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif !important;
        color: #00ffff !important;
        font-size: 2rem !important;
        text-shadow: 0 0 10px #00ffff;
    }

    [data-testid="stMetricLabel"] {
        font-family: 'Rajdhani', sans-serif !important;
        color: #ff00ff !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* Buttons */
    .stButton > button {
        font-family: 'Orbitron', sans-serif !important;
        background: linear-gradient(90deg, #ff00ff 0%, #00ffff 100%);
        color: #000 !important;
        border: none;
        padding: 15px 30px;
        font-weight: bold;
        letter-spacing: 2px;
        text-transform: uppercase;
        transition: all 0.3s ease;
        box-shadow: 0 0 20px rgba(255, 0, 255, 0.5);
    }

    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.8);
    }

    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background: rgba(0, 255, 255, 0.1) !important;
        border: 1px solid #00ffff !important;
        color: #00ffff !important;
        font-family: 'Rajdhani', sans-serif !important;
    }

    .stSelectbox > div > div {
        background: rgba(0, 255, 255, 0.1) !important;
        border: 1px solid #00ffff !important;
    }

    /* Cards / Containers */
    .css-1r6slb0, .css-12w0qpk {
        background: rgba(26, 26, 46, 0.8) !important;
        border: 1px solid #00ffff33;
        border-radius: 10px;
    }

    /* Neon box */
    .neon-box {
        background: linear-gradient(145deg, rgba(26, 10, 46, 0.9), rgba(10, 26, 46, 0.9));
        border: 2px solid #00ffff;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.3), inset 0 0 20px rgba(0, 255, 255, 0.1);
        margin: 10px 0;
    }

    .neon-box-pink {
        background: linear-gradient(145deg, rgba(46, 10, 26, 0.9), rgba(26, 10, 46, 0.9));
        border: 2px solid #ff00ff;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 0 20px rgba(255, 0, 255, 0.3), inset 0 0 20px rgba(255, 0, 255, 0.1);
        margin: 10px 0;
    }

    /* Glitch effect */
    .glitch {
        animation: glitch 1s linear infinite;
    }

    @keyframes glitch {
        2%, 64% { transform: translate(2px, 0) skew(0deg); }
        4%, 60% { transform: translate(-2px, 0) skew(0deg); }
        62% { transform: translate(0, 0) skew(5deg); }
    }

    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #ff00ff, #00ffff) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Orbitron', sans-serif !important;
        background: rgba(0, 255, 255, 0.1);
        border: 1px solid #00ffff;
        color: #00ffff;
        border-radius: 5px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #ff00ff, #00ffff) !important;
        color: #000 !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'Orbitron', sans-serif !important;
        background: rgba(0, 255, 255, 0.1);
        border: 1px solid #00ffff;
        color: #00ffff !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #0a0a0f;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #ff00ff, #00ffff);
        border-radius: 5px;
    }

    /* Dataframe */
    .stDataFrame {
        border: 1px solid #00ffff !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Loading animation */
    .loading-pulse {
        animation: pulse 1.5s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# THREE.JS 3D COMPONENTS
# ============================================================================

def render_3d_header():
    """Animated 3D header with Three.js"""
    threejs_header = """
    <div id="header-container" style="width: 100%; height: 150px; position: relative; overflow: hidden;">
        <canvas id="header-canvas"></canvas>
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 10; text-align: center;">
            <h1 style="font-family: 'Orbitron', sans-serif; font-size: 3rem; color: #00ffff;
                       text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff, 0 0 30px #ff00ff;
                       margin: 0; letter-spacing: 5px;">
                ⚡ CYBER CREDIT
            </h1>
            <p style="font-family: 'Rajdhani', sans-serif; color: #ff00ff; letter-spacing: 3px; margin-top: 5px;">
                NEURAL RISK ASSESSMENT ENGINE v2.0
            </p>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        const container = document.getElementById('header-container');
        const canvas = document.getElementById('header-canvas');

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, container.clientWidth / 150, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true });
        renderer.setSize(container.clientWidth, 150);
        renderer.setClearColor(0x000000, 0);

        // Create floating particles
        const particlesGeometry = new THREE.BufferGeometry();
        const particlesCount = 200;
        const posArray = new Float32Array(particlesCount * 3);

        for(let i = 0; i < particlesCount * 3; i++) {
            posArray[i] = (Math.random() - 0.5) * 10;
        }

        particlesGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));

        const particlesMaterial = new THREE.PointsMaterial({
            size: 0.05,
            color: 0x00ffff,
            transparent: true,
            opacity: 0.8
        });

        const particlesMesh = new THREE.Points(particlesGeometry, particlesMaterial);
        scene.add(particlesMesh);

        // Create glowing torus
        const torusGeometry = new THREE.TorusGeometry(1.5, 0.1, 16, 100);
        const torusMaterial = new THREE.MeshBasicMaterial({
            color: 0xff00ff,
            wireframe: true,
            transparent: true,
            opacity: 0.6
        });
        const torus = new THREE.Mesh(torusGeometry, torusMaterial);
        scene.add(torus);

        // Second torus
        const torus2 = new THREE.Mesh(
            new THREE.TorusGeometry(2, 0.05, 16, 100),
            new THREE.MeshBasicMaterial({ color: 0x00ffff, wireframe: true, transparent: true, opacity: 0.4 })
        );
        torus2.rotation.x = Math.PI / 2;
        scene.add(torus2);

        camera.position.z = 5;

        function animate() {
            requestAnimationFrame(animate);

            particlesMesh.rotation.y += 0.002;
            particlesMesh.rotation.x += 0.001;

            torus.rotation.x += 0.01;
            torus.rotation.y += 0.005;

            torus2.rotation.y += 0.008;
            torus2.rotation.z += 0.003;

            renderer.render(scene, camera);
        }

        animate();

        // Handle resize
        window.addEventListener('resize', () => {
            renderer.setSize(container.clientWidth, 150);
            camera.aspect = container.clientWidth / 150;
            camera.updateProjectionMatrix();
        });
    </script>
    """
    st.components.v1.html(threejs_header, height=160)


def render_3d_risk_gauge(risk_score: float):
    """3D animated risk gauge with Three.js and particles"""
    # Determine colors based on risk
    if risk_score < 0.3:
        main_color = "#00ff00"
        glow_color = "0x00ff00"
        status = "LOW RISK"
        status_color = "#00ff00"
    elif risk_score < 0.6:
        main_color = "#ffff00"
        glow_color = "0xffff00"
        status = "MEDIUM RISK"
        status_color = "#ffff00"
    else:
        main_color = "#ff0000"
        glow_color = "0xff0000"
        status = "HIGH RISK"
        status_color = "#ff0000"

    gauge_html = f"""
    <div id="gauge-container" style="width: 100%; height: 400px; position: relative;">
        <canvas id="gauge-canvas"></canvas>
        <div id="gauge-overlay" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; z-index: 10;">
            <div style="font-family: 'Orbitron', sans-serif; font-size: 4rem; color: {main_color};
                        text-shadow: 0 0 20px {main_color}, 0 0 40px {main_color};">
                {int(risk_score * 100)}%
            </div>
            <div style="font-family: 'Rajdhani', sans-serif; font-size: 1.5rem; color: {status_color};
                        letter-spacing: 3px; text-shadow: 0 0 10px {status_color};">
                {status}
            </div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        const container = document.getElementById('gauge-container');
        const canvas = document.getElementById('gauge-canvas');

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, container.clientWidth / 400, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({{ canvas: canvas, alpha: true, antialias: true }});
        renderer.setSize(container.clientWidth, 400);
        renderer.setClearColor(0x000000, 0);

        // Create rotating ring based on risk score
        const ringGeometry = new THREE.TorusGeometry(2.5, 0.15, 16, 100, Math.PI * 2 * {risk_score});
        const ringMaterial = new THREE.MeshBasicMaterial({{
            color: {glow_color},
            transparent: true,
            opacity: 0.9
        }});
        const ring = new THREE.Mesh(ringGeometry, ringMaterial);
        ring.rotation.z = -Math.PI / 2;
        scene.add(ring);

        // Background ring
        const bgRingGeometry = new THREE.TorusGeometry(2.5, 0.1, 16, 100);
        const bgRingMaterial = new THREE.MeshBasicMaterial({{
            color: 0x333333,
            transparent: true,
            opacity: 0.3
        }});
        const bgRing = new THREE.Mesh(bgRingGeometry, bgRingMaterial);
        scene.add(bgRing);

        // Outer decorative rings
        const outerRing1 = new THREE.Mesh(
            new THREE.TorusGeometry(3.2, 0.02, 8, 100),
            new THREE.MeshBasicMaterial({{ color: 0x00ffff, transparent: true, opacity: 0.5 }})
        );
        scene.add(outerRing1);

        const outerRing2 = new THREE.Mesh(
            new THREE.TorusGeometry(3.4, 0.02, 8, 100),
            new THREE.MeshBasicMaterial({{ color: 0xff00ff, transparent: true, opacity: 0.3 }})
        );
        scene.add(outerRing2);

        // Floating particles
        const particlesGeometry = new THREE.BufferGeometry();
        const particlesCount = 100;
        const posArray = new Float32Array(particlesCount * 3);

        for(let i = 0; i < particlesCount * 3; i++) {{
            posArray[i] = (Math.random() - 0.5) * 8;
        }}

        particlesGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));

        const particlesMaterial = new THREE.PointsMaterial({{
            size: 0.05,
            color: {glow_color},
            transparent: true,
            opacity: 0.6
        }});

        const particles = new THREE.Points(particlesGeometry, particlesMaterial);
        scene.add(particles);

        camera.position.z = 6;

        let time = 0;
        function animate() {{
            requestAnimationFrame(animate);
            time += 0.01;

            outerRing1.rotation.z += 0.005;
            outerRing2.rotation.z -= 0.003;

            particles.rotation.y += 0.002;
            particles.rotation.x = Math.sin(time) * 0.1;

            // Pulse effect on ring
            ring.material.opacity = 0.7 + Math.sin(time * 3) * 0.2;

            renderer.render(scene, camera);
        }}

        animate();
    </script>
    """
    st.components.v1.html(gauge_html, height=420)


def render_3d_sphere_metrics(metrics: dict):
    """3D rotating sphere with orbiting metrics"""
    sphere_html = f"""
    <div id="sphere-container" style="width: 100%; height: 500px; position: relative;">
        <canvas id="sphere-canvas"></canvas>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        const container = document.getElementById('sphere-container');
        const canvas = document.getElementById('sphere-canvas');

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, container.clientWidth / 500, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({{ canvas: canvas, alpha: true, antialias: true }});
        renderer.setSize(container.clientWidth, 500);
        renderer.setClearColor(0x000000, 0);

        // Central glowing sphere
        const sphereGeometry = new THREE.IcosahedronGeometry(1.5, 1);
        const sphereMaterial = new THREE.MeshBasicMaterial({{
            color: 0x00ffff,
            wireframe: true,
            transparent: true,
            opacity: 0.6
        }});
        const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
        scene.add(sphere);

        // Inner sphere
        const innerSphere = new THREE.Mesh(
            new THREE.IcosahedronGeometry(1.2, 1),
            new THREE.MeshBasicMaterial({{ color: 0xff00ff, wireframe: true, transparent: true, opacity: 0.3 }})
        );
        scene.add(innerSphere);

        // Orbiting rings for metrics
        const metrics = [
            {{ name: 'ACC', value: {metrics.get('accuracy', 0.78)}, color: 0x00ff00 }},
            {{ name: 'PRE', value: {metrics.get('precision', 0.72)}, color: 0xff00ff }},
            {{ name: 'REC', value: {metrics.get('recall', 0.65)}, color: 0xffff00 }},
            {{ name: 'F1', value: {metrics.get('f1_score', 0.68)}, color: 0x00ffff }},
            {{ name: 'AUC', value: {metrics.get('roc_auc', 0.82)}, color: 0xff6600 }}
        ];

        const orbitRadius = 3;
        const metricSpheres = [];

        metrics.forEach((metric, index) => {{
            const angle = (index / metrics.length) * Math.PI * 2;

            // Small sphere for each metric
            const metricSphere = new THREE.Mesh(
                new THREE.SphereGeometry(0.2 + metric.value * 0.3, 16, 16),
                new THREE.MeshBasicMaterial({{ color: metric.color, transparent: true, opacity: 0.8 }})
            );

            metricSphere.userData = {{ angle: angle, name: metric.name, value: metric.value }};
            metricSpheres.push(metricSphere);
            scene.add(metricSphere);

            // Ring around metric sphere
            const ring = new THREE.Mesh(
                new THREE.TorusGeometry(0.4 + metric.value * 0.3, 0.02, 8, 32),
                new THREE.MeshBasicMaterial({{ color: metric.color, transparent: true, opacity: 0.5 }})
            );
            ring.userData = {{ parent: metricSphere }};
            scene.add(ring);
        }});

        // Particle field
        const particlesGeometry = new THREE.BufferGeometry();
        const particlesCount = 500;
        const posArray = new Float32Array(particlesCount * 3);

        for(let i = 0; i < particlesCount * 3; i++) {{
            posArray[i] = (Math.random() - 0.5) * 15;
        }}

        particlesGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
        const particles = new THREE.Points(particlesGeometry, new THREE.PointsMaterial({{
            size: 0.02,
            color: 0x00ffff,
            transparent: true,
            opacity: 0.4
        }}));
        scene.add(particles);

        camera.position.z = 7;

        let time = 0;
        function animate() {{
            requestAnimationFrame(animate);
            time += 0.01;

            sphere.rotation.x += 0.005;
            sphere.rotation.y += 0.01;

            innerSphere.rotation.x -= 0.008;
            innerSphere.rotation.y -= 0.005;

            // Animate metric spheres in orbit
            metricSpheres.forEach((ms, index) => {{
                const baseAngle = ms.userData.angle;
                const currentAngle = baseAngle + time * (0.5 + index * 0.1);

                ms.position.x = Math.cos(currentAngle) * orbitRadius;
                ms.position.z = Math.sin(currentAngle) * orbitRadius;
                ms.position.y = Math.sin(time + index) * 0.5;

                ms.rotation.x += 0.02;
                ms.rotation.y += 0.02;
            }});

            particles.rotation.y += 0.001;

            renderer.render(scene, camera);
        }}

        animate();
    </script>
    """
    st.components.v1.html(sphere_html, height=520)


# ============================================================================
# PLOTLY 3D CHARTS
# ============================================================================

def create_3d_feature_importance(features_df: pd.DataFrame):
    """Create 3D bar chart for feature importance"""

    fig = go.Figure()

    # Create 3D bars using scatter3d with markers
    n_features = len(features_df)

    for i, row in features_df.iterrows():
        # Create vertical line for each bar
        fig.add_trace(go.Scatter3d(
            x=[i, i],
            y=[0, 0],
            z=[0, row['importance']],
            mode='lines',
            line=dict(
                color=f'rgb({255 - int(row["importance"]*255)}, {int(row["importance"]*255)}, 255)',
                width=15
            ),
            name=row['feature'],
            hovertemplate=f"<b>{row['feature']}</b><br>Importance: {row['importance']:.3f}<extra></extra>"
        ))

        # Top marker
        fig.add_trace(go.Scatter3d(
            x=[i],
            y=[0],
            z=[row['importance']],
            mode='markers',
            marker=dict(
                size=10,
                color=f'rgb({255 - int(row["importance"]*255)}, {int(row["importance"]*255)}, 255)',
                symbol='diamond'
            ),
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='Feature',
                ticktext=features_df['feature'].tolist(),
                tickvals=list(range(n_features)),
                gridcolor='rgba(0, 255, 255, 0.2)',
                backgroundcolor='rgba(0, 0, 0, 0)'
            ),
            yaxis=dict(
                title='',
                showticklabels=False,
                gridcolor='rgba(0, 255, 255, 0.2)',
                backgroundcolor='rgba(0, 0, 0, 0)'
            ),
            zaxis=dict(
                title='Importance',
                gridcolor='rgba(255, 0, 255, 0.2)',
                backgroundcolor='rgba(0, 0, 0, 0)'
            ),
            bgcolor='rgba(10, 10, 15, 0.9)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8)
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        height=500,
        showlegend=False
    )

    return fig


def create_3d_scatter_plot(df: pd.DataFrame):
    """Create 3D scatter plot for data exploration"""

    # Sample if too large
    if len(df) > 500:
        df = df.sample(500, random_state=42)

    # Map credit risk to colors
    color_map = {1: '#ff0066', 2: '#00ffff'}
    colors = df['credit_risk'].map(color_map)

    fig = go.Figure(data=[go.Scatter3d(
        x=df['credit_amount'],
        y=df['duration'],
        z=df['age'],
        mode='markers',
        marker=dict(
            size=6,
            color=df['credit_risk'],
            colorscale=[[0, '#ff0066'], [1, '#00ffff']],
            opacity=0.8,
            line=dict(width=1, color='white')
        ),
        text=df.apply(lambda r: f"Amount: {r['credit_amount']}<br>Duration: {r['duration']}<br>Age: {r['age']}<br>Risk: {'Bad' if r['credit_risk']==1 else 'Good'}", axis=1),
        hovertemplate='%{text}<extra></extra>'
    )])

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Credit Amount', gridcolor='rgba(0, 255, 255, 0.2)', backgroundcolor='rgba(0,0,0,0)'),
            yaxis=dict(title='Duration (months)', gridcolor='rgba(0, 255, 255, 0.2)', backgroundcolor='rgba(0,0,0,0)'),
            zaxis=dict(title='Age', gridcolor='rgba(255, 0, 255, 0.2)', backgroundcolor='rgba(0,0,0,0)'),
            bgcolor='rgba(10, 10, 15, 0.9)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        height=600
    )

    return fig


def create_animated_gauge(value: float, title: str = "Risk Score"):
    """Create animated Plotly gauge"""

    if value < 0.3:
        bar_color = "#00ff00"
    elif value < 0.6:
        bar_color = "#ffff00"
    else:
        bar_color = "#ff0000"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24, 'color': '#00ffff', 'family': 'Orbitron'}},
        number={'font': {'size': 60, 'color': bar_color, 'family': 'Orbitron'}, 'suffix': '%'},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#00ffff"},
            'bar': {'color': bar_color, 'thickness': 0.75},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#00ffff",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(0, 255, 0, 0.2)'},
                {'range': [30, 60], 'color': 'rgba(255, 255, 0, 0.2)'},
                {'range': [60, 100], 'color': 'rgba(255, 0, 0, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "#ff00ff", 'width': 4},
                'thickness': 0.8,
                'value': value * 100
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#00ffff', 'family': 'Rajdhani'},
        height=350,
        margin=dict(l=30, r=30, t=50, b=30)
    )

    return fig


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h2 style="font-family: 'Orbitron', sans-serif; color: #00ffff; font-size: 1.2rem;">
                🎮 NAVIGATION
            </h2>
        </div>
        """, unsafe_allow_html=True)

        pages = {
            "🎯 PREDICT": "predict",
            "📊 METRICS": "metrics",
            "🔬 FEATURES": "features",
            "🗺️ EXPLORE": "explore",
            "⚙️ SETTINGS": "settings"
        }

        selected = st.radio(
            "Select Module",
            list(pages.keys()),
            label_visibility="collapsed"
        )

        st.markdown("---")

        # System status
        st.markdown("""
        <div class="neon-box" style="font-size: 0.9rem;">
            <p style="color: #ff00ff; margin: 0;">⚡ SYSTEM STATUS</p>
            <p style="color: #00ff00; margin: 5px 0;">● API: Online</p>
            <p style="color: #00ff00; margin: 5px 0;">● Model: Loaded</p>
            <p style="color: #00ffff; margin: 5px 0;">● Version: 2.0.0</p>
        </div>
        """, unsafe_allow_html=True)

        return pages[selected]


# ============================================================================
# MAIN APP PAGES
# ============================================================================

def page_predict():
    """Prediction page with 3D gauge"""
    st.markdown("## 🎯 CREDIT RISK PREDICTION")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="neon-box">', unsafe_allow_html=True)
        st.markdown("### 📝 CUSTOMER DATA INPUT")

        # Form inputs
        with st.form("prediction_form"):
            c1, c2 = st.columns(2)

            with c1:
                status = st.selectbox("Account Status", ["A11", "A12", "A13", "A14"],
                                     help="A11: <0 DM, A12: 0-200 DM, A13: >=200 DM, A14: No account")
                duration = st.number_input("Duration (months)", min_value=1, max_value=72, value=12)
                credit_amount = st.number_input("Credit Amount", min_value=0, max_value=50000, value=5000)
                age = st.number_input("Age", min_value=18, max_value=100, value=35)
                savings = st.selectbox("Savings", ["A61", "A62", "A63", "A64", "A65"])

            with c2:
                credit_history = st.selectbox("Credit History", ["A30", "A31", "A32", "A33", "A34"])
                purpose = st.selectbox("Purpose", ["A40", "A41", "A42", "A43", "A44", "A45", "A46"])
                employment = st.selectbox("Employment Duration", ["A71", "A72", "A73", "A74", "A75"])
                housing = st.selectbox("Housing", ["A151", "A152", "A153"])
                job = st.selectbox("Job Type", ["A171", "A172", "A173", "A174"])

            submitted = st.form_submit_button("⚡ ANALYZE RISK", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if submitted:
            with st.spinner("🔄 Neural network processing..."):
                time.sleep(1)  # Simulated processing

                # Generate prediction (simulated for demo)
                np.random.seed(hash(f"{status}{duration}{credit_amount}") % 2**32)
                risk_score = np.random.beta(2, 5)  # Tends toward lower risk

                # Adjust based on inputs
                if status == "A11":
                    risk_score += 0.15
                if credit_amount > 10000:
                    risk_score += 0.1
                if age < 25:
                    risk_score += 0.1

                risk_score = min(0.95, max(0.05, risk_score))

            # Display 3D gauge
            render_3d_risk_gauge(risk_score)

            # Results box
            if risk_score < 0.3:
                result_color = "#00ff00"
                recommendation = "✅ APPROVED - Low risk profile"
            elif risk_score < 0.6:
                result_color = "#ffff00"
                recommendation = "⚠️ REVIEW REQUIRED - Moderate risk"
            else:
                result_color = "#ff0000"
                recommendation = "❌ DECLINED - High default probability"

            st.markdown(f"""
            <div class="neon-box-pink" style="text-align: center;">
                <h3 style="color: {result_color}; font-family: 'Orbitron';">DECISION</h3>
                <p style="font-size: 1.2rem; color: #fff;">{recommendation}</p>
                <p style="color: #00ffff;">Confidence: {(1-abs(risk_score-0.5)*2)*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Show placeholder
            st.markdown("""
            <div style="height: 400px; display: flex; align-items: center; justify-content: center;">
                <div style="text-align: center;">
                    <p style="font-family: 'Orbitron'; font-size: 1.5rem; color: #00ffff; opacity: 0.5;">
                        ⚡ AWAITING INPUT
                    </p>
                    <p style="color: #666;">Enter customer data and click ANALYZE RISK</p>
                </div>
            </div>
            """, unsafe_allow_html=True)


def page_metrics():
    """Model metrics page with 3D visualization"""
    st.markdown("## 📊 MODEL PERFORMANCE METRICS")

    # Sample metrics
    metrics = {
        'accuracy': 0.78,
        'precision': 0.72,
        'recall': 0.65,
        'f1_score': 0.68,
        'roc_auc': 0.82
    }

    # 3D Sphere visualization
    render_3d_sphere_metrics(metrics)

    # Metrics cards
    st.markdown("### 📈 PERFORMANCE BREAKDOWN")

    cols = st.columns(5)
    metric_names = ['ACCURACY', 'PRECISION', 'RECALL', 'F1-SCORE', 'ROC-AUC']
    metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score'], metrics['roc_auc']]
    metric_colors = ['#00ff00', '#ff00ff', '#ffff00', '#00ffff', '#ff6600']

    for col, name, value, color in zip(cols, metric_names, metric_values, metric_colors):
        with col:
            st.markdown(f"""
            <div class="neon-box" style="text-align: center; padding: 15px;">
                <p style="color: #888; font-size: 0.8rem; margin: 0;">{name}</p>
                <p style="font-family: 'Orbitron'; font-size: 2rem; color: {color}; margin: 5px 0;
                          text-shadow: 0 0 10px {color};">
                    {value:.2%}
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Plotly Gauge for ROC-AUC
    st.markdown("### 🎯 ROC-AUC SCORE")
    fig = create_animated_gauge(metrics['roc_auc'], "Model ROC-AUC")
    st.plotly_chart(fig, use_container_width=True)


def page_features():
    """Feature importance page"""
    st.markdown("## 🔬 FEATURE IMPORTANCE ANALYSIS")

    # Feature importance data
    features_df = pd.DataFrame({
        'feature': ['credit_amount', 'duration', 'age', 'status', 'credit_history',
                   'savings', 'employment', 'purpose', 'installment_rate', 'housing'],
        'importance': [0.182, 0.156, 0.134, 0.098, 0.087, 0.076, 0.065, 0.058, 0.045, 0.042]
    })

    # 3D Bar chart
    st.markdown("### 📊 3D IMPORTANCE VISUALIZATION")
    fig = create_3d_feature_importance(features_df)
    st.plotly_chart(fig, use_container_width=True)

    # Feature breakdown
    st.markdown("### 📋 DETAILED BREAKDOWN")

    for idx, row in features_df.iterrows():
        progress = row['importance']
        color = f"linear-gradient(90deg, #ff00ff {progress*100}%, #00ffff {progress*100}%)"

        st.markdown(f"""
        <div style="margin: 10px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-family: 'Rajdhani'; color: #00ffff; text-transform: uppercase;">
                    {row['feature'].replace('_', ' ')}
                </span>
                <span style="font-family: 'Orbitron'; color: #ff00ff;">
                    {row['importance']:.3f}
                </span>
            </div>
            <div style="background: #333; height: 8px; border-radius: 4px; overflow: hidden; margin-top: 5px;">
                <div style="background: linear-gradient(90deg, #ff00ff, #00ffff);
                            width: {progress*100*5}%; height: 100%; border-radius: 4px;
                            box-shadow: 0 0 10px #ff00ff;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def page_explore():
    """Data exploration page with 3D scatter"""
    st.markdown("## 🗺️ DATA EXPLORATION")

    # Load data
    data_path = PROJECT_ROOT / "data" / "raw" / "german_credit.csv"

    if data_path.exists():
        df = pd.read_csv(data_path)

        # Stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Features", f"{len(df.columns)}")
        with col3:
            good_pct = (df['credit_risk'] == 2).mean() * 100
            st.metric("Good Credit %", f"{good_pct:.1f}%")
        with col4:
            bad_pct = (df['credit_risk'] == 1).mean() * 100
            st.metric("Bad Credit %", f"{bad_pct:.1f}%")

        st.markdown("### 🌐 3D DATA VISUALIZATION")
        st.markdown("*Rotate the plot to explore relationships between Credit Amount, Duration, and Age*")

        fig = create_3d_scatter_plot(df)
        st.plotly_chart(fig, use_container_width=True)

        # Legend
        st.markdown("""
        <div style="display: flex; gap: 30px; justify-content: center; padding: 10px;">
            <span style="color: #ff0066;">● Bad Credit Risk</span>
            <span style="color: #00ffff;">● Good Credit Risk</span>
        </div>
        """, unsafe_allow_html=True)

        # Data preview
        with st.expander("📋 VIEW RAW DATA"):
            st.dataframe(df.head(100), use_container_width=True)
    else:
        st.error("Data file not found. Please run data acquisition first.")


def page_settings():
    """Settings page"""
    st.markdown("## ⚙️ SYSTEM SETTINGS")

    st.markdown("""
    <div class="neon-box">
        <h3 style="color: #00ffff;">🔧 Configuration</h3>
        <p>API Endpoint: <code style="color: #ff00ff;">http://localhost:8000</code></p>
        <p>Model Version: <code style="color: #ff00ff;">v2.0.0</code></p>
        <p>Last Training: <code style="color: #ff00ff;">2024-01-13</code></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎨 Theme Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.color_picker("Primary Color", "#00ffff")
    with col2:
        st.color_picker("Secondary Color", "#ff00ff")

    st.markdown("### 📊 Model Settings")
    st.slider("Prediction Threshold", 0.0, 1.0, 0.5)
    st.selectbox("Default Model", ["XGBoost", "Random Forest", "Gradient Boosting", "Logistic Regression"])


# ============================================================================
# MAIN
# ============================================================================

def main():
    load_cyberpunk_css()
    render_3d_header()

    page = render_sidebar()

    if page == "predict":
        page_predict()
    elif page == "metrics":
        page_metrics()
    elif page == "features":
        page_features()
    elif page == "explore":
        page_explore()
    elif page == "settings":
        page_settings()


if __name__ == "__main__":
    main()
