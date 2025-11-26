#!/usr/bin/env python3
"""
AI Quality Inspector - Computer Vision for Manufacturing Quality Control
Combining trending AI automation with industrial computer vision

Author: Vinesh Thota
Date: November 26, 2024
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
import base64
from io import BytesIO
import time

# Configure Streamlit page
st.set_page_config(
    page_title="AI Quality Inspector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class QualityInspector:
    def __init__(self):
        self.confidence_threshold = 0.8
        self.defect_types = [
            "Surface Scratch", "Dent", "Color Variation", 
            "Dimensional Error", "Assembly Defect", "Contamination"
        ]
        self.quality_standards = {
            "automotive": {"threshold": 0.95, "critical_defects": ["Assembly Defect", "Dimensional Error"]},
            "electronics": {"threshold": 0.98, "critical_defects": ["Surface Scratch", "Contamination"]},
            "pharmaceutical": {"threshold": 0.99, "critical_defects": ["Contamination", "Dimensional Error"]}
        }
        
    def preprocess_image(self, image):
        """Preprocess image for defect detection"""
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection using Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        return gray, blurred, edges
    
    def detect_defects(self, image, product_type="automotive"):
        """Simulate AI-powered defect detection"""
        gray, blurred, edges = self.preprocess_image(image)
        
        # Simulate defect detection results
        defects_found = []
        quality_score = np.random.uniform(0.85, 0.99)
        
        # Find contours (potential defects)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze contours for defects
        for i, contour in enumerate(contours[:5]):  # Limit to top 5 contours
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small noise
                x, y, w, h = cv2.boundingRect(contour)
                
                # Simulate defect classification
                defect_type = np.random.choice(self.defect_types)
                confidence = np.random.uniform(0.7, 0.95)
                
                if confidence > self.confidence_threshold:
                    defects_found.append({
                        "type": defect_type,
                        "confidence": confidence,
                        "location": (x, y, w, h),
                        "severity": "Critical" if defect_type in self.quality_standards[product_type]["critical_defects"] else "Minor"
                    })
        
        # Adjust quality score based on defects
        if defects_found:
            critical_defects = sum(1 for d in defects_found if d["severity"] == "Critical")
            quality_score -= (critical_defects * 0.1 + len(defects_found) * 0.02)
            quality_score = max(0.0, quality_score)
        
        return {
            "quality_score": quality_score,
            "defects": defects_found,
            "pass_fail": quality_score >= self.quality_standards[product_type]["threshold"],
            "processed_image": edges
        }
    
    def draw_defects(self, image, defects):
        """Draw bounding boxes around detected defects"""
        result_image = image.copy()
        
        for defect in defects:
            x, y, w, h = defect["location"]
            color = (255, 0, 0) if defect["severity"] == "Critical" else (255, 165, 0)
            
            # Draw bounding box
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            
            # Add label
            label = f"{defect['type']} ({defect['confidence']:.2f})"
            cv2.putText(result_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result_image

def create_dashboard():
    """Create the main Streamlit dashboard"""
    st.title("🔍 AI Quality Inspector")
    st.markdown("### Real-time Manufacturing Quality Control with Computer Vision")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    product_type = st.sidebar.selectbox(
        "Product Type",
        ["automotive", "electronics", "pharmaceutical"]
    )
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.5,
        max_value=1.0,
        value=0.8,
        step=0.05
    )
    
    # Initialize quality inspector
    inspector = QualityInspector()
    inspector.confidence_threshold = confidence_threshold
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Image Upload & Analysis")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload product image for quality inspection",
            type=["jpg", "jpeg", "png"],
            help="Upload an image of the product to be inspected"
        )
        
        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            st.image(image, caption="Original Image", use_column_width=True)
            
            # Perform quality inspection
            with st.spinner("Analyzing image for defects..."):
                results = inspector.detect_defects(image_array, product_type)
            
            # Display results
            st.subheader("Inspection Results")
            
            # Quality score and pass/fail status
            col_score, col_status = st.columns(2)
            
            with col_score:
                quality_score = results["quality_score"]
                st.metric(
                    "Quality Score",
                    f"{quality_score:.2%}",
                    delta=f"{quality_score - inspector.quality_standards[product_type]['threshold']:.2%}"
                )
            
            with col_status:
                status = "✅ PASS" if results["pass_fail"] else "❌ FAIL"
                st.metric("Status", status)
            
            # Defects found
            if results["defects"]:
                st.subheader("Defects Detected")
                
                defects_df = pd.DataFrame(results["defects"])
                defects_df["confidence"] = defects_df["confidence"].apply(lambda x: f"{x:.2%}")
                
                st.dataframe(
                    defects_df[["type", "confidence", "severity"]],
                    use_container_width=True
                )
                
                # Draw defects on image
                annotated_image = inspector.draw_defects(image_array, results["defects"])
                st.image(annotated_image, caption="Defects Highlighted", use_column_width=True)
            else:
                st.success("No defects detected!")
            
            # Edge detection visualization
            st.subheader("Edge Detection Analysis")
            st.image(results["processed_image"], caption="Edge Detection", use_column_width=True)
    
    with col2:
        st.subheader("Quality Standards")
        
        # Display quality standards for selected product type
        standards = inspector.quality_standards[product_type]
        st.info(f"**Threshold**: {standards['threshold']:.1%}")
        st.info(f"**Critical Defects**: {', '.join(standards['critical_defects'])}")
        
        # Real-time metrics simulation
        st.subheader("Live Production Metrics")
        
        # Generate sample data
        current_time = datetime.now()
        sample_data = {
            "Total Inspected": np.random.randint(1000, 5000),
            "Pass Rate": np.random.uniform(0.92, 0.98),
            "Defects Found": np.random.randint(20, 150),
            "Avg Quality Score": np.random.uniform(0.88, 0.96)
        }
        
        for metric, value in sample_data.items():
            if isinstance(value, float):
                st.metric(metric, f"{value:.2%}" if "Rate" in metric or "Score" in metric else f"{value:.2f}")
            else:
                st.metric(metric, f"{value:,}")
        
        # Quality trend chart
        st.subheader("Quality Trend")
        
        # Generate sample trend data
        hours = list(range(24))
        quality_scores = [np.random.uniform(0.85, 0.98) for _ in hours]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hours,
            y=quality_scores,
            mode='lines+markers',
            name='Quality Score',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.add_hline(
            y=standards["threshold"],
            line_dash="dash",
            line_color="red",
            annotation_text="Quality Threshold"
        )
        
        fig.update_layout(
            title="24-Hour Quality Trend",
            xaxis_title="Hour",
            yaxis_title="Quality Score",
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Defect distribution
        st.subheader("Defect Distribution")
        
        defect_counts = {defect: np.random.randint(0, 20) for defect in inspector.defect_types}
        
        fig_pie = px.pie(
            values=list(defect_counts.values()),
            names=list(defect_counts.keys()),
            title="Defect Types Distribution"
        )
        fig_pie.update_layout(height=300)
        
        st.plotly_chart(fig_pie, use_container_width=True)

def main():
    """Main application entry point"""
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    create_dashboard()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**AI Quality Inspector** - Combining trending AI automation with industrial computer vision "
        "to solve real manufacturing challenges. Built with ❤️ for the future of smart manufacturing."
    )

if __name__ == "__main__":
    main()