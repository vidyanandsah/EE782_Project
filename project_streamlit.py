"""
AI Outfit Recommender - MULTIMODAL Streamlit GUI
EE782 Project - DeepFashion MultiModal Dataset

Interactive web interface showcasing:
- Visual similarity (RGB + Segmentation)
- Text similarity (Captions)
- Attribute matching (Fabric, Pattern, Shape)
- Pose similarity (Keypoints)
"""

import streamlit as st
import os
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ee782_project import MultiModalOutfitRecommender


# Page configuration
st.set_page_config(
    page_title="Multimodal AI Outfit Recommender",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.3rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .modality-badge {
        display: inline-block;
        padding: 5px 12px;
        margin: 3px;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .badge-visual { background-color: #FF6B6B; color: white; }
    .badge-text { background-color: #4ECDC4; color: white; }
    .badge-attr { background-color: #FFD93D; color: black; }
    .badge-pose { background-color: #95E1D3; color: black; }
    .badge-segm { background-color: #AA96DA; color: white; }
    .similarity-score {
        font-size: 1.5rem;
        font-weight: bold;
        color: #4CAF50;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(120deg, #FF6B6B 0%, #4ECDC4 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 12px;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_recommender_model(data_dir='./deepfashion_data'):
    """Load the trained multimodal recommender system"""
    try:
        recommender = MultiModalOutfitRecommender(data_dir=data_dir, auto_download=False)
        recommender.load_artifacts()
        return recommender, None
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


def display_modality_badges():
    """Display badges for all modalities"""
    st.markdown("""
        <div style='text-align: center; margin: 20px 0;'>
            <span class='modality-badge badge-visual'>🎨 Visual</span>
            <span class='modality-badge badge-segm'>✂ Segmentation</span>
            <span class='modality-badge badge-text'>📝 Text</span>
            <span class='modality-badge badge-attr'>🏷 Attributes</span>
            <span class='modality-badge badge-pose'>🦴 Pose</span>
        </div>
    """, unsafe_allow_html=True)


def display_image_with_segmentation(image_path, segm_path=None, title="Image"):
    """Display image with optional segmentation overlay"""
    try:
        img = Image.open(image_path).convert('RGB')
        
        if segm_path and os.path.exists(segm_path):
            # Create side-by-side view
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(img, caption=f"{title} - Original", use_container_width=True)
            
            with col2:
                segm = Image.open(segm_path).convert('L')
                st.image(segm, caption=f"{title} - Segmentation", use_container_width=True)
        else:
            st.image(img, caption=title, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error displaying image: {e}")


def display_result_card(result, show_segm=True):
    """Display a recommendation result card"""
    
    st.markdown(f"""
        <div style='border: 3px solid #f0f0f0; border-radius: 15px; padding: 20px; margin: 15px 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;'>
            <h3 style='margin: 0;'>🏆 Rank #{result['rank']}</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Image display
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Main image
        img = Image.open(result['image_path'])
        st.image(img, use_container_width=True)
        
        # Segmentation if available
        if show_segm and result.get('has_segmentation'):
            segm_path = result['image_path'].replace('.jpg', '_segm.png').replace('/images/', '/segm/')
            if os.path.exists(segm_path):
                segm = Image.open(segm_path).convert('L')
                st.image(segm, caption="Segmentation Mask", use_container_width=True)
    
    with col2:
        # Similarity score
        score_pct = result['similarity_score'] * 100
        st.markdown(f"<div class='similarity-score'>Similarity: {score_pct:.1f}%</div>", 
                   unsafe_allow_html=True)
        
        # Progress bar
        st.progress(result['similarity_score'])
        
        # Metadata
        st.markdown("### 📝 Details")
        st.markdown(f"*Image:* {result['image_name']}")
        st.markdown(f"*Materials:* {result['materials']}")
        st.markdown(f"*Patterns:* {result['patterns']}")
        st.markdown(f"*Category:* {result['category']}")
        
        # Caption
        if result.get('caption'):
            with st.expander("📖 Full Description"):
                st.write(result['caption'])
    
    st.markdown("---")


def main():
    # Header
    st.markdown('<div class="main-header">👗 Multimodal AI Outfit Recommender</div>', 
                unsafe_allow_html=True)
    st.markdown('''<div class="sub-header">
        Powered by 5 Modalities: Visual + Segmentation + Text + Attributes + Pose<br>
        EE782 Project | DeepFashion MultiModal Dataset
        </div>''', 
                unsafe_allow_html=True)
    
    # Display modality badges
    display_modality_badges()
    
    # Load model
    with st.spinner("🔄 Loading multimodal AI model..."):
        recommender, error = load_recommender_model()
    
    if error:
        st.error(f"❌ {error}")
        st.info("""
        *Please run training first:*
        
        Quick test (500 images):
        bash
        python ee782_project.py --test
        
        
        Full dataset:
        bash
        python ee782_project.py
        
        """)
        st.stop()
    
    st.success(f"✅ Multimodal model loaded! Database: {len(recommender.metadata)} outfits")
    
    # Sidebar
    st.sidebar.title("⚙ Settings")
    
    # Info about modalities
    with st.sidebar.expander("ℹ What makes this Multimodal?"):
        st.markdown("""
        This system uses *5 different data modalities*:
        
        1. *🎨 Visual Features* (2048-D)
           - ResNet50 CNN features from RGB images
        
        2. *✂ Segmentation Features* (256-D)
           - Masked clothing region features
        
        3. *📝 Text Features* (384-D)
           - Semantic embeddings from captions
        
        4. *🏷 Attribute Features* (18-D)
           - Fabric labels (3)
           - Pattern labels (3)
           - Shape labels (12)
        
        5. *🦴 Pose Features* (63-D)
           - 21 keypoint locations (42-D)
           - Keypoint visibility (21-D)
        
        *Total: ~2,769 dimensional representation!*
        """)
    
    # Input method
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Upload Image", "Select from Dataset", "Browse by Attributes"]
    )
    
    # Number of recommendations
    top_k = st.sidebar.slider(
        "Number of recommendations:",
        min_value=3,
        max_value=20,
        value=10,
        step=1
    )
    
    # Show segmentation masks
    show_segm = st.sidebar.checkbox("Show segmentation masks", value=True)
    
    # Main content
    query_image_path = None
    
    if input_method == "Upload Image":
        st.markdown("### 📤 Upload Your Outfit Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a fashion/outfit image"
        )
        
        if uploaded_file is not None:
            # Save temporarily
            temp_dir = Path("/Users/akashjaiswal/Desktop/EE782 Project/temp")
            temp_dir.mkdir(exist_ok=True)
            query_image_path = temp_dir / uploaded_file.name
            
            with open(query_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success("✅ Image uploaded!")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(uploaded_file, caption="Your Query Image", use_container_width=True)
    
    elif input_method == "Select from Dataset":
        st.markdown("### 🗂 Select from Dataset")
        
        selection_mode = st.radio("", ["Random Selection", "Browse"], horizontal=True)
        
        if selection_mode == "Random Selection":
            if st.button("🎲 Pick Random Image"):
                random_idx = np.random.randint(0, len(recommender.metadata))
                query_image_path = recommender.metadata[random_idx]['image_path']
                st.session_state['selected_image'] = query_image_path
                st.rerun()
        else:
            # Show selectbox
            display_images = recommender.metadata[:min(500, len(recommender.metadata))]
            image_options = {f"{i}: {m['image_name'][:50]}": m['image_path'] 
                           for i, m in enumerate(display_images)}
            
            selected_key = st.selectbox("Choose an image:", list(image_options.keys()))
            if selected_key:
                query_image_path = image_options[selected_key]
        
        if 'selected_image' in st.session_state:
            query_image_path = st.session_state['selected_image']
        
        if query_image_path and os.path.exists(query_image_path):
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                # Show with segmentation if available
                segm_path = query_image_path.replace('.jpg', '_segm.png').replace('/images/', '/segm/')
                display_image_with_segmentation(query_image_path, segm_path, "Selected Query")
    
    elif input_method == "Browse by Attributes":
        st.markdown("### 📂 Browse by Attributes")
        
        # Extract unique materials and patterns
        materials_set = set()
        patterns_set = set()
        
        for meta in recommender.metadata:
            mat = meta.get('materials', '')
            pat = meta.get('patterns', '')
            
            if mat and mat != 'not specified':
                materials_set.update([m.strip() for m in mat.split(',')])
            if pat and pat != 'not specified':
                patterns_set.update([p.strip() for p in pat.split(',')])
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_material = st.selectbox("Material:", ["Any"] + sorted(list(materials_set)))
        
        with col2:
            selected_pattern = st.selectbox("Pattern:", ["Any"] + sorted(list(patterns_set)))
        
        # Filter
        filtered = recommender.metadata.copy()
        
        if selected_material != "Any":
            filtered = [m for m in filtered if selected_material in m.get('materials', '')]
        
        if selected_pattern != "Any":
            filtered = [m for m in filtered if selected_pattern in m.get('patterns', '')]
        
        st.info(f"📊 Found {len(filtered)} matching outfits")
        
        if filtered and st.button("🎲 Pick Random from Filtered"):
            random_idx = np.random.randint(0, len(filtered))
            query_image_path = filtered[random_idx]['image_path']
            st.session_state['selected_image'] = query_image_path
            st.rerun()
    
    # Find similar button
    st.markdown("---")
    
    if query_image_path and os.path.exists(query_image_path):
        if st.button("🔍 Find Similar Outfits with Multimodal AI", type="primary"):
            
            with st.spinner(f"🤖 Analyzing with 5 modalities and searching {top_k} similar outfits..."):
                results = recommender.find_similar_outfits(query_image_path, top_k=top_k)
            
            if results:
                st.balloons()
                
                st.markdown(f"## 🎯 Top {len(results)} Multimodal Recommendations")
                st.markdown("Ranked by fusion of Visual + Segmentation + Text + Attributes + Pose similarities")
                
                # Display results
                for result in results:
                    display_result_card(result, show_segm=show_segm)
                
                # Export button
                results_df = pd.DataFrame(results)
                csv = results_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="multimodal_outfit_recommendations.csv",
                    mime="text/csv"
                )
                
                # Statistics
                st.markdown("### 📊 Recommendation Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_sim = np.mean([r['similarity_score'] for r in results])
                    st.metric("Average Similarity", f"{avg_sim:.2%}")
                
                with col2:
                    with_segm = sum(1 for r in results if r.get('has_segmentation'))
                    st.metric("With Segmentation", f"{with_segm}/{len(results)}")
                
                with col3:
                    top_score = results[0]['similarity_score']
                    st.metric("Top Match Score", f"{top_score:.2%}")
            
            else:
                st.error("No results found. Please try another image.")
    
    else:
        st.info("👆 Please select or upload an image to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p><b>🎓 EE782 Machine Learning Project - Multimodal Deep Learning</b></p>
            <p>Technologies: PyTorch • ResNet50 • Sentence Transformers • Streamlit</p>
            <p>5 Modalities: Visual • Segmentation • Text • Attributes • Pose</p>
        </div>
    """, unsafe_allow_html=True)


if _name_ == "_main_":
    main()
