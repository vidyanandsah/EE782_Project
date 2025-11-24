"""
AI Outfit Recommender - MULTIMODAL Core ML Module
EE782 Project - DeepFashion MultiModal Dataset

This module uses ALL available data:
1. Images - Visual features via ResNet50
2. Segmentation - Masked region features
3. Keypoints - Body pose/structure features
4. Texture Labels - Fabric + Pattern attributes
5. Shape Labels - 12 shape attributes
6. Captions - Text features via Sentence Transformers

Complete multimodal fusion for outfit recommendation!

Dataset Source: https://github.com/yumingj/DeepFashion-MultiModal
"""

import os
import pickle
import json
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import warnings
import requests
warnings.filterwarnings('ignore')

# Deep Learning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Text processing
from sentence_transformers import SentenceTransformer

# Similarity search
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize, StandardScaler


# ============================================================================
# STEP 0: DATASET DOWNLOADER (Google Drive Compatible)
# ============================================================================

class DeepFashionDownloader:
    """
    Downloads DeepFashion-MultiModal dataset from Google Drive
    Official Repository: https://github.com/yumingj/DeepFashion-MultiModal
    
    Citation:
    Jiang et al., "Text2Human: Text-Driven Controllable Human Image Generation"
    SIGGRAPH 2022
    """
    
    # Direct download links from the shared folder
    DATASET_LINKS = {
        'images': 'https://drive.google.com/uc?id=1U2PljA7NE57jcSSzPs21ZurdIPXdYZtN&export=download',
        'segm': 'https://drive.google.com/uc?id=1r-5t-VgDaAQidZLVgWtguaG7DvMoyUv9&export=download',
        'keypoints': 'https://drive.google.com/uc?id=1ZXdOQI-d4zNhqRJdUEWSQvPwAtLdjovo&export=download',
        'labels': 'https://drive.google.com/uc?id=11WoM5ZFwWpVjrIvZajW0g8EmQCNKMAWH&export=download',
        'captions': 'https://drive.google.com/uc?id=1d1TRm8UMcQhZCb6HpPo8l3OPEin4Ztk2&export=download',
        'densepose': 'https://drive.google.com/uc?id=14uyqBUDDcL1VLaXm7qmqghdcbkFuQa1s&export=download',
    }
    
    def __init__(self, root_dir="./deepfashion_data"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        print(f"📂 Dataset directory: {self.root_dir.absolute()}")
    
    def download_file_from_google_drive(self, url, output_path, file_name):
        """Download file from Google Drive with proper handling"""
        print(f"\n📥 Downloading {file_name}...")
        
        if output_path.exists():
            print(f"   ✅ {file_name} already exists, skipping download")
            return True
        
        try:
            session = requests.Session()
            
            # First request to get the file
            response = session.get(url, stream=True)
            
            # Check if we need to handle the virus scan warning
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    params = {'confirm': value}
                    response = session.get(url, params=params, stream=True)
                    break
            
            # Get total file size
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            
            # Download with progress bar
            with open(output_path, 'wb') as f:
                if total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=file_name) as pbar:
                        for chunk in response.iter_content(chunk_size=block_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    # No content-length header, download without progress bar
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
            
            print(f"   ✅ Downloaded {file_name}")
            return True
            
        except Exception as e:
            print(f"   ❌ Error downloading {file_name}: {e}")
            print(f"   💡 Please download manually from:")
            print(f"      {url}")
            if output_path.exists():
                output_path.unlink()
            return False
    
    def extract_zip(self, zip_path, extract_to):
        """Extract ZIP file"""
        print(f"\n📦 Extracting {zip_path.name}...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get list of files
                file_list = zip_ref.namelist()
                
                # Extract with progress bar
                with tqdm(total=len(file_list), desc="Extracting") as pbar:
                    for file in file_list:
                        zip_ref.extract(file, extract_to)
                        pbar.update(1)
            
            print(f"   ✅ Extracted to {extract_to}")
            
            # Remove ZIP file after extraction to save space
            zip_path.unlink()
            print(f"   🗑  Removed ZIP file to save space")
            
            return True
        except Exception as e:
            print(f"   ❌ Error extracting: {e}")
            return False
    
    def rename_folders(self):
        """Rename folders to match expected structure"""
        # Rename 'image' to 'images' if needed
        old_name = self.root_dir / "image"
        new_name = self.root_dir / "images"
        if old_name.exists() and not new_name.exists():
            old_name.rename(new_name)
            print(f"   📁 Renamed 'image' → 'images'")
    
    def create_metadata_csv(self):
        """Create deepfashion_processed.csv from downloaded data"""
        print("\n📝 Creating metadata CSV...")
        
        images_dir = self.root_dir / "images"
        csv_path = self.root_dir / "deepfashion_processed.csv"
        
        if csv_path.exists():
            print("   ✅ CSV already exists")
            return csv_path
        
        if not images_dir.exists():
            print("   ⚠  Images directory not found, skipping CSV creation")
            return None
        
        # Get all image files
        image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
        
        if len(image_files) == 0:
            print("   ⚠  No images found")
            return None
        
        # Load captions if available
        captions_path = self.root_dir / "captions.json"
        captions = {}
        if captions_path.exists():
            with open(captions_path, 'r') as f:
                captions = json.load(f)
        
        # Create DataFrame
        data = []
        for img_path in tqdm(image_files, desc="Processing images"):
            img_name = img_path.name
            data.append({
                'image_name': img_name,
                'image_path': str(img_path.absolute()),
                'caption': captions.get(img_name, 'No caption available'),
                'category': 'unknown',
                'colors': 'not specified',
                'styles': 'not specified',
                'patterns': 'not specified',
                'materials': 'not specified',
            })
        
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        
        print(f"   ✅ Created CSV with {len(df)} images")
        return csv_path
    
    def download_dataset(self, components='minimal'):
        """
        Download specified components of the dataset
        
        Args:
            components: 
                - 'minimal': Essential files only (images, keypoints, labels, captions)
                - 'standard': Minimal + segmentation masks
                - 'all': Everything
                - list: Specific components ['images', 'keypoints', etc.]
        """
        
        print("="*70)
        print("📥 DEEPFASHION-MULTIMODAL DATASET DOWNLOADER")
        print("   Source: https://github.com/yumingj/DeepFashion-MultiModal")
        print("   Citation: Jiang et al., SIGGRAPH 2022")
        print("="*70)
        
        # Determine what to download
        if components == 'minimal':
            to_download = ['keypoints', 'labels', 'captions']
            print("\n📦 Mode: MINIMAL (keypoints, labels, captions)")
            print("⚠️  Note: Images are large (6.35 GB). Download separately if needed.")
        elif components == 'standard':
            to_download = ['segm', 'keypoints', 'labels', 'captions']
            print("\n📦 Mode: STANDARD (minimal + segmentation masks)")
        elif components == 'all':
            to_download = list(self.DATASET_LINKS.keys())
            print("\n📦 Mode: ALL COMPONENTS (includes images, densepose)")
            print("⚠️  Warning: Total size ~12 GB")
        else:
            to_download = components if isinstance(components, list) else [components]
            print(f"\n📦 Mode: CUSTOM ({', '.join(to_download)})")
        
        # Download each component
        for component in to_download:
            if component not in self.DATASET_LINKS:
                print(f"⚠  Unknown component: {component}")
                continue
            
            url = self.DATASET_LINKS[component]
            
            # Determine output paths
            if component == 'captions':
                output_path = self.root_dir / "captions.json"
                needs_extraction = False
            else:
                output_path = self.root_dir / f"{component}.zip"
                needs_extraction = True
            
            # Download
            success = self.download_file_from_google_drive(url, output_path, component)
            
            # Extract if needed
            if success and needs_extraction and output_path.suffix == '.zip':
                self.extract_zip(output_path, self.root_dir)
        
        # Rename folders to match expected structure
        self.rename_folders()
        
        # Create metadata CSV
        if 'images' in to_download or (self.root_dir / "images").exists():
            self.create_metadata_csv()
        
        print("\n" + "="*70)
        print("✅ DATASET DOWNLOAD COMPLETE!")
        print("="*70)
        print(f"\n📂 Dataset location: {self.root_dir.absolute()}")
        
        # Show directory structure
        if self.root_dir.exists():
            print(f"\n📊 Directory structure:")
            for item in sorted(self.root_dir.iterdir()):
                if item.is_dir():
                    num_files = len(list(item.glob("*")))
                    print(f"   📁 {item.name}/ ({num_files} files)")
                else:
                    size_mb = item.stat().st_size / (1024*1024)
                    print(f"   📄 {item.name} ({size_mb:.1f} MB)")
        
        print("\n💡 You can now run training with this dataset!")
        return self.root_dir


# ============================================================================
# STEP 1: DATA LOADING AND PREPROCESSING
# ============================================================================

class MultiModalDataLoader:
    """Loads all modalities from DeepFashion dataset"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.segm_dir = self.data_dir / "segm"
        self.csv_path = self.data_dir / "deepfashion_processed.csv"
        self.captions_path = self.data_dir / "captions.json"
        
        # Label files
        self.fabric_path = self.data_dir / "labels/texture/fabric_ann.txt"
        self.pattern_path = self.data_dir / "labels/texture/pattern_ann.txt"
        self.shape_path = self.data_dir / "labels/shape/shape_anno_all.txt"
        self.keypoint_loc_path = self.data_dir / "keypoints/keypoints_loc.txt"
        self.keypoint_vis_path = self.data_dir / "keypoints/keypoints_vis.txt"
    
    def load_all_data(self, sample_size=None):
        """Load and merge all modalities"""
        print("\n" + "="*70)
        print("📁 STEP 1: LOADING MULTIMODAL DATA")
        print("="*70)
        
        # Load base CSV
        print("\n1️⃣ Loading base CSV...")
        df = pd.read_csv(self.csv_path)
        print(f"   Loaded {len(df)} records")
        
        # Filter existing images
        df['image_exists'] = df['image_path'].apply(lambda x: os.path.exists(x))
        df = df[df['image_exists']].copy()
        print(f"   Found {len(df)} existing images")
        
        # Load captions
        print("\n2️⃣ Loading captions...")
        if self.captions_path.exists():
            with open(self.captions_path, 'r') as f:
                captions = json.load(f)
            df['caption'] = df['image_name'].map(captions)
            df['caption'] = df['caption'].fillna('No caption')
            print(f"   Loaded {len(captions)} captions")
        
        # Load fabric labels
        print("\n3️⃣ Loading fabric labels...")
        fabric_dict = self._load_label_file(self.fabric_path, num_cols=3)
        df['fabric_0'] = df['image_name'].map(lambda x: fabric_dict.get(x, [0,0,0])[0])
        df['fabric_1'] = df['image_name'].map(lambda x: fabric_dict.get(x, [0,0,0])[1])
        df['fabric_2'] = df['image_name'].map(lambda x: fabric_dict.get(x, [0,0,0])[2])
        print(f"   Loaded {len(fabric_dict)} fabric annotations")
        
        # Load pattern labels
        print("\n4️⃣ Loading pattern labels...")
        pattern_dict = self._load_label_file(self.pattern_path, num_cols=3)
        df['pattern_0'] = df['image_name'].map(lambda x: pattern_dict.get(x, [0,0,0])[0])
        df['pattern_1'] = df['image_name'].map(lambda x: pattern_dict.get(x, [0,0,0])[1])
        df['pattern_2'] = df['image_name'].map(lambda x: pattern_dict.get(x, [0,0,0])[2])
        print(f"   Loaded {len(pattern_dict)} pattern annotations")
        
        # Load shape labels
        print("\n5️⃣ Loading shape labels...")
        shape_dict = self._load_label_file(self.shape_path, num_cols=12)
        for i in range(12):
            df[f'shape_{i}'] = df['image_name'].map(lambda x: shape_dict.get(x, [0]*12)[i])
        print(f"   Loaded {len(shape_dict)} shape annotations")
        
        # Load keypoint locations
        print("\n6️⃣ Loading keypoint locations...")
        keypoint_loc_dict = self._load_label_file(self.keypoint_loc_path, num_cols=42)
        for i in range(42):
            df[f'kp_loc_{i}'] = df['image_name'].map(lambda x: keypoint_loc_dict.get(x, [0]*42)[i])
        print(f"   Loaded {len(keypoint_loc_dict)} keypoint locations (21 points)")
        
        # Load keypoint visibility
        print("\n7️⃣ Loading keypoint visibility...")
        keypoint_vis_dict = self._load_label_file(self.keypoint_vis_path, num_cols=21)
        for i in range(21):
            df[f'kp_vis_{i}'] = df['image_name'].map(lambda x: keypoint_vis_dict.get(x, [0]*21)[i])
        print(f"   Loaded {len(keypoint_vis_dict)} keypoint visibility flags")
        
        # Check segmentation availability
        print("\n8️⃣ Checking segmentation masks...")
        df['segm_path'] = df['image_name'].apply(
            lambda x: str(self.segm_dir / (x.replace('.jpg', '_segm.png')))
        )
        df['has_segm'] = df['segm_path'].apply(lambda x: os.path.exists(x))
        print(f"   Found {df['has_segm'].sum()} segmentation masks")
        
        # Sample if requested
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            print(f"\n✂  Sampled {sample_size} images for processing")
        
        print(f"\n✅ Total multimodal records: {len(df)}")
        print(f"   Columns: {len(df.columns)}")
        
        return df
    
    def _load_label_file(self, filepath, num_cols):
        """Load label file with multiple columns"""
        label_dict = {}
        if filepath.exists():
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= num_cols + 1:
                        img_name = parts[0]
                        labels = [int(x) for x in parts[1:num_cols+1]]
                        label_dict[img_name] = labels
        return label_dict


# ============================================================================
# STEP 2: MULTIMODAL FEATURE EXTRACTION
# ============================================================================

class MultiModalFashionDataset(Dataset):
    """Dataset that loads image + segmentation"""
    
    def __init__(self, df, img_transform=None, segm_transform=None):
        self.df = df
        self.img_transform = img_transform
        self.segm_transform = segm_transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        try:
            img = Image.open(row['image_path']).convert('RGB')
            if self.img_transform:
                img = self.img_transform(img)
        except:
            img = self.img_transform(Image.new('RGB', (224, 224)))
        
        # Load segmentation if available
        if row['has_segm']:
            try:
                segm = Image.open(row['segm_path']).convert('L')
                if self.segm_transform:
                    segm = self.segm_transform(segm)
            except:
                segm = torch.zeros(1, 224, 224)
        else:
            segm = torch.zeros(1, 224, 224)
        
        return img, segm, row['image_path']


class MultiModalFeatureExtractor(nn.Module):
    """
    Extract features from multiple modalities:
    - RGB image features (ResNet50)
    - Segmentation-masked features
    - Fused multimodal representation
    """
    
    def __init__(self, pretrained=True):
        super(MultiModalFeatureExtractor, self).__init__()
        
        # ResNet50 backbone for RGB images
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.rgb_features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Segmentation branch
        self.segm_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(2048 + 256, 2048, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
    
    def forward(self, img, segm):
        # RGB features
        rgb_feat = self.rgb_features(img)
        
        # Segmentation features
        segm_feat = self.segm_conv(segm)
        
        # Resize segm_feat to match rgb_feat
        if segm_feat.shape[2:] != rgb_feat.shape[2:]:
            segm_feat = F.interpolate(segm_feat, size=rgb_feat.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate and fuse
        combined = torch.cat([rgb_feat, segm_feat], dim=1)
        fused = self.fusion(combined)
        fused = fused.view(fused.size(0), -1)
        
        return fused


# ============================================================================
# STEP 3: TEXT FEATURE EXTRACTION
# ============================================================================

class TextFeatureExtractor:
    """Extract semantic features from captions using Sentence Transformers"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"\n🔤 Loading text encoder: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"   Text embedding dimension: {self.embedding_dim}")
    
    def extract_batch(self, captions, batch_size=32):
        """Extract features from list of captions"""
        print(f"\n📝 Extracting text features from {len(captions)} captions...")
        embeddings = self.model.encode(
            captions,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings


# ============================================================================
# STEP 4: MULTIMODAL OUTFIT RECOMMENDER
# ============================================================================

class MultiModalOutfitRecommender:
    """Main class for multimodal outfit recommendation"""
    
    def __init__(self, data_dir="./deepfashion_data", auto_download=False):
        """
        Initialize the recommender
        
        Args:
            data_dir: Directory containing the dataset
            auto_download: If True, automatically download dataset if not found
        """
        self.data_dir = Path(data_dir)
        self.auto_download = auto_download
        
        # Check if dataset exists, download if needed
        if auto_download and not self._check_dataset_exists():
            print("\n⚠  Dataset not found locally.")
            print("📥 Auto-download is enabled. Downloading from Google Drive...")
            self._download_dataset()
        
        # Artifacts directory
        self.artifacts_dir = self.data_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Artifact paths
        self.visual_features_path = self.artifacts_dir / "visual_features.npy"
        self.text_features_path = self.artifacts_dir / "text_features.npy"
        self.attribute_features_path = self.artifacts_dir / "attribute_features.npy"
        self.keypoint_features_path = self.artifacts_dir / "keypoint_features.npy"
        self.fused_features_path = self.artifacts_dir / "fused_features.npy"
        self.metadata_path = self.artifacts_dir / "metadata.pkl"
        self.model_path = self.artifacts_dir / "multimodal_model.pth"
        self.knn_path = self.artifacts_dir / "knn_index.pkl"
        self.scaler_path = self.artifacts_dir / "scaler.pkl"
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n🖥  Using device: {self.device}")
        
        # Transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.segm_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        self.model = None
        self.text_model = None
        self.scaler = None
        self.knn = None
        self.metadata = None
    
    def _check_dataset_exists(self):
        """Check if dataset is already downloaded"""
        required_paths = [
            self.data_dir / "deepfashion_processed.csv",
        ]
        exists = all(p.exists() for p in required_paths)
        
        if exists:
            print(f"✅ Dataset found at {self.data_dir}")
        
        return exists
    
    def _download_dataset(self, components='standard'):
        """Download dataset from Google Drive"""
        downloader = DeepFashionDownloader(root_dir=str(self.data_dir))
        downloader.download_dataset(components=components)
        print("\n✅ Dataset ready!")
    
    def download_full_dataset(self):
        """Download the complete dataset"""
        print("\n🚀 Downloading FULL dataset...")
        print("⚠  Note: This will download ~12 GB of data")
        self._download_dataset(components='all')
    
    def train_pipeline(self, sample_size=None, batch_size=32):
        """Complete multimodal training pipeline"""
        
        print("\n" + "="*70)
        print("🚀 MULTIMODAL AI OUTFIT RECOMMENDER - TRAINING PIPELINE")
        print("="*70)
        
        # STEP 1: Load data
        loader = MultiModalDataLoader(self.data_dir)
        df = loader.load_all_data(sample_size=sample_size)
        
        # STEP 2: Extract visual features
        visual_features = self._extract_visual_features(df, batch_size)
        
        # STEP 3: Extract text features
        text_features = self._extract_text_features(df)
        
        # STEP 4: Extract attribute features
        attribute_features = self._extract_attribute_features(df)
        
        # STEP 5: Extract keypoint features
        keypoint_features = self._extract_keypoint_features(df)
        
        # STEP 6: Fuse all modalities
        fused_features = self._fuse_features(
            visual_features, text_features, attribute_features, keypoint_features
        )
        
        # STEP 7: Build similarity index
        self._build_similarity_index(fused_features)
        
        # STEP 8: Save everything
        self._save_artifacts(df, visual_features, text_features, 
                           attribute_features, keypoint_features, fused_features)
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\n📊 Summary:")
        print(f"   - Total samples: {len(df)}")
        print(f"   - Visual features: {visual_features.shape}")
        print(f"   - Text features: {text_features.shape}")
        print(f"   - Attribute features: {attribute_features.shape}")
        print(f"   - Keypoint features: {keypoint_features.shape}")
        print(f"   - Fused features: {fused_features.shape}")
        print(f"   - Artifacts saved to: {self.artifacts_dir}")
    
    def _extract_visual_features(self, df, batch_size):
        """Extract visual features from images + segmentation"""
        print("\n" + "="*70)
        print("🎨 STEP 2: EXTRACTING VISUAL FEATURES")
        print("="*70)
        
        self.model = MultiModalFeatureExtractor(pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("✅ Multimodal CNN loaded")
        
        dataset = MultiModalFashionDataset(df, self.img_transform, self.segm_transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        all_features = []
        
        with torch.no_grad():
            for imgs, segms, paths in tqdm(dataloader, desc="Extracting visual features"):
                imgs = imgs.to(self.device)
                segms = segms.to(self.device)
                
                features = self.model(imgs, segms)
                all_features.append(features.cpu().numpy())
        
        visual_features = np.vstac
