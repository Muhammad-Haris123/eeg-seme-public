"""
ChemBERTa Embedding Module.

This module generates molecular embeddings using ChemBERTa model from HuggingFace.
ChemBERTa is a transformer model pre-trained on chemical SMILES strings.

Author: Research Team
Date: 2026
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from ..utils.config import CHEMBERTA_CONFIG, DRUG_EMBEDDINGS_DIR

warnings.filterwarnings('ignore')


class ChemBERTaEmbedder:
    """
    Embedder for drug molecules using ChemBERTa.
    
    ChemBERTa generates contextual embeddings from SMILES strings,
    suitable for downstream fusion with EEG features.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize ChemBERTa embedder.
        
        Args:
            model_name: HuggingFace model name (defaults to config)
            device: Device ('cpu' or 'cuda', defaults to auto-detect)
        """
        self.model_name = model_name or CHEMBERTA_CONFIG['model_name']
        self.max_length = CHEMBERTA_CONFIG['max_length']
        
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Loading ChemBERTa model: {self.model_name}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading ChemBERTa model: {str(e)}")
            print("Falling back to alternative model or manual SMILES embedding")
            raise
    
    def embed_smiles(
        self,
        smiles: str,
        pooling: str = 'mean'
    ) -> np.ndarray:
        """
        Generate embedding for a SMILES string.
        
        Args:
            smiles: SMILES string
            pooling: Pooling strategy ('mean', 'cls', 'max')
        
        Returns:
            Embedding vector (numpy array)
        """
        # Tokenize
        encoded = self.tokenizer(
            smiles,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**encoded)
            hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
        
        # Pooling
        if pooling == 'mean':
            # Mean pooling (excluding padding)
            attention_mask = encoded['attention_mask'].unsqueeze(-1)
            embedding = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        elif pooling == 'cls':
            # Use [CLS] token
            embedding = hidden_states[:, 0, :]
        elif pooling == 'max':
            # Max pooling
            embedding = hidden_states.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}")
        
        # Convert to numpy
        embedding = embedding.cpu().numpy().squeeze()
        
        return embedding
    
    def embed_multiple_smiles(
        self,
        smiles_list: List[str],
        pooling: str = 'mean'
    ) -> np.ndarray:
        """
        Generate embeddings for multiple SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            pooling: Pooling strategy
        
        Returns:
            Embedding matrix (n_smiles, embedding_dim)
        """
        embeddings = []
        
        for smiles in smiles_list:
            embedding = self.embed_smiles(smiles, pooling=pooling)
            embeddings.append(embedding)
        
        return np.stack(embeddings, axis=0)
    
    def embed_drugs_from_dict(
        self,
        smiles_dict: Dict[str, str],
        pooling: str = 'mean'
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for drugs from SMILES dictionary.
        
        Args:
            smiles_dict: Dictionary mapping drug keys to SMILES
            pooling: Pooling strategy
        
        Returns:
            Dictionary mapping drug keys to embeddings
        """
        drug_embeddings = {}
        
        for drug_key, smiles in smiles_dict.items():
            print(f"Embedding {drug_key}...")
            embedding = self.embed_smiles(smiles, pooling=pooling)
            drug_embeddings[drug_key] = embedding
            print(f"  Embedding shape: {embedding.shape}")
        
        return drug_embeddings
    
    def save_embeddings(
        self,
        embeddings_dict: Dict[str, np.ndarray],
        output_dir: Optional[Path] = None
    ):
        """
        Save drug embeddings to disk.
        
        Args:
            embeddings_dict: Dictionary mapping drug keys to embeddings
            output_dir: Output directory
        """
        if output_dir is None:
            output_dir = DRUG_EMBEDDINGS_DIR
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual embeddings
        for drug_key, embedding in embeddings_dict.items():
            embedding_path = output_dir / f"{drug_key}_embedding.npy"
            np.save(embedding_path, embedding)
            print(f"Saved {drug_key} embedding: {embedding_path}")
        
        # Save stacked embeddings
        drug_keys = list(embeddings_dict.keys())
        embeddings_array = np.stack([embeddings_dict[key] for key in drug_keys], axis=0)
        stacked_path = output_dir / "drug_embeddings.npy"
        np.save(stacked_path, embeddings_array)
        
        # Save metadata
        metadata = {
            'drug_keys': drug_keys,
            'embedding_dim': embeddings_array.shape[1],
            'n_drugs': len(drug_keys),
            'model_name': self.model_name,
            'device': self.device
        }
        metadata_path = output_dir / "drug_embeddings_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved stacked embeddings: {stacked_path}")
        print(f"Saved metadata: {metadata_path}")
        
        return stacked_path


def generate_drug_embeddings(
    smiles_dict: Optional[Dict[str, str]] = None,
    smiles_file: Optional[Path] = None,
    save_output: bool = True
) -> Dict[str, np.ndarray]:
    """
    Generate embeddings for drugs from SMILES.
    
    Args:
        smiles_dict: Dictionary of SMILES (if None, will load from file)
        smiles_file: Path to SMILES JSON file
        save_output: Whether to save embeddings
    
    Returns:
        Dictionary of drug embeddings
    """
    # Load SMILES if not provided
    if smiles_dict is None:
        if smiles_file is None:
            smiles_file = DRUG_EMBEDDINGS_DIR / "drug_smiles.json"
        
        if not smiles_file.exists():
            raise FileNotFoundError(
                f"SMILES file not found: {smiles_file}. "
                "Please run fetch_smiles.py first."
            )
        
        with open(smiles_file, 'r') as f:
            smiles_dict = json.load(f)
    
    # Initialize embedder
    embedder = ChemBERTaEmbedder()
    
    # Generate embeddings
    embeddings = embedder.embed_drugs_from_dict(smiles_dict)
    
    # Save if requested
    if save_output:
        embedder.save_embeddings(embeddings)
    
    return embeddings


if __name__ == "__main__":
    # Example usage
    embeddings = generate_drug_embeddings(save_output=True)
    print(f"\nGenerated embeddings for {len(embeddings)} drugs")



