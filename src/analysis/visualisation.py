import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import Optional, List

class StylometricVisualizer:
    """
    A class to handle the visualization of stylometric features extracted from LLMs.
    Handles automatic data scaling and dimensionality reduction.
    """
    
    def __init__(self, data: pd.DataFrame, feature_cols: List[str]):
        """
        Initializes the visualizer with the dataset and scales the features.
        
        Args:
            data (pd.DataFrame): The full dataset containing features and metadata.
            feature_cols (List[str]): List of column names representing the numerical features.
        """
        self.data = data.copy()
        self.feature_cols = feature_cols
        
        # Standardize features (Mean=0, Variance=1) - Critical for PCA and t-SNE
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.data[self.feature_cols])

    def plot_pca_tsne(
        self, 
        hue_col: str, 
        style_col: Optional[str] = None, 
        perplexity: int = 30, 
        random_state: int = 42
    ) -> plt.Figure:
        """
        Generates side-by-side PCA and t-SNE plots of the stylometric features.
        
        Args:
            hue_col (str): The column used for color encoding (e.g., 'model').
            style_col (str, optional): The column used for marker styling (e.g., 'genre').
            perplexity (int): The perplexity hyperparameter for t-SNE.
            random_state (int): Seed for reproducibility.
            
        Returns:
            plt.Figure: The generated matplotlib figure.
        """
        # Compute PCA
        pca = PCA(n_components=2, random_state=random_state)
        X_pca = pca.fit_transform(self.X_scaled)
        
        # Compute t-SNE
        # Note: perplexity should be strictly less than the number of samples.
        tsne_perplexity = min(perplexity, max(1, len(self.data) - 1)) 
        tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=random_state)
        X_tsne = tsne.fit_transform(self.X_scaled)
        
        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        titles = ['PCA: Linear Stylometric Separation', 't-SNE: Non-linear Stylometric Clusters']
        
        for ax, X_red, title in zip(axes, [X_pca, X_tsne], titles):
            sns.scatterplot(
                x=X_red[:, 0], y=X_red[:, 1],
                hue=self.data[hue_col],
                style=self.data[style_col] if style_col else None,
                ax=ax,
                alpha=0.75,
                palette="tab10",
                s=60 # Marker size
            )
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            
            # Improve legend placement
            if ax.get_legend():
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
        plt.tight_layout()
        return fig
    def plot_aesthetic_landscape(self, hue_col: str, perplexity: int = 30, random_state: int = 42) -> plt.Figure:
        """
        Creates a topographical 'landscape' map of stylistic embeddings using KDE (Kernel Density Estimation).
        """
        # Compute t-SNE for the base map
        tsne_perplexity = min(perplexity, max(1, len(self.data) - 1)) 
        tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=random_state)
        X_tsne = tsne.fit_transform(self.X_scaled)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 1. Draw the "landscape" (topographical contours)
        sns.kdeplot(
            x=X_tsne[:, 0], y=X_tsne[:, 1],
            hue=self.data[hue_col],
            fill=True, alpha=0.4, levels=5, ax=ax, palette="tab10"
        )
        
        # 2. Overlay the actual texts as small dots
        sns.scatterplot(
            x=X_tsne[:, 0], y=X_tsne[:, 1],
            hue=self.data[hue_col],
            ax=ax, alpha=0.8, s=30, edgecolor="w", palette="tab10", legend=False
        )
        
        ax.set_title("Aesthetic Landscape: Topographical Map of Model Styles", fontsize=16)
        ax.set_xlabel("Stylometric Dimension 1")
        ax.set_ylabel("Stylometric Dimension 2")
        
        plt.tight_layout()
        return fig