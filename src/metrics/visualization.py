# src/metrics/visualization.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from typing import Optional, Dict, Tuple


@torch.no_grad()
def plot_representation_space(
    model,
    retain_loader,
    forget_loader,
    test_loader=None,
    device: str = "cuda",
    save_path: Optional[Path] = None,
    title: str = "Representation Space",
    figsize: Tuple[int, int] = (10, 8),
    class_names: Optional[list] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> Dict[str, np.ndarray]:
    """
    Plot 2D representation space with retain, forget, and test samples.
    
    Args:
        model: Model with get_representation() method
        retain_loader: DataLoader for retain set
        forget_loader: DataLoader for forget set
        test_loader: DataLoader for test set (optional)
        device: Device to run on
        save_path: Path to save figure
        title: Plot title
        figsize: Figure size
        class_names: List of class names for legend
    
    Returns:
        Dictionary with representations and labels for each split
    """
    model.eval()
    device = torch.device(device)
    
    def extract_representations(loader, split_name):
        """Extract representations and labels from a loader."""
        reprs = []
        labels = []
        
        for x, y in loader:
            x = x.to(device)
            
            # Get representation
            if hasattr(model, 'get_representation'):
                repr = model.get_representation(x)
            elif hasattr(model, 'encoder'):
                repr = model.encoder(x)
            else:
                # Assume forward returns representation
                repr = model(x, return_repr=True)[1] if hasattr(model, 'forward') else model(x)
            
            reprs.append(repr.cpu().numpy())
            labels.append(y.cpu().numpy())
        
        reprs = np.vstack(reprs)
        labels = np.concatenate(labels)
        
        return reprs, labels
    
    # Extract representations
    retain_reprs, retain_labels = extract_representations(retain_loader, "retain")
    forget_reprs, forget_labels = extract_representations(forget_loader, "forget")
    
    results = {
        "retain_reprs": retain_reprs,
        "retain_labels": retain_labels,
        "forget_reprs": forget_reprs,
        "forget_labels": forget_labels,
    }
    
    if test_loader is not None:
        test_reprs, test_labels = extract_representations(test_loader, "test")
        results["test_reprs"] = test_reprs
        results["test_labels"] = test_labels
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color scheme
    n_classes = max(
        retain_labels.max() if len(retain_labels) > 0 else 0,
        forget_labels.max() if len(forget_labels) > 0 else 0,
        test_labels.max() if test_loader and len(test_labels) > 0 else 0,
    ) + 1
    
    # Identify forget classes
    forget_class_set = set(forget_labels.tolist())
    
    # Detect if this is class_forget (all forget samples from same classes) 
    # or random_forget (samples from multiple classes mixed)
    # Heuristic: if forget samples span many classes with small counts per class, it's likely random_forget
    unique_forget_classes = len(forget_class_set)
    is_class_forget = unique_forget_classes <= 2  # Assume class_forget if 1-2 classes in forget set
    
    if is_class_forget:
        # For class_forget: use bright distinct colors for forget classes, pastels for retain
        # Define pastel colors for retain classes (excluding yellow to avoid confusion)
        pastel_colors_list = [
            '#FF9999',  # More vibrant pastel red/pink
            '#99FF99',  # More vibrant pastel green
            '#99CCFF',  # More vibrant pastel blue
            '#FFCC99',  # More vibrant pastel orange
            '#CC99FF',  # More vibrant pastel purple
            '#99E6FF',  # More vibrant pastel cyan
            '#FF99CC',  # More vibrant pastel pink
            '#B399FF',  # More vibrant pastel lavender
            '#99FF99',  # More vibrant pastel mint
        ]
        
        # Bright/highlighted colors for forget classes
        highlight_colors_list = [
            '#FFFF00',  # Bright yellow
            '#FF00FF',  # Bright magenta
            '#00FFFF',  # Bright cyan
            '#FF6600',  # Bright orange
            '#00FF00',  # Bright green
            '#FF0099',  # Bright pink
            '#9900FF',  # Bright purple
        ]
        
        # Assign colors based on whether class is in forget set
        pastel_colors = {}
        bright_colors = {}
        pastel_idx = 0
        highlight_idx = 0
        
        for class_idx in range(n_classes):
            if class_idx in forget_class_set:
                # Forget class: use bright color
                color = highlight_colors_list[highlight_idx % len(highlight_colors_list)]
                pastel_colors[class_idx] = color
                bright_colors[class_idx] = color
                highlight_idx += 1
            else:
                # Retain class: use pastel color
                color = pastel_colors_list[pastel_idx % len(pastel_colors_list)]
                pastel_colors[class_idx] = color
                bright_colors[class_idx] = color
                pastel_idx += 1
    else:
        # For random_forget: use matching pastel/bright pairs per class
        color_pairs = [
            ('#FF9999', '#FF3333'),  # More vibrant pastel red/pink -> Bright red
            ('#99FF99', '#00FF00'),  # More vibrant pastel green -> Bright green
            ('#99CCFF', '#00CCFF'),  # More vibrant pastel blue -> Bright sky blue
            ('#FFCC99', '#FF6600'),  # More vibrant pastel orange -> Bright orange
            ('#CC99FF', '#9900FF'),  # More vibrant pastel purple -> Bright purple
            ('#99E6FF', '#00FFFF'),  # More vibrant pastel cyan -> Bright cyan
            ('#FF99CC', '#FF0099'),  # More vibrant pastel pink -> Bright pink
            ('#B399FF', '#6666FF'),  # More vibrant pastel lavender -> Bright lavender
            ('#99FFCC', '#00CC99'),  # More vibrant pastel mint -> Bright mint
            ('#FFDD99', '#FFCC00'),  # More vibrant pastel peach -> Bright gold
        ]
        
        # Assign pastel color to each class (for retain samples)
        pastel_colors = {}
        bright_colors = {}
        for class_idx in range(n_classes):
            pastel, bright = color_pairs[class_idx % len(color_pairs)]
            pastel_colors[class_idx] = pastel
            bright_colors[class_idx] = bright
    
    # Plot retain samples (circles with pastel colors)
    for class_idx in range(n_classes):
        mask = retain_labels == class_idx
        if mask.sum() > 0:
            label = f"Class {class_idx} (retain)" if class_names is None else f"{class_names[class_idx]} (retain)"
            ax.scatter(
                retain_reprs[mask, 0],
                retain_reprs[mask, 1],
                c=[pastel_colors[class_idx]],
                marker='o',
                s=100,
                alpha=0.7,
                edgecolors='none',
                label=label,
            )
    
    # Plot forget samples (stars with bright colors)
    for class_idx in range(n_classes):
        mask = forget_labels == class_idx
        if mask.sum() > 0:
            label = f"Class {class_idx} (forget)" if class_names is None else f"{class_names[class_idx]} (forget)"
            ax.scatter(
                forget_reprs[mask, 0],
                forget_reprs[mask, 1],
                c=[bright_colors[class_idx]],
                marker='*',
                s=400,
                alpha=0.95,
                edgecolors='none',
                label=label,
            )
    
    # Plot test samples if available (small dots)
    if test_loader is not None:
        for class_idx in range(n_classes):
            mask = test_labels == class_idx
            if mask.sum() > 0:
                label = f"Class {class_idx} (test)" if class_names is None else f"{class_names[class_idx]} (test)"
                ax.scatter(
                    test_reprs[mask, 0],
                    test_reprs[mask, 1],
                    c=[pastel_colors[class_idx]],
                    marker='.',
                    s=40,
                    alpha=0.4,
                    label=label,
                )
    
    # Remove axis labels, ticks, legend, and grid
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Set axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Get current axis limits to return
    current_xlim = ax.get_xlim()
    current_ylim = ax.get_ylim()
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Visualization] Saved plot to {save_path}")
    
    plt.close()
    
    # Add axis limits to results
    results['xlim'] = current_xlim
    results['ylim'] = current_ylim
    
    return results


def compare_representations(
    model_before,
    model_after,
    retain_loader,
    forget_loader,
    device: str = "cuda",
    save_path: Optional[Path] = None,
    class_names: Optional[list] = None,
) -> None:
    """
    Create side-by-side comparison of representation space before and after unlearning.
    
    Args:
        model_before: Model before unlearning
        model_after: Model after unlearning
        retain_loader: DataLoader for retain set
        forget_loader: DataLoader for forget set
        device: Device to run on
        save_path: Path to save figure
        class_names: List of class names for legend
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot before
    plt.sca(axes[0])
    plot_representation_space(
        model_before,
        retain_loader,
        forget_loader,
        device=device,
        title="Before Unlearning",
        class_names=class_names,
    )
    axes[0].legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize=9)
    
    # Plot after
    plt.sca(axes[1])
    plot_representation_space(
        model_after,
        retain_loader,
        forget_loader,
        device=device,
        title="After Unlearning",
        class_names=class_names,
    )
    axes[1].legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Visualization] Saved comparison plot to {save_path}")
    
    plt.close()
