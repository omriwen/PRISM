"""
Module: spids.visualization.style.colormaps
Purpose: Astronomy-specific colormaps for scientific visualization
Dependencies: matplotlib, numpy

Description:
    Custom colormaps designed for astronomical imaging:
    - prism_cosmic_gray: Enhanced grayscale with better shadow detail
    - prism_europa_ice: Blue-white tones for icy moon surfaces
    - prism_betelgeuse_fire: Red-orange for stellar surfaces
    - prism_kspace_power: Purple-coral for k-space magnitude display
"""

from __future__ import annotations

from typing import Any

from matplotlib.colors import Colormap, LinearSegmentedColormap


# Custom colormap definitions
ASTRONOMY_COLORMAPS: dict[str, dict[str, Any]] = {
    "prism_cosmic_gray": {
        "colors": [
            (0.00, "#000000"),  # Pure black for deep space
            (0.10, "#1A1A2E"),  # Dark blue-gray (star shadows)
            (0.30, "#3A3A4A"),  # Medium-dark gray
            (0.50, "#6A6A7A"),  # Medium gray
            (0.70, "#9A9AAA"),  # Light gray
            (0.85, "#C8C8D8"),  # Near white
            (1.00, "#FFFFFF"),  # Pure white (bright sources)
        ],
        "description": "Enhanced grayscale optimized for astronomical images",
    },
    "prism_europa_ice": {
        "colors": [
            (0.00, "#0A1628"),  # Deep shadow
            (0.20, "#1E3A5F"),  # Ice shadow blue
            (0.40, "#4A7A9F"),  # Ice mid-tone
            (0.60, "#7AAACF"),  # Ice highlight
            (0.80, "#B0D4EF"),  # Bright ice
            (1.00, "#E8F4FC"),  # Pure ice white
        ],
        "description": "Europa-style icy surface tones",
    },
    "prism_betelgeuse_fire": {
        "colors": [
            (0.00, "#1A0A00"),  # Deep shadow
            (0.20, "#4A1800"),  # Dark red
            (0.40, "#8A3500"),  # Orange-red
            (0.60, "#C85A00"),  # Bright orange
            (0.80, "#FF8C00"),  # Hot orange
            (1.00, "#FFD700"),  # Bright yellow (hot spots)
        ],
        "description": "Betelgeuse-style red supergiant surface",
    },
    "prism_kspace_power": {
        "colors": [
            (0.00, "#000004"),  # Near black (low power)
            (0.25, "#3B0F6F"),  # Deep purple
            (0.50, "#8C2981"),  # Magenta
            (0.75, "#DE4968"),  # Coral
            (1.00, "#FEC287"),  # Bright yellow (high power)
        ],
        "description": "K-space magnitude optimized for log scale display",
    },
    "prism_phase": {
        "colors": [
            (0.00, "#313695"),  # Blue (negative phase)
            (0.25, "#74ADD1"),  # Light blue
            (0.50, "#FFFFBF"),  # Yellow (zero)
            (0.75, "#F46D43"),  # Orange
            (1.00, "#A50026"),  # Red (positive phase)
        ],
        "description": "Diverging colormap for phase visualization",
    },
}


def _create_colormap(name: str, color_data: dict[str, Any]) -> LinearSegmentedColormap:
    """Create matplotlib colormap from color data.

    Parameters
    ----------
    name : str
        Colormap name
    color_data : dict
        Dictionary with 'colors' key containing list of (position, hex) tuples

    Returns
    -------
    LinearSegmentedColormap
        Matplotlib colormap
    """
    colors = color_data["colors"]
    positions = [c[0] for c in colors]
    hex_colors = [c[1] for c in colors]

    # Convert hex to RGB
    rgb_colors = []
    for hex_color in hex_colors:
        hex_color = hex_color.lstrip("#")
        rgb = tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
        rgb_colors.append(rgb)

    # Create colormap
    cmap = LinearSegmentedColormap.from_list(name, list(zip(positions, rgb_colors)))
    return cmap


def register_astronomy_colormaps() -> None:
    """Register all astronomy colormaps with matplotlib.

    This function is called automatically when the style module is imported.
    After registration, colormaps can be used via:
        plt.imshow(data, cmap='prism_cosmic_gray')
    """
    import matplotlib.pyplot as plt

    for name, color_data in ASTRONOMY_COLORMAPS.items():
        try:
            # Check if already registered
            plt.colormaps.get_cmap(name)
        except ValueError:
            # Not registered, create and register
            cmap = _create_colormap(name, color_data)
            plt.colormaps.register(cmap=cmap, name=name)


def get_colormap(name: str) -> Colormap:
    """Get colormap by name.

    Parameters
    ----------
    name : str
        Colormap name (with or without 'spids_' prefix)

    Returns
    -------
    Colormap
        Matplotlib colormap

    Raises
    ------
    ValueError
        If colormap not found
    """
    import matplotlib.pyplot as plt

    # Try with spids_ prefix
    if not name.startswith("prism_"):
        prefixed_name = f"spids_{name}"
        if prefixed_name in ASTRONOMY_COLORMAPS:
            name = prefixed_name

    return plt.colormaps.get_cmap(name)


def list_astronomy_colormaps() -> list[str]:
    """List all available astronomy colormaps.

    Returns
    -------
    list[str]
        List of colormap names
    """
    return list(ASTRONOMY_COLORMAPS.keys())
