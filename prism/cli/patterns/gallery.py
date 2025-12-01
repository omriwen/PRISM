"""HTML gallery generation for pattern visualization."""

from __future__ import annotations

from typing import Any, Dict, List


def generate_gallery_html(pattern_images: List[Dict[str, Any]]) -> str:
    """Generate HTML gallery from pattern images.

    Parameters
    ----------
    pattern_images : List[Dict[str, Any]]
        List of pattern data dictionaries with keys:
        - info: PatternInfo object
        - image: base64-encoded image string
        - stats: statistics dictionary

    Returns
    -------
    str
        Complete HTML document as string
    """
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SPIDS Pattern Library</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }
            .intro {
                background: white;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 30px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .pattern-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
                gap: 25px;
            }
            .pattern-card {
                background: white;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                transition: transform 0.2s, box-shadow 0.2s;
            }
            .pattern-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }
            .pattern-card.recommended {
                border: 2px solid #27ae60;
                background: #f0fff4;
            }
            .pattern-image {
                width: 100%;
                border-radius: 4px;
                margin-bottom: 15px;
            }
            .pattern-name {
                font-size: 1.4em;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 10px;
            }
            .recommended-badge {
                display: inline-block;
                background: #27ae60;
                color: white;
                padding: 3px 10px;
                border-radius: 12px;
                font-size: 0.8em;
                margin-left: 10px;
            }
            .pattern-description {
                color: #555;
                line-height: 1.6;
                margin-bottom: 15px;
            }
            .pattern-properties {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-bottom: 15px;
            }
            .property-tag {
                background: #3498db;
                color: white;
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 0.85em;
            }
            .pattern-stats {
                background: #f8f9fa;
                padding: 12px;
                border-radius: 4px;
                font-size: 0.9em;
                font-family: monospace;
            }
            .stat-row {
                display: flex;
                justify-content: space-between;
                margin-bottom: 5px;
            }
            .stat-label {
                color: #666;
            }
            .stat-value {
                font-weight: bold;
                color: #2c3e50;
            }
            .reference {
                margin-top: 15px;
                padding-top: 15px;
                border-top: 1px solid #eee;
                font-size: 0.85em;
                color: #777;
                font-style: italic;
            }
        </style>
    </head>
    <body>
        <h1>SPIDS Pattern Library</h1>
        <div class="intro">
            <p>
                SPIDS supports multiple k-space sampling patterns for sparse aperture synthesis.
                Each pattern has different properties and is suited for different applications.
                Patterns marked with <span class="recommended-badge">RECOMMENDED</span> are
                suggested for general use.
            </p>
        </div>
        <div class="pattern-grid">
    """

    for pattern_data in pattern_images:
        info = pattern_data["info"]
        stats = pattern_data["stats"]
        img = pattern_data["image"]

        card_class = "pattern-card recommended" if info.recommended else "pattern-card"
        recommended_badge = (
            '<span class="recommended-badge">RECOMMENDED</span>' if info.recommended else ""
        )
        reference_html = (
            f'<div class="reference">Reference: {info.reference}</div>' if info.reference else ""
        )
        properties_html = "".join(
            f'<span class="property-tag">{prop}</span>' for prop in info.properties
        )

        html += f"""
            <div class="{card_class}">
                <div class="pattern-name">
                    {info.name}
                    {recommended_badge}
                </div>
                <img src="data:image/png;base64,{img}" class="pattern-image" alt="{info.name}">
                <div class="pattern-description">{info.description}</div>
                <div class="pattern-properties">
                    {properties_html}
                </div>
                <div class="pattern-stats">
                    <div class="stat-row">
                        <span class="stat-label">Samples:</span>
                        <span class="stat-value">{stats["n_samples"]}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Mean Radius:</span>
                        <span class="stat-value">{stats["radial_mean"]:.1f} px</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Coverage:</span>
                        <span class="stat-value">{stats["coverage_percentage"]:.1f}%</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Type:</span>
                        <span class="stat-value">{"Line" if stats["is_line_sampling"] else "Point"}</span>
                    </div>
                </div>
                {reference_html}
            </div>
        """

    html += """
        </div>
    </body>
    </html>
    """

    return html
