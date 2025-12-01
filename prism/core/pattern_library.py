"""
Pattern library and metadata registry.

Provides centralized metadata and information about available sampling patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class PatternInfo:
    """
    Metadata and information about a sampling pattern.

    Attributes
    ----------
    name : str
        Display name of the pattern
    spec : str
        Pattern specification string (e.g., "builtin:fermat")
    description : str
        Brief description of the pattern
    properties : List[str]
        Pattern properties (e.g., 'uniform', 'incoherent', 'radial')
    parameters : List[str]
        Required/optional configuration parameters
    reference : str
        Academic reference or citation (if applicable)
    recommended : bool
        Whether this is a recommended pattern for general use
    """

    name: str
    spec: str
    description: str
    properties: List[str]
    parameters: List[str]
    reference: str = ""
    recommended: bool = False


class PatternLibrary:
    """
    Central registry of sampling patterns with metadata.

    Provides information about built-in patterns and utilities for
    listing, searching, and comparing patterns.
    """

    # Pattern metadata registry
    PATTERNS: Dict[str, PatternInfo] = {
        "fermat": PatternInfo(
            name="Fermat Spiral",
            spec="builtin:fermat",
            description=(
                "Logarithmic spiral using golden angle (≈137.5°) for optimal "
                "k-space coverage. This is the recommended pattern for most "
                "SPIDS applications as it provides approximately uniform density "
                "with minimal aliasing artifacts."
            ),
            properties=["uniform", "incoherent", "optimal", "spiral"],
            parameters=[
                "n_samples",
                "roi_diameter",
                "sample_length",
                "line_angle",
                "samples_r_cutoff",
            ],
            reference=(
                "Vogel, H. (1979). A better way to construct the sunflower head. "
                "Mathematical Biosciences, 44(3-4), 179-189."
            ),
            recommended=True,
        ),
        "star": PatternInfo(
            name="Star Pattern",
            spec="builtin:star",
            description=(
                "Radial lines emanating from center at evenly-spaced angles. "
                "Lines are distributed at multiple radii for dense coverage. "
                "Useful for testing rotational symmetry in reconstructions."
            ),
            properties=["radial", "structured", "symmetric"],
            parameters=["n_angs", "sample_length", "roi_diameter"],
            reference="",
            recommended=False,
        ),
        "random": PatternInfo(
            name="Random Uniform",
            spec="builtin:random",
            description=(
                "Uniformly random sampling positions within circular or square "
                "region. Provides incoherent sampling but less optimal coverage "
                "than Fermat spiral. Useful for comparison and baseline testing."
            ),
            properties=["random", "incoherent", "uniform"],
            parameters=["n_samples", "sample_length", "roi_diameter"],
            reference="",
            recommended=False,
        ),
    }

    @classmethod
    def list_patterns(cls) -> List[PatternInfo]:
        """
        Get list of all available patterns.

        Returns
        -------
        List[PatternInfo]
            List of pattern information objects
        """
        return list(cls.PATTERNS.values())

    @classmethod
    def get_pattern_info(cls, pattern_name: str) -> PatternInfo:
        """
        Get information about a specific pattern.

        Parameters
        ----------
        pattern_name : str
            Name of the pattern (e.g., 'fermat', 'star', 'random')

        Returns
        -------
        PatternInfo
            Pattern information object

        Raises
        ------
        KeyError
            If pattern name is not found in registry
        """
        if pattern_name not in cls.PATTERNS:
            available = ", ".join(cls.PATTERNS.keys())
            raise KeyError(f"Pattern '{pattern_name}' not found. Available patterns: {available}")
        return cls.PATTERNS[pattern_name]

    @classmethod
    def get_recommended_patterns(cls) -> List[PatternInfo]:
        """
        Get list of recommended patterns for general use.

        Returns
        -------
        List[PatternInfo]
            List of recommended pattern information objects
        """
        return [p for p in cls.PATTERNS.values() if p.recommended]

    @classmethod
    def search_patterns(cls, query: str) -> List[PatternInfo]:
        """
        Search patterns by name, description, or properties.

        Parameters
        ----------
        query : str
            Search query string (case-insensitive)

        Returns
        -------
        List[PatternInfo]
            List of matching pattern information objects
        """
        query_lower = query.lower()
        results = []

        for pattern in cls.PATTERNS.values():
            # Search in name
            if query_lower in pattern.name.lower():
                results.append(pattern)
                continue

            # Search in description
            if query_lower in pattern.description.lower():
                results.append(pattern)
                continue

            # Search in properties
            if any(query_lower in prop.lower() for prop in pattern.properties):
                results.append(pattern)
                continue

        return results

    @classmethod
    def get_pattern_names(cls) -> List[str]:
        """
        Get list of pattern names (keys).

        Returns
        -------
        List[str]
            List of pattern names
        """
        return list(cls.PATTERNS.keys())

    @classmethod
    def get_pattern_by_property(cls, property: str) -> List[PatternInfo]:
        """
        Get patterns that have a specific property.

        Parameters
        ----------
        property : str
            Property to filter by (e.g., 'uniform', 'incoherent', 'radial')

        Returns
        -------
        List[PatternInfo]
            List of patterns with the specified property
        """
        return [
            p
            for p in cls.PATTERNS.values()
            if property.lower() in [prop.lower() for prop in p.properties]
        ]
