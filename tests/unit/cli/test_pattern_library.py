"""Tests for pattern library and metadata registry."""

from __future__ import annotations

import pytest

from prism.core.pattern_library import PatternInfo, PatternLibrary


def test_pattern_info_dataclass():
    """Test PatternInfo dataclass initialization."""
    info = PatternInfo(
        name="Test Pattern",
        spec="builtin:test",
        description="A test pattern",
        properties=["uniform", "incoherent"],
        parameters=["n_samples", "roi_diameter"],
        reference="Test Reference",
        recommended=True,
    )

    assert info.name == "Test Pattern"
    assert info.spec == "builtin:test"
    assert info.description == "A test pattern"
    assert "uniform" in info.properties
    assert "incoherent" in info.properties
    assert "n_samples" in info.parameters
    assert info.reference == "Test Reference"
    assert info.recommended is True


def test_pattern_info_default_values():
    """Test PatternInfo default values."""
    info = PatternInfo(
        name="Test",
        spec="builtin:test",
        description="Test",
        properties=[],
        parameters=[],
    )

    assert info.reference == ""
    assert info.recommended is False


def test_list_patterns():
    """Test listing all available patterns."""
    patterns = PatternLibrary.list_patterns()

    assert isinstance(patterns, list)
    assert len(patterns) > 0
    assert all(isinstance(p, PatternInfo) for p in patterns)

    # Check that builtin patterns are present
    pattern_names = [p.name for p in patterns]
    assert "Fermat Spiral" in pattern_names
    assert "Star Pattern" in pattern_names
    assert "Random Uniform" in pattern_names


def test_get_pattern_info():
    """Test getting specific pattern information."""
    # Test valid pattern
    fermat_info = PatternLibrary.get_pattern_info("fermat")
    assert isinstance(fermat_info, PatternInfo)
    assert fermat_info.name == "Fermat Spiral"
    assert fermat_info.spec == "builtin:fermat"
    assert fermat_info.recommended is True
    assert "optimal" in fermat_info.properties

    # Test another valid pattern
    star_info = PatternLibrary.get_pattern_info("star")
    assert star_info.name == "Star Pattern"
    assert star_info.spec == "builtin:star"
    assert "radial" in star_info.properties


def test_get_pattern_info_invalid():
    """Test getting invalid pattern raises KeyError."""
    with pytest.raises(KeyError, match="Pattern 'nonexistent' not found"):
        PatternLibrary.get_pattern_info("nonexistent")


def test_get_recommended_patterns():
    """Test getting recommended patterns."""
    recommended = PatternLibrary.get_recommended_patterns()

    assert isinstance(recommended, list)
    assert len(recommended) > 0
    assert all(p.recommended for p in recommended)

    # Fermat should be recommended
    recommended_names = [p.name for p in recommended]
    assert "Fermat Spiral" in recommended_names


def test_search_patterns_by_name():
    """Test searching patterns by name."""
    results = PatternLibrary.search_patterns("Fermat")
    assert len(results) > 0
    assert any(p.name == "Fermat Spiral" for p in results)


def test_search_patterns_by_description():
    """Test searching patterns by description."""
    results = PatternLibrary.search_patterns("optimal")
    assert len(results) > 0
    # Fermat spiral mentions optimal coverage
    assert any("Fermat" in p.name for p in results)


def test_search_patterns_by_property():
    """Test searching patterns by property."""
    results = PatternLibrary.search_patterns("incoherent")
    assert len(results) > 0
    # Multiple patterns should have incoherent property
    assert any("Fermat" in p.name for p in results)
    assert any("Random" in p.name for p in results)


def test_search_patterns_case_insensitive():
    """Test that search is case-insensitive."""
    results_lower = PatternLibrary.search_patterns("fermat")
    results_upper = PatternLibrary.search_patterns("FERMAT")
    results_mixed = PatternLibrary.search_patterns("FeRmAt")

    assert len(results_lower) == len(results_upper)
    assert len(results_lower) == len(results_mixed)


def test_search_patterns_no_results():
    """Test searching with no matches returns empty list."""
    results = PatternLibrary.search_patterns("nonexistent_pattern_xyz")
    assert results == []


def test_get_pattern_names():
    """Test getting list of pattern names."""
    names = PatternLibrary.get_pattern_names()

    assert isinstance(names, list)
    assert "fermat" in names
    assert "star" in names
    assert "random" in names
    assert len(names) == len(PatternLibrary.PATTERNS)


def test_get_pattern_by_property_uniform():
    """Test getting patterns by 'uniform' property."""
    uniform_patterns = PatternLibrary.get_pattern_by_property("uniform")

    assert len(uniform_patterns) > 0
    assert all("uniform" in [p.lower() for p in pattern.properties] for pattern in uniform_patterns)


def test_get_pattern_by_property_radial():
    """Test getting patterns by 'radial' property."""
    radial_patterns = PatternLibrary.get_pattern_by_property("radial")

    assert len(radial_patterns) > 0
    # Star pattern should be radial
    assert any(p.name == "Star Pattern" for p in radial_patterns)


def test_get_pattern_by_property_case_insensitive():
    """Test that property filtering is case-insensitive."""
    results_lower = PatternLibrary.get_pattern_by_property("uniform")
    results_upper = PatternLibrary.get_pattern_by_property("UNIFORM")
    results_mixed = PatternLibrary.get_pattern_by_property("UnIfOrM")

    assert len(results_lower) == len(results_upper)
    assert len(results_lower) == len(results_mixed)


def test_get_pattern_by_property_no_matches():
    """Test getting patterns with non-existent property returns empty list."""
    results = PatternLibrary.get_pattern_by_property("nonexistent_property")
    assert results == []


def test_pattern_metadata_completeness():
    """Test that all patterns have complete metadata."""
    patterns = PatternLibrary.list_patterns()

    for pattern in patterns:
        # All patterns should have required fields
        assert pattern.name
        assert pattern.spec
        assert pattern.description
        assert isinstance(pattern.properties, list)
        assert isinstance(pattern.parameters, list)
        assert isinstance(pattern.reference, str)
        assert isinstance(pattern.recommended, bool)


def test_fermat_pattern_metadata():
    """Test Fermat spiral pattern metadata completeness."""
    fermat = PatternLibrary.get_pattern_info("fermat")

    assert fermat.name == "Fermat Spiral"
    assert fermat.spec == "builtin:fermat"
    assert "optimal" in fermat.properties
    assert "spiral" in fermat.properties
    assert "n_samples" in fermat.parameters
    assert "roi_diameter" in fermat.parameters
    assert "Vogel" in fermat.reference
    assert fermat.recommended is True


def test_star_pattern_metadata():
    """Test star pattern metadata completeness."""
    star = PatternLibrary.get_pattern_info("star")

    assert star.name == "Star Pattern"
    assert star.spec == "builtin:star"
    assert "radial" in star.properties
    assert "n_angs" in star.parameters
    assert star.recommended is False


def test_random_pattern_metadata():
    """Test random pattern metadata completeness."""
    random = PatternLibrary.get_pattern_info("random")

    assert random.name == "Random Uniform"
    assert random.spec == "builtin:random"
    assert "random" in random.properties
    assert "incoherent" in random.properties
    assert "n_samples" in random.parameters
    assert random.recommended is False


def test_pattern_spec_format():
    """Test that all pattern specs follow correct format."""
    patterns = PatternLibrary.list_patterns()

    for pattern in patterns:
        # All builtin patterns should have 'builtin:' prefix
        assert pattern.spec.startswith("builtin:")
        # Extract pattern key
        key = pattern.spec.split(":", 1)[1]
        # Key should be in the PATTERNS dict
        assert key in PatternLibrary.PATTERNS


def test_pattern_registry_consistency():
    """Test consistency between PATTERNS dict and list_patterns."""
    patterns_list = PatternLibrary.list_patterns()
    patterns_dict = PatternLibrary.PATTERNS

    assert len(patterns_list) == len(patterns_dict)

    for pattern in patterns_list:
        # Extract key from spec
        key = pattern.spec.split(":", 1)[1]
        # Should exist in dict
        assert key in patterns_dict
        # Should be the same object
        assert patterns_dict[key] == pattern
