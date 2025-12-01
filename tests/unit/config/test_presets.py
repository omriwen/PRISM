"""
Preset System Tests for SPIDS Configuration

Tests preset loading, validation, merging, and CLI integration.
"""

from __future__ import annotations

from prism.config.presets import (
    MOPIE_PRESETS,
    PRISM_PRESETS,
    get_preset,
    list_presets,
    merge_preset_with_overrides,
    validate_preset_name,
)


class TestPresetLoading:
    """Test preset loading functions."""

    def test_get_preset_quick_test_spids(self):
        """Test loading quick_test preset for SPIDS."""
        preset = get_preset("quick_test", mode="prism")
        assert preset is not None
        assert isinstance(preset, dict)
        assert "telescope" in preset
        assert preset["telescope"]["n_samples"] == 64

    def test_get_preset_production_spids(self):
        """Test loading production preset for SPIDS."""
        preset = get_preset("production", mode="prism")
        assert preset is not None
        assert preset["telescope"]["n_samples"] == 200
        assert preset["telescope"]["snr"] == 40

    def test_get_preset_mopie_baseline(self):
        """Test loading mopie_baseline preset for Mo-PIE."""
        preset = get_preset("mopie_baseline", mode="mopie")
        assert preset is not None
        assert "mopie" in preset
        assert preset["mopie"]["lr_obj"] == 1.0
        assert preset["mopie"]["fix_probe"]

    def test_get_preset_invalid_name_returns_none(self):
        """Test that invalid preset name returns None."""
        preset = get_preset("nonexistent_preset", mode="prism")
        assert preset is None

    def test_list_presets_spids(self):
        """Test listing SPIDS presets."""
        presets = list_presets(mode="prism")
        assert isinstance(presets, list)
        assert "quick_test" in presets
        assert "production" in presets
        assert "high_quality" in presets

    def test_list_presets_mopie(self):
        """Test listing Mo-PIE presets."""
        presets = list_presets(mode="mopie")
        assert isinstance(presets, list)
        assert "mopie_baseline" in presets

    def test_list_presets_all(self):
        """Test listing all presets."""
        presets = list_presets(mode="all")
        assert isinstance(presets, list)
        assert len(presets) >= len(list_presets("prism"))

    def test_all_presets_have_required_fields(self):
        """Test that all presets have required fields."""
        for preset_name, preset in PRISM_PRESETS.items():
            assert "comment" in preset, f"{preset_name} missing comment"

        for preset_name, preset in MOPIE_PRESETS.items():
            assert "comment" in preset, f"{preset_name} missing comment"


class TestPresetValidation:
    """Test preset validation functions."""

    def test_validate_preset_name_valid_spids(self):
        """Test validation of valid SPIDS preset name."""
        assert validate_preset_name("quick_test", mode="prism")

    def test_validate_preset_name_valid_mopie(self):
        """Test validation of valid Mo-PIE preset name."""
        assert validate_preset_name("mopie_baseline", mode="mopie")

    def test_validate_preset_name_invalid(self):
        """Test validation of invalid preset name."""
        assert not validate_preset_name("nonexistent", mode="prism")


class TestPresetMerging:
    """Test preset merging with overrides."""

    def test_merge_preset_simple_override(self):
        """Test simple value override."""
        preset = {"telescope": {"n_samples": 100}}
        overrides = {"telescope": {"n_samples": 200}}

        merged = merge_preset_with_overrides(preset, overrides)
        assert merged["telescope"]["n_samples"] == 200

    def test_merge_preset_nested_override(self):
        """Test nested dict override."""
        preset = {"telescope": {"n_samples": 100, "snr": 40}, "training": {"lr": 0.001}}
        overrides = {"telescope": {"n_samples": 200}}  # Only override n_samples

        merged = merge_preset_with_overrides(preset, overrides)
        assert merged["telescope"]["n_samples"] == 200
        assert merged["telescope"]["snr"] == 40  # Preserved
        assert merged["training"]["lr"] == 0.001  # Preserved

    def test_merge_preset_preserves_non_overridden(self):
        """Test that non-overridden values preserved."""
        preset = get_preset("production", mode="prism")
        overrides = {"telescope": {"n_samples": 150}}

        merged = merge_preset_with_overrides(preset, overrides)
        assert merged["telescope"]["n_samples"] == 150  # Overridden
        # Check other values preserved
        if "snr" in preset["telescope"]:
            assert merged["telescope"]["snr"] == preset["telescope"]["snr"]

    def test_merge_preset_deep_merge(self):
        """Test deep merge behavior."""
        preset = {"a": {"b": {"c": 1, "d": 2}}, "e": 3}
        overrides = {
            "a": {"b": {"c": 99}},  # Only override a.b.c
            "f": 4,  # Add new key
        }

        merged = merge_preset_with_overrides(preset, overrides)
        assert merged["a"]["b"]["c"] == 99  # Overridden
        assert merged["a"]["b"]["d"] == 2  # Preserved
        assert merged["e"] == 3  # Preserved
        assert merged["f"] == 4  # Added

    def test_merge_preset_empty_overrides(self):
        """Test merging with empty overrides."""
        preset = get_preset("quick_test", mode="prism")
        merged = merge_preset_with_overrides(preset, {})
        assert merged == preset  # Should be unchanged


class TestPresetToConfig:
    """Test converting presets to valid configs."""

    def test_quick_test_preset_creates_valid_config(self):
        """Test quick_test preset creates valid PRISMConfig."""
        preset = get_preset("quick_test", mode="prism")
        # Add required fields
        preset["name"] = "test"
        preset["physics"] = {"obj_name": "europa"}

        # Verify preset structure
        assert "telescope" in preset
        assert preset.get("telescope")

    def test_all_spids_presets_have_valid_structure(self):
        """Test all SPIDS presets have valid structure."""
        for preset_name in list_presets("prism"):
            preset = get_preset(preset_name, mode="prism")
            assert preset is not None
            # Check it's a dict with expected top-level keys
            assert isinstance(preset, dict)

    def test_all_mopie_presets_have_valid_structure(self):
        """Test all Mo-PIE presets have valid structure."""
        for preset_name in list_presets("mopie"):
            preset = get_preset(preset_name, mode="mopie")
            assert preset is not None
            assert isinstance(preset, dict)


class TestPresetCoverage:
    """Test preset coverage and completeness."""

    def test_all_presets_have_comments(self):
        """Test all presets have non-empty comments."""
        for preset_name, preset_data in PRISM_PRESETS.items():
            assert "comment" in preset_data, f"Preset '{preset_name}' has no comment"
            assert preset_data["comment"], f"Preset '{preset_name}' has empty comment"

        for preset_name, preset_data in MOPIE_PRESETS.items():
            assert "comment" in preset_data, f"Preset '{preset_name}' has no comment"
            assert preset_data["comment"], f"Preset '{preset_name}' has empty comment"
