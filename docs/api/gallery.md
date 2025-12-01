# prism.cli.patterns.gallery

HTML gallery generation for pattern visualization.

## Classes

## Functions

### generate_gallery_html

```python
generate_gallery_html(pattern_images: List[Dict[str, Any]]) -> str
```

Generate HTML gallery from pattern images.

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
