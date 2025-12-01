# prism.reporting.generator

Report Generator Module
=======================

Generates comprehensive HTML and PDF reports from experiment results.

## Classes

### ReportGenerator

```python
ReportGenerator(template_dir: Optional[pathlib.Path] = None)
```

Generate HTML and PDF reports from experiment results.

#### Methods

##### `__init__`

Initialize report generator.

Parameters
----------
template_dir : Path, optional
    Directory containing Jinja2 templates. If None, uses default templates.

##### `generate_html`

Generate HTML report from experiments.

Parameters
----------
experiment_paths : List[Path]
    Paths to experiment directories
output_path : Path
    Output path for HTML report
template_name : str
    Template file name to use
include_appendix : bool
    Whether to include appendix section

##### `generate_pdf`

Generate PDF report from experiments.

Parameters
----------
experiment_paths : List[Path]
    Paths to experiment directories
output_path : Path
    Output path for PDF report
include_appendix : bool
    Whether to include appendix section
