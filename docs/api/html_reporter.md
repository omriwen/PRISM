# prism.reporting.html_reporter

HTML Report Generator
====================

Generate professional HTML reports for PRISM experiments with embedded visualizations.

## Classes

### HTMLReporter

```python
HTMLReporter(template_path: Optional[pathlib.Path] = None)
```

Generate HTML reports with embedded visualizations.

Parameters
----------
template_path : str or Path, optional
    Path to Jinja2 template file. If not provided, uses default template.

Attributes
----------
template : jinja2.Template
    Loaded Jinja2 template for rendering

Methods
-------
generate_report(experiment_data)
    Generate complete HTML report from experiment data

Examples
--------
>>> reporter = HTMLReporter()
>>> html = reporter.generate_report(experiment_data)
>>> with open('report.html', 'w') as f:
...     f.write(html)

#### Methods

##### `__init__`

Initialize HTML reporter.

Parameters
----------
template_path : str or Path, optional
    Path to custom Jinja2 template

##### `generate_report`

Generate complete HTML report from experiment data.

Parameters
----------
experiment_data : dict
    Experiment data containing:
    - name: Experiment name
    - figures: Dict of matplotlib figures
    - metrics: Dict of final metrics
    - config: Dict of configuration parameters
    - statistics: Dict of statistical summaries

Returns
-------
str
    Complete HTML report as string

Examples
--------
>>> reporter = HTMLReporter()
>>> data = {
...     'name': 'europa_test',
...     'figures': {'reconstruction': fig},
...     'metrics': {'final_ssim': 0.95},
...     'config': {'n_samples': 100}
... }
>>> html = reporter.generate_report(data)

##### `load_template`

Load Jinja2 template from file.

Parameters
----------
template_path : Path
    Path to template file

Returns
-------
Template
    Loaded Jinja2 template

##### `save_report`

Generate and save HTML report to file.

Parameters
----------
experiment_data : dict
    Experiment data for report
output_path : Path
    Output file path for HTML report
