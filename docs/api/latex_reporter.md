# prism.reporting.latex_reporter

LaTeX/PDF Report Generator
==========================

Generate publication-quality LaTeX/PDF reports for PRISM experiments.

## Classes

### LaTeXReporter

```python
LaTeXReporter(template_path: Optional[pathlib.Path] = None)
```

Generate LaTeX/PDF reports for PRISM experiments.

Parameters
----------
template_path : str or Path, optional
    Path to LaTeX template file. If not provided, uses default template.

Attributes
----------
template : str
    LaTeX template content

Methods
-------
generate_report(experiment_data, output_path)
    Generate PDF report from experiment data
create_latex(experiment_data)
    Create LaTeX source code

Examples
--------
>>> reporter = LaTeXReporter()
>>> reporter.generate_report(experiment_data, 'report.pdf')

Notes
-----
Requires pdflatex to be installed for PDF compilation.

#### Methods

##### `__init__`

Initialize LaTeX reporter.

Parameters
----------
template_path : str or Path, optional
    Path to custom LaTeX template

##### `compile_pdf`

Compile LaTeX file to PDF using pdflatex.

Parameters
----------
tex_path : Path
    Path to .tex file
output_dir : Path, optional
    Output directory for PDF (defaults to same as .tex file)

Returns
-------
bool
    True if compilation successful, False otherwise

##### `create_latex`

Create LaTeX source code from experiment data.

Parameters
----------
experiment_data : dict
    Experiment data containing:
    - name: Experiment name
    - figures: Dict of matplotlib figures
    - metrics: Dict of final metrics
    - config: Dict of configuration parameters
    - statistics: Dict of statistical summaries
figures_dir : Path, optional
    Directory where figures are saved

Returns
-------
str
    LaTeX source code

##### `generate_report`

Generate complete PDF report from experiment data.

Parameters
----------
experiment_data : dict
    Experiment data for report
output_path : Path
    Output path for PDF file

Returns
-------
bool
    True if successful, False otherwise

Examples
--------
>>> reporter = LaTeXReporter()
>>> success = reporter.generate_report(data, 'report.pdf')
