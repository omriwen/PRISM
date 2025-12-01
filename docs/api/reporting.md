# prism.reporting

PRISM Reporting Module
======================

This module provides functionality for generating comprehensive reports from experiment results.

Features:
    - HTML report generation
    - PDF report generation (via WeasyPrint)
    - Multi-experiment comparison reports
    - Customizable templates
    - Embedded visualizations

Usage:
    from prism.reporting import ReportGenerator

    generator = ReportGenerator()
    generator.generate_html(['runs/exp1'], 'report.html')
    generator.generate_pdf(['runs/exp1'], 'report.pdf')

## Classes
