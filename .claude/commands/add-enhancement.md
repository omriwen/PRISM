# Add Enhancement to FUTURE_ENHANCEMENTS.md

Add a new planned enhancement to the docs/FUTURE_ENHANCEMENTS.md file.

## Instructions

The user wants to add a new future enhancement: $ARGUMENTS

1. First, read the current contents of docs/FUTURE_ENHANCEMENTS.md to understand the existing structure and sections.

2. Determine which section the enhancement belongs to based on its description:
   - **Optical Physics**: Aberrations, propagation methods, atmospheric effects, wavelength-related
   - **Imaging Scenarios**: New scenario types (satellite, telescope, etc.), scenario configurations
   - **Test Targets**: Resolution targets, calibration patterns (Siemens star, dot grids, etc.)
   - **GPU & Performance**: CUDA, AMP, memory optimization, speed improvements
   - **User Experience**: CLI tools, error messages, wizards, dashboards, templates
   - **Advanced Networks**: Neural network architectures, complex-valued networks
   - **Code Architecture**: Package structure, refactoring, API changes

3. If the category is ambiguous, ask the user to clarify which section it belongs to.

4. Format the enhancement entry following this template:
   ```markdown
   ### [Enhancement Title]

   **Priority**: [High/Medium/Low]
   **Source**: [User Request / Plan Name]
   **Effort**: [Estimate if known, otherwise omit]

   [Brief description of what the enhancement does]

   [Optional: Use cases, example API, or implementation notes]

   ---
   ```

5. Add the new enhancement to the appropriate section in docs/FUTURE_ENHANCEMENTS.md, placing it after existing entries in that section.

6. Update the "Last Updated" date at the top of the file to today's date.

7. Show the user what was added and where.

## Example Usage

```
/add-enhancement Support for vignetting simulation in camera lenses
/add-enhancement Add a "dry run" mode that shows what would happen without executing
/add-enhancement Siemens star test target for MTF measurement
```
