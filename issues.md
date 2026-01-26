# Quarto Conversion Issues

This document tracks warnings and issues from the DocOnce to Quarto conversion that need to be addressed.

## Build Status

- **PDF**: Renders successfully (680 pages, 6.2MB)
- **HTML**: Renders successfully

## Unresolved Cross-References

These cross-references from the original DocOnce source were not properly converted to Quarto format. They appear as `@sec-`, `@eq-`, or `@tbl-` references that don't resolve to existing labels.

### Truncation Appendix (trunc)

#### Section References

- `@sec-trunc-vib-gen-2x2model-ode-u`
- `@sec-trunc-vib-gen-2x2model-ode-v`
- `@sec-trunc-vib-gen-2x2model-ode-u-bw`
- `@sec-trunc-vib-gen-2x2model-ode-v-fw`
- `@sec-trunc-vib-gen-2x2model-ode-u-fw-R`
- `@sec-trunc-vib-gen-2x2model-ode-v-bw-R`
- `@sec-trunc-vib-gen-2x2model-ode-u-fw-R2`
- `@sec-trunc-vib-gen-2x2model-ode-v-bw-R2`

#### Table References

- `@tbl-trunc-table-fd1-fw-eq` / `@tbl-trunc-table-fd1-fw`
- `@tbl-trunc-table-fd1-bw-eq` / `@tbl-trunc-table-fd1-bw`
- `@tbl-trunc-table-fd1-bw2-eq` / `@tbl-trunc-table-fd1-bw2`
- `@tbl-trunc-table-fd1-center-eq` / `@tbl-trunc-table-fd1-center`
- `@tbl-trunc-table-fd1-center2-eq` / `@tbl-trunc-table-fd1-center2`
- `@tbl-trunc-table-fd2-center-eq` / `@tbl-trunc-table-fd2-center`
- `@tbl-trunc-table-avg-arith-eq` / `@tbl-trunc-table-avg-arith`
- `@tbl-trunc-table-avg-geom-eq` / `@tbl-trunc-table-avg-geom`
- `@tbl-trunc-table-avg-theta-eq` / `@tbl-trunc-table-avg-theta`

#### Equation References

- `@eq-trunc-wave-1D-varcoeff`
- `@eq-trunc-decay-estimate-R`
- `@eq-trunc-decay-corr`
- `@eq-trunc-vib-undamped`

### Wave Chapter

#### Equation References

- `@eq-wave-pde2-Neumann`
- `@eq-wave-pde2-var-c`
- `@eq-wave-pde2-software-ueq2`
- `@eq-wave-pde2-software-bcL2`
- `@eq-wave-pde2-fd-standing-waves`
- `@eq-wave-pde1-analysis`

#### Section References

- `@sec-wave-pde1-impl`
- `@sec-wave-pde1-impl-vec`
- `@sec-wave-pde1-verify`
- `@sec-wave-pde1-impl-verify-rate`
- `@sec-wave-2D3D-impl`
- `@sec-wave2D3D-impl-scalar`
- `@sec-wave2D3D-impl-vectorized`
- `@sec-wave-2D3D-impl1-2Du0-ueq-discrete`

### Software Engineering Appendix (softeng2)

#### Equation References

- `@eq-softeng2-wave1D-filestorage`
- `@eq-softeng2-wave1D-filestorage-joblib`
- `@eq-softeng2-wave1D-filestorage-savez`

#### Section References

- `@sec-softeng2-exer-savez`
- `@sec-softeng2-exer-pulse1D-C`

## Missing Citations

The following citations are referenced but not found in `references.bib`:

- `CODE`
- `CODE-8`
- `Grief_Ascher_2011`

## Technical Fixes Applied

### LaTeX Macro Conflicts

The following LaTeX macros conflicted with built-in LaTeX commands and were renamed:

| Original | Renamed | Reason |
|----------|---------|--------|
| `\u` | `\uu` | Conflicts with LaTeX breve accent `\u{}` |
| `\v` | `\vv` | Conflicts with LaTeX caron accent `\v{}` |

Files affected:

- `chapters/wave/wave_app.qmd`
- `chapters/diffu/diffu_app.qmd`
- `chapters/advec/advec.qmd`

### Raw LaTeX Blocks

The `alignat` environment required wrapping in raw LaTeX blocks to prevent Pandoc from escaping special characters:

~~~~markdown
```{=latex}
\begin{alignat}{2}
...
\end{alignat}
```
~~~~

Files with alignat environments:

- `chapters/nonlin/nonlin_pde1D.qmd`
- `chapters/wave/wave2D_prog.qmd`
- `chapters/wave/wave1D_fd2.qmd`
- `chapters/diffu/diffu_fd1.qmd`
- `chapters/diffu/diffu_fd2.qmd`
- `chapters/diffu/diffu_exer.qmd`

## How to Fix Cross-References

1. **Find the original label**: Search the `.qmd` files for the section/equation/table that should have the label
2. **Add Quarto label**: Add the appropriate label syntax:
   - Sections: `{#sec-label-name}` after the heading
   - Equations: `{#eq-label-name}` after `$$`
   - Tables: Add `label` attribute to table div
3. **Verify**: Run `quarto render` and check warnings

## How to Fix Citations

1. Add missing entries to `references.bib`
2. Or remove/update the citation references in the source files

## Priority

1. **High**: Missing citations (easy to fix, improves credibility)
2. **Medium**: Section cross-references (affects navigation)
3. **Low**: Table/equation references (cosmetic, shows as "??" in text)
