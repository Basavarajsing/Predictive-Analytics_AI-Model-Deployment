# Exploratory Data Analysis

- Source workbook: `online_retail_II.xlsx`
- Rows after sheet-overlap reconciliation: 1,044,848
- Date coverage: 2009-12-01 through 2011-12-09
- Countries: 43; unique products: 5,305
- Missing customer IDs: 235,287 (22.519%)
- Missing descriptions: 4,275 (0.409%)
- Cancellation-marked rows: 19,165; negative quantity rows: 22,557
- Adjustment rows excluded from sales target: 6
- Net line sales (quantity × price, excluding A-prefixed adjustments): 19,057,376.20

Daily totals include recorded returns/cancellations. Calendar dates without transaction rows are represented as zero recorded net sales. Exact repeated rows within an individual sheet are retained because this workbook has no line identifier to distinguish a repeated legitimate line from a duplicate; cross-sheet overlap is reconciled by preserving the largest exact-row multiplicity found in any sheet.
