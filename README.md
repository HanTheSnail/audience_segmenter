# Audience Segmentation Tool

A flexible Streamlit application for segmenting market research audiences with nested quotas and constraints.

## Features

- **Flexible Sample Sizing**: Split audiences into prompted (product testing) and unprompted (retailer-only) segments
- **Nested Quotas**: Support for up to 3 levels of nested segmentation (e.g., Product → Cooking Method → Pack Size)
- **Constraint Handling**: Respect participant preferences (e.g., ASDA shoppers only go to ASDA)
- **Dynamic Configuration**: Each project can have different columns and requirements
- **Balanced Distribution**: Automatically balances "Either" preferences across options

## Quick Start

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run audience_segmenter.py
```

3. Open your browser to `http://localhost:8501`

### Deploy to Streamlit Cloud (GitHub)

1. Push this repository to GitHub

2. Go to [share.streamlit.io](https://share.streamlit.io)

3. Click "New app"

4. Connect your GitHub repository

5. Set the main file path to: `audience_segmenter.py`

6. Click "Deploy"

### Deploy to Streamlit Cloud (Direct)

Alternatively, you can deploy directly:

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Paste your GitHub repository URL
4. Select the branch (usually `main`)
5. Set main file path: `audience_segmenter.py`
6. Click "Deploy"

## Usage Guide

### Step 1: Upload CSV
Upload your audience CSV file containing participant information.

### Step 2: Set Sample Sizes
- **Prompted users**: Number of participants who will test products (full segmentation logic applies)
- **Unprompted users**: Number of participants for retailer-only tracking (no product assignment)

### Step 3: Select Retailer Column
Choose the column containing retailer preferences (e.g., "ASDA", "Morrisons", "Either")

### Step 4: Additional Constraints (Optional)
Add any additional constraint columns that should be respected for prompted users:
- Cooking Preference
- Flavour Preference
- Dietary Requirements
- etc.

### Step 5: Product Assignment (Optional)
Configure nested quotas for prompted users:

**Level 1**: Primary product/variant split
- Example: Chicken 50% / Vegetable 50%

**Level 2 (Optional)**: Nested within Level 1
- Example: Soup 50% / Casserole 50%
- This creates: Chicken+Soup, Chicken+Casserole, Vegetable+Soup, Vegetable+Casserole

**Level 3 (Optional)**: Further nested within Level 2
- Example: Large Pack 50% / Small Pack 50%

### Step 6: Run Segmentation
Click "Run Segmentation" to process your audience and download the results.

## Output Format

The tool adds the following columns to your original CSV:

- **Segment_Type**: "Prompted" or "Unprompted"
- **Assigned_Retailer**: The retailer assigned based on preference
- **[Level 1 Name]**: First level assignment (e.g., "Product_Variant")
- **[Level 2 Name]**: Second level assignment if configured (e.g., "Cooking_Method")
- **[Level 3 Name]**: Third level assignment if configured

## Example Scenarios

### Scenario 1: Stock Testing with Cooking Method
- 70 prompted, 30 unprompted
- Retailer + Cooking Preference constraints
- Level 1: Chicken 50% / Vegetable 50%
- Level 2: Soup 50% / Casserole 50%

### Scenario 2: Simple Flavour Test
- 80 prompted, 20 unprompted
- Retailer only constraint
- Level 1: Original 33% / BBQ 33% / Salt & Vinegar 34%
- No Level 2

### Scenario 3: Complex 3-Level Test
- 100 prompted, 0 unprompted
- Retailer + Dietary Preference constraints
- Level 1: Product A 40% / Product B 60%
- Level 2: Large 50% / Small 50%
- Level 3: Gift Pack 30% / Standard Pack 70%

## Technical Details

### Nested Quota Logic
The tool uses recursive assignment to ensure nested quotas are respected:
1. Split all prompted users according to Level 1 percentages
2. Within each Level 1 group, split according to Level 2 percentages
3. Within each Level 2 group, split according to Level 3 percentages

### Constraint Handling
- Specific preferences (e.g., "ASDA") are always respected
- "Either" preferences are distributed to balance across all options
- Multiple constraints are evaluated independently

### Randomization
- Random seed set to 42 for reproducibility
- Users are shuffled before assignment
- "Either" users are randomly distributed for balance

## Troubleshooting

**Percentages don't sum to 100%**
- Adjust your percentages so each level totals exactly 100%

**Total sample size doesn't match**
- Ensure Prompted + Unprompted = Total rows in your CSV

**Uneven distribution in results**
- This is normal due to rounding when percentages don't divide evenly
- The tool minimizes variance by assigning remainder to the last option

## Support

For issues or questions, please create an issue on GitHub.

## License

MIT License - feel free to use and modify as needed.
