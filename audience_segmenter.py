import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import io

st.set_page_config(page_title="Audience Segmentation Tool", layout="wide")

def initialize_session_state():
    """Initialize session state variables"""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'constraint_columns' not in st.session_state:
        st.session_state.constraint_columns = []
    if 'level_configs' not in st.session_state:
        st.session_state.level_configs = []

def validate_sample_sizes(prompted, unprompted, total):
    """Check if sample sizes are valid"""
    selected_total = prompted + unprompted
    if selected_total > total:
        return False, f"Prompted ({prompted}) + Unprompted ({unprompted}) = {selected_total} exceeds total rows ({total})"
    if prompted < 0 or unprompted < 0:
        return False, "Sample sizes must be positive numbers"
    return True, ""

def assign_nested_quotas(df_subset: pd.DataFrame, level_configs: List[Dict], 
                        constraint_columns: List[str], constraint_mappings: Dict[str, str]) -> pd.DataFrame:
    """
    Assign nested quotas to prompted users with constraint respect
    
    Args:
        df_subset: DataFrame of users to assign
        level_configs: List of level configurations with name, options, and percentages
        constraint_columns: List of columns to respect as constraints
        constraint_mappings: Dict mapping level names to their constraint columns
    
    Returns:
        DataFrame with assignment columns added
    """
    result_df = df_subset.copy()
    
    # If no levels configured, return as-is
    if not level_configs:
        return result_df
    
    # Start with all users available
    remaining_indices = result_df.index.tolist()
    assignments = {level['name']: [None] * len(result_df) for level in level_configs}
    
    def get_valid_options_for_user(idx, level_name, options):
        """Get valid options for a user based on constraints"""
        # Check if this level has a constraint mapping
        if level_name not in constraint_mappings or not constraint_mappings[level_name]:
            # No constraint, all options valid
            return [opt['value'] for opt in options]
        
        constraint_col = constraint_mappings[level_name]
        user_preference = result_df.loc[idx, constraint_col]
        
        # If user has "Either" or similar flexible preference, all options valid
        if str(user_preference).lower() == 'either':
            return [opt['value'] for opt in options]
        
        # Otherwise, only return matching option if it exists
        matching_options = [opt['value'] for opt in options 
                          if opt['value'].lower() == str(user_preference).lower()]
        
        # If no exact match found, return all options (fallback)
        return matching_options if matching_options else [opt['value'] for opt in options]
    
    def assign_level(indices: List, level_idx: int, parent_assignment: str = None):
        """Recursively assign quotas at each level with constraint respect"""
        if level_idx >= len(level_configs):
            return
        
        level = level_configs[level_idx]
        level_name = level['name']
        options = level['options']
        
        # Separate indices by their constraint eligibility
        indices_by_option = {opt['value']: [] for opt in options}
        flexible_indices = []  # Users who can go to any option (have "Either")
        
        for idx in indices:
            valid_options = get_valid_options_for_user(idx, level_name, options)
            
            if len(valid_options) == len(options):
                # User is flexible (has "Either" or no constraint)
                flexible_indices.append(idx)
            elif len(valid_options) == 1:
                # User has specific constraint
                indices_by_option[valid_options[0]].append(idx)
            else:
                # Multiple valid options but not all - treat as flexible
                flexible_indices.append(idx)
        
        # Calculate targets
        total_at_level = len(indices)
        targets = {}
        cumulative = 0
        
        for i, opt in enumerate(options):
            if i == len(options) - 1:
                targets[opt['value']] = total_at_level - cumulative
            else:
                targets[opt['value']] = int(total_at_level * opt['percentage'] / 100)
                cumulative += targets[opt['value']]
        
        # Assign constrained users first
        assigned_by_option = {}
        for opt_value in targets.keys():
            constrained = indices_by_option[opt_value]
            assigned_by_option[opt_value] = constrained[:targets[opt_value]]
            
            # Record assignments
            for idx in assigned_by_option[opt_value]:
                assignments[level_name][result_df.index.get_loc(idx)] = opt_value
        
        # Distribute flexible users to meet targets
        np.random.shuffle(flexible_indices)
        flexible_idx = 0
        
        for opt_value, target_count in targets.items():
            current_count = len(assigned_by_option[opt_value])
            needed = target_count - current_count
            
            # Assign flexible users to meet target
            while needed > 0 and flexible_idx < len(flexible_indices):
                idx = flexible_indices[flexible_idx]
                assigned_by_option[opt_value].append(idx)
                assignments[level_name][result_df.index.get_loc(idx)] = opt_value
                flexible_idx += 1
                needed -= 1
        
        # Recursively assign next level within each option
        if level_idx + 1 < len(level_configs):
            for opt_value, opt_indices in assigned_by_option.items():
                assign_level(opt_indices, level_idx + 1, opt_value)
    
    # Start recursive assignment
    assign_level(remaining_indices, 0)
    
    # Add assignment columns to result
    for level_name, values in assignments.items():
        result_df[level_name] = values
    
    return result_df

def assign_retailer(df_subset: pd.DataFrame, retailer_col: str) -> pd.DataFrame:
    """Assign retailer based on preference, balancing 'Either' across options"""
    result_df = df_subset.copy()
    result_df['Assigned_Retailer'] = None
    
    # Get unique retailers (excluding 'Either')
    all_retailers = result_df[retailer_col].unique()
    specific_retailers = [r for r in all_retailers if r.lower() != 'either']
    
    # First pass: assign specific preferences
    for retailer in specific_retailers:
        mask = result_df[retailer_col] == retailer
        result_df.loc[mask, 'Assigned_Retailer'] = retailer
    
    # Second pass: distribute 'Either' to balance retailers
    either_mask = result_df[retailer_col].str.lower() == 'either'
    either_indices = result_df[either_mask].index.tolist()
    
    if either_indices and specific_retailers:
        # Count current assignments
        retailer_counts = result_df['Assigned_Retailer'].value_counts().to_dict()
        
        # Shuffle 'Either' users
        np.random.shuffle(either_indices)
        
        # Assign to balance retailers
        for idx in either_indices:
            # Find retailer with lowest count
            min_retailer = min(specific_retailers, key=lambda r: retailer_counts.get(r, 0))
            result_df.loc[idx, 'Assigned_Retailer'] = min_retailer
            retailer_counts[min_retailer] = retailer_counts.get(min_retailer, 0) + 1
    
    return result_df

def segment_audience(df: pd.DataFrame, prompted_count: int, unprompted_count: int, 
                     retailer_col: str, constraint_columns: List[str], 
                     level_configs: List[Dict], constraint_mappings: Dict[str, str]) -> pd.DataFrame:
    """
    Main segmentation function
    
    Args:
        df: Input DataFrame
        prompted_count: Number of prompted users
        unprompted_count: Number of unprompted users
        retailer_col: Name of retailer column
        constraint_columns: Additional constraint columns for prompted users
        level_configs: List of level configurations
        constraint_mappings: Dict mapping level names to their constraint columns
    
    Returns:
        DataFrame with segmentation assignments
    """
    result_df = df.copy()
    
    # Shuffle the dataframe
    result_df = result_df.sample(frac=1).reset_index(drop=True)
    
    # Calculate total selected
    total_selected = prompted_count + unprompted_count
    
    # Split into prompted, unprompted, and backup
    prompted_df = result_df.iloc[:prompted_count].copy()
    unprompted_df = result_df.iloc[prompted_count:total_selected].copy()
    backup_df = result_df.iloc[total_selected:].copy()
    
    # Add segment type column
    prompted_df['Segment_Type'] = 'Prompted'
    unprompted_df['Segment_Type'] = 'Unprompted'
    backup_df['Segment_Type'] = 'Backup'
    
    # Process prompted users - apply nested quotas
    if level_configs:
        prompted_df = assign_nested_quotas(prompted_df, level_configs, constraint_columns, constraint_mappings)
    
    # Assign retailers for prompted and unprompted (not for backups)
    prompted_df = assign_retailer(prompted_df, retailer_col)
    unprompted_df = assign_retailer(unprompted_df, retailer_col)
    
    # For backups, don't assign anything - just add the column as None
    backup_df['Assigned_Retailer'] = None
    
    # Add None values for level columns in backup
    if level_configs:
        for level in level_configs:
            backup_df[level['name']] = None
    
    # Combine results
    final_df = pd.concat([prompted_df, unprompted_df, backup_df], ignore_index=True)
    
    return final_df

def main():
    st.title("🎯 Audience Segmentation Tool")
    st.markdown("Segment your audience with flexible nested quotas and constraints")
    
    initialize_session_state()
    
    # Step 1: Upload CSV
    st.header("1. Upload CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success(f"✓ Loaded {len(st.session_state.df)} rows")
        
        with st.expander("Preview Data"):
            st.dataframe(st.session_state.df.head(10))
    
    if st.session_state.df is not None:
        df = st.session_state.df
        total_rows = len(df)
        
        # Step 2: Set Sample Sizes
        st.header("2. Set Sample Sizes")
        st.markdown("*Select how many people you need - remaining will be marked as Backups*")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            prompted_count = st.number_input("Prompted users", min_value=0, max_value=total_rows, value=min(50, total_rows))
        
        with col2:
            unprompted_count = st.number_input("Unprompted users", min_value=0, max_value=total_rows, value=min(20, total_rows))
        
        with col3:
            total_selected = prompted_count + unprompted_count
            backup_count = total_rows - total_selected
            st.metric("Selected", f"{total_selected}")
        
        with col4:
            if backup_count > 0:
                st.metric("Backups", f"{backup_count}", delta=None, delta_color="off")
            else:
                st.metric("Backups", "0")
        
        valid, error_msg = validate_sample_sizes(prompted_count, unprompted_count, total_rows)
        if not valid:
            st.error(error_msg)
            return
        
        if backup_count > 0:
            st.info(f"ℹ️ {backup_count} participants will be marked as 'Backup' (unassigned)")
        
        # Step 3: Select Retailer Column
        st.header("3. Select Retailer Column")
        st.markdown("*Required - used for both prompted and unprompted users*")
        
        retailer_col = st.selectbox("Retailer Column", options=df.columns.tolist())
        
        # Show retailer distribution
        with st.expander("View Retailer Distribution"):
            retailer_counts = df[retailer_col].value_counts()
            st.bar_chart(retailer_counts)
        
        # Step 4: Additional Constraint Columns
        st.header("4. Additional Constraint Columns (Optional)")
        st.markdown("*These constraints apply only to prompted users*")
        
        add_constraints = st.checkbox("Add constraint columns")
        constraint_columns = []
        
        if add_constraints:
            available_cols = [col for col in df.columns.tolist() if col != retailer_col]
            constraint_columns = st.multiselect(
                "Select constraint columns",
                options=available_cols,
                help="e.g., Cooking Preference, Flavour Preference"
            )
            
            if constraint_columns:
                for col in constraint_columns:
                    with st.expander(f"View {col} Distribution"):
                        col_counts = df[col].value_counts()
                        st.bar_chart(col_counts)
        
        # Step 5: Product Assignment (Prompted Users)
        st.header("5. Product Assignment (Prompted Users)")
        
        add_assignment = st.checkbox("Add product/variant assignment")
        level_configs = []
        
        if add_assignment:
            # Level 1
            st.subheader("Level 1")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                level1_name = st.text_input("Level 1 Name", value="Product_Variant", key="level1_name")
            
            with col2:
                num_level1_options = st.number_input("Number of options", min_value=2, max_value=10, value=2, key="level1_num")
            
            level1_options = []
            cols = st.columns(num_level1_options)
            
            for i in range(num_level1_options):
                with cols[i]:
                    opt_name = st.text_input(f"Option {i+1}", value=f"Option_{i+1}", key=f"level1_opt_{i}")
                    opt_pct = st.number_input(f"% {opt_name}", min_value=0, max_value=100, value=100//num_level1_options, key=f"level1_pct_{i}")
                    level1_options.append({"value": opt_name, "percentage": opt_pct})
            
            # Validate percentages
            total_pct = sum(opt['percentage'] for opt in level1_options)
            if total_pct != 100:
                st.warning(f"⚠️ Level 1 percentages sum to {total_pct}% (should be 100%)")
            
            level_configs.append({
                "name": level1_name,
                "options": level1_options
            })
            
            # Level 2
            add_level2 = st.checkbox("Add Level 2 (nested within Level 1)")
            
            if add_level2:
                st.subheader("Level 2 (Nested)")
                st.info(f"This will be split within EACH {level1_name} option")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    level2_name = st.text_input("Level 2 Name", value="Cooking_Method", key="level2_name")
                
                with col2:
                    num_level2_options = st.number_input("Number of options", min_value=2, max_value=10, value=2, key="level2_num")
                
                level2_options = []
                cols = st.columns(num_level2_options)
                
                for i in range(num_level2_options):
                    with cols[i]:
                        opt_name = st.text_input(f"Option {i+1}", value=f"Option_{i+1}", key=f"level2_opt_{i}")
                        opt_pct = st.number_input(f"% {opt_name}", min_value=0, max_value=100, value=100//num_level2_options, key=f"level2_pct_{i}")
                        level2_options.append({"value": opt_name, "percentage": opt_pct})
                
                # Validate percentages
                total_pct = sum(opt['percentage'] for opt in level2_options)
                if total_pct != 100:
                    st.warning(f"⚠️ Level 2 percentages sum to {total_pct}% (should be 100%)")
                
                level_configs.append({
                    "name": level2_name,
                    "options": level2_options
                })
                
                # Level 3
                add_level3 = st.checkbox("Add Level 3 (nested within Level 2)")
                
                if add_level3:
                    st.subheader("Level 3 (Nested)")
                    st.info(f"This will be split within EACH {level2_name} option")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        level3_name = st.text_input("Level 3 Name", value="Additional_Attribute", key="level3_name")
                    
                    with col2:
                        num_level3_options = st.number_input("Number of options", min_value=2, max_value=10, value=2, key="level3_num")
                    
                    level3_options = []
                    cols = st.columns(num_level3_options)
                    
                    for i in range(num_level3_options):
                        with cols[i]:
                            opt_name = st.text_input(f"Option {i+1}", value=f"Option_{i+1}", key=f"level3_opt_{i}")
                            opt_pct = st.number_input(f"% {opt_name}", min_value=0, max_value=100, value=100//num_level3_options, key=f"level3_pct_{i}")
                            level3_options.append({"value": opt_name, "percentage": opt_pct})
                    
                    # Validate percentages
                    total_pct = sum(opt['percentage'] for opt in level3_options)
                    if total_pct != 100:
                        st.warning(f"⚠️ Level 3 percentages sum to {total_pct}% (should be 100%)")
                    
                    level_configs.append({
                        "name": level3_name,
                        "options": level3_options
                    })
            
            # Constraint Mapping Section
            if level_configs and constraint_columns:
                st.markdown("---")
                st.subheader("🔗 Map Constraints to Assignment Levels")
                st.markdown("*Optional: Link constraint columns to assignment levels to respect user preferences*")
                st.info("💡 Example: Map 'Cooking Preference' to 'Cooking Method' so users with 'Soup' preference only get 'Soup' assignments")
                
                constraint_mappings = {}
                
                for level in level_configs:
                    level_name = level['name']
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.write(f"**{level_name}**")
                    with col2:
                        mapping_options = ["None (no constraint)"] + constraint_columns
                        selected_constraint = st.selectbox(
                            f"Constraint for {level_name}",
                            options=mapping_options,
                            key=f"constraint_map_{level_name}",
                            help=f"Select which constraint column should be respected when assigning {level_name}"
                        )
                        
                        if selected_constraint != "None (no constraint)":
                            constraint_mappings[level_name] = selected_constraint
                        else:
                            constraint_mappings[level_name] = None
                
                # Show mapping summary
                if any(constraint_mappings.values()):
                    st.success("✓ Active constraint mappings:")
                    for level_name, constraint_col in constraint_mappings.items():
                        if constraint_col:
                            st.write(f"  - {level_name} respects {constraint_col}")
            else:
                constraint_mappings = {}
        else:
            constraint_mappings = {}
        
        # Step 6: Run Segmentation
        st.header("6. Run Segmentation")
        
        if st.button("🚀 Run Segmentation", type="primary", use_container_width=True):
            with st.spinner("Segmenting audience..."):
                # Set random seed for reproducibility
                np.random.seed(42)
                
                # Run segmentation
                result_df = segment_audience(
                    df=df,
                    prompted_count=prompted_count,
                    unprompted_count=unprompted_count,
                    retailer_col=retailer_col,
                    constraint_columns=constraint_columns,
                    level_configs=level_configs,
                    constraint_mappings=constraint_mappings
                )
                
                st.session_state.result_df = result_df
                st.success("✓ Segmentation complete!")
        
        # Display Results
        if 'result_df' in st.session_state:
            st.header("Results")
            
            result_df = st.session_state.result_df
            
            # Summary Statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                prompted = (result_df['Segment_Type'] == 'Prompted').sum()
                st.metric("Prompted Users", prompted)
            
            with col2:
                unprompted = (result_df['Segment_Type'] == 'Unprompted').sum()
                st.metric("Unprompted Users", unprompted)
            
            with col3:
                backups = (result_df['Segment_Type'] == 'Backup').sum()
                st.metric("Backups", backups)
            
            with col4:
                st.metric("Total", len(result_df))
            
            # Detailed Breakdown
            tabs = st.tabs(["Distribution Summary", "Full Data Preview"])
            
            with tabs[0]:
                # Retailer distribution
                st.subheader("Retailer Distribution")
                retailer_dist = result_df.groupby(['Segment_Type', 'Assigned_Retailer']).size().reset_index(name='Count')
                st.dataframe(retailer_dist, use_container_width=True)
                
                # Level distributions
                if level_configs:
                    prompted_df = result_df[result_df['Segment_Type'] == 'Prompted']
                    
                    for level in level_configs:
                        st.subheader(f"{level['name']} Distribution (Prompted Only)")
                        level_dist = prompted_df[level['name']].value_counts().reset_index()
                        level_dist.columns = [level['name'], 'Count']
                        level_dist['Percentage'] = (level_dist['Count'] / len(prompted_df) * 100).round(1)
                        st.dataframe(level_dist, use_container_width=True)
                    
                    # Cross-tab for nested levels
                    if len(level_configs) >= 2:
                        st.subheader("Nested Distribution")
                        crosstab = pd.crosstab(
                            prompted_df[level_configs[0]['name']], 
                            prompted_df[level_configs[1]['name']], 
                            margins=True
                        )
                        st.dataframe(crosstab, use_container_width=True)
            
            with tabs[1]:
                st.dataframe(result_df, use_container_width=True)
            
            # Download
            st.subheader("Download Results")
            
            # Convert to CSV
            csv_buffer = io.StringIO()
            result_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Download Segmented CSV",
                data=csv_data,
                file_name="segmented_audience.csv",
                mime="text/csv",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
