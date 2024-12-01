# Filter rows where 'depreciation' is greater than 300000
high_depreciation_rows = train_df_dropped[train_df_dropped['depreciation'] > 300000]

selected_columns = [
    'listing_id', 'make', 'model', 'manufactured', 'reg_date', 'type_of_vehicle',
    'category', 'transmission', 'curb_weight', 'power', 'fuel_type', 'engine_cap',
    'no_of_owners', 'depreciation', 'coe', 'road_tax', 'dereg_value', 'mileage',
    'omv', 'arf', 'price'
]

# Filter the high depreciation rows with selected columns
high_depreciation_selected = high_depreciation_rows[selected_columns]

# Filter rows where 'road_tax' is greater than 12000
high_road_tax_rows = train_df_dropped[train_df_dropped['road_tax'] > 12000]

# Select the specific columns to display
high_road_tax_selected = high_road_tax_rows[selected_columns]

# Filter rows where 'dereg_value' is greater than 1 million (1e6)
high_dereg_value_rows = train_df_dropped[train_df_dropped['dereg_value'] > 1e6]

# Select the specific columns to display
high_dereg_value_selected = high_dereg_value_rows[selected_columns]

# Filter rows where 'mileage' is greater than 600,000 (0.6 * 1e6)
high_mileage_rows = train_df_dropped[train_df_dropped['mileage'] > 0.6 * 1e6]

# Select the specific columns to display
high_mileage_selected = high_mileage_rows[selected_columns]

# Filter rows where 'omv' is greater than 600,000
high_omv_rows = train_df_dropped[train_df_dropped['omv'] > 600000]

# Select the specific columns to display
high_omv_selected = high_omv_rows[selected_columns]

# Filter rows where 'arf' is greater than 1,100,000 (1.1e6)
high_arf_rows = train_df_dropped[train_df_dropped['arf'] > 1.1e6]

# Select the specific columns to display
high_arf_selected = high_arf_rows[selected_columns]