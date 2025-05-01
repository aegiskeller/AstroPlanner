from corePlanner import get_targets, get_target_airmass

# Fetch the targets DataFrame from the csv file seleceted_target.csv
targets_df = get_targets()  # Fetch the targets DataFrame
# save the first row of the DataFrame to a CSV file
targets_df.iloc[0:1].to_csv('selected_target.csv', index=False)
# Example usage of get_target_airmass function
selected_data = targets_df.iloc[0]  # Replace with actual selected data
airmass = get_target_airmass(selected_data)
print("Airmass values for the selected target:")
print(airmass)
print("Selected target data:")
print(selected_data)
