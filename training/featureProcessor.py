import os
import pandas as pd
from RouteProcessor3 import runRouteProcessor

def calculate_acceleration(df, velocity_col='target_speed', distance_col='range'):
    """
    Calculate acceleration using forward, backward, and central differences.
    """
    acceleration = [0] * len(df)
    for i in range(len(df)):
        try: 
            if i == 0:  # Forward difference for the first row
                acceleration[i] = (df[velocity_col][i + 1] - df[velocity_col][i]) * df[velocity_col][i] / (df[distance_col][i + 1] - df[distance_col][i])
            elif i == len(df) - 1:  # Backward difference for the last row
                acceleration[i] = (df[velocity_col][i] - df[velocity_col][i - 1]) * df[velocity_col][i] / (df[distance_col][i] - df[distance_col][i - 1])
            else:  # Central difference for the rest
                acceleration[i] = (df[velocity_col][i + 1] - df[velocity_col][i - 1]) * df[velocity_col][i] / (df[distance_col][i + 1] - df[distance_col][i - 1])
        except ZeroDivisionError:
            acceleration[i] = 0 # Handle division by zero if distance difference is zero
        except Exception as e:
            print(f"Error calculating acceleration for row {i}: {e}")
    return acceleration

def process_csv_files(folder_path):
    """
    Process all .csv files in the specified folder and its subfolders.
    """
    skipped_files = []  # Initialize a list to store skipped file paths

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                try:
                    print(f"Processing file: {file_path}")

                    # Read the CSV file
                    df = pd.read_csv(file_path, low_memory=False)

                    # Ensure required columns exist
                    if 'latitude' not in df.columns or 'longitude' not in df.columns:
                        print(f"Skipping file: '{file_path}' - 'latitude' or 'longitude' columns missing.")
                        skipped_files.append(file_path)
                        
                        # Write skipped file to skipped_files.txt immediately
                        with open(os.path.join(folder_path, "skipped_files.txt"), "a") as f:
                            f.write(file_path + "\n")
                        
                        continue

                    # Extract coordinates in [[lon1, lat1], [lon2, lat2], ...] format
                    coords = df[['longitude', 'latitude']].values.tolist()

                    # # Debugging log for coords
                    # print(f"Extracted coordinates: {coords}")

                    # Ensure coords is not empty
                    if not coords:
                        print(f"Skipping file: '{file_path}' - No valid coordinates found.")
                        skipped_files.append(file_path)
                        
                        # Write skipped file to skipped_files.txt immediately
                        with open(os.path.join(folder_path, "skipped_files.txt"), "a") as f:
                            f.write(file_path + "\n")
                        
                        continue

                    # Ensure no NaN values in longitude and latitude
                    if df[['longitude', 'latitude']].isnull().values.any():
                        print(f"Skipping file: '{file_path}' - Contains NaN values in 'longitude' or 'latitude'.")
                        skipped_files.append(file_path)
                        
                        # Write skipped file to skipped_files.txt immediately
                        with open(os.path.join(folder_path, "skipped_files.txt"), "a") as f:
                            f.write(file_path + "\n")
                        
                        continue

                    # Call runRouteProcessor with the extracted coordinates
                    nodePD = runRouteProcessor(coords)

                    if nodePD.empty:
                        print(f"Skipping file: '{file_path}' - runRouteProcessor returned empty data.")
                        skipped_files.append(file_path)
                        
                        # Write skipped file to skipped_files.txt immediately
                        with open(os.path.join(folder_path, "skipped_files.txt"), "a") as f:
                            f.write(file_path + "\n")
                        
                        continue

                    # Reset index of original DataFrame for proper merging
                    df = df.reset_index(drop=True)

                    # Combine the original DataFrame and the augmented DataFrame
                    combined_df = pd.concat([df, nodePD], axis=1)

                    # Calculate acceleration and append as a new column
                    combined_df['acceleration'] = calculate_acceleration(combined_df)

                    # Save the combined DataFrame back to the original file path
                    combined_df.to_csv(file_path, index=False)

                    print(f"Successfully processed and saved: {file_path}")

                except FileNotFoundError:
                    print(f"Skipping file: '{file_path}' - File not found.")
                    skipped_files.append(file_path)
                    
                    # Write skipped file to skipped_files.txt immediately
                    with open(os.path.join(folder_path, "skipped_files.txt"), "a") as f:
                        f.write(file_path + "\n")
                        
                except pd.errors.EmptyDataError:
                    print(f"Skipping file: '{file_path}' - Empty file.")
                    skipped_files.append(file_path)
                    
                    # Write skipped file to skipped_files.txt immediately
                    with open(os.path.join(folder_path, "skipped_files.txt"), "a") as f:
                        f.write(file_path + "\n")
                        
                except pd.errors.ParserError as e:
                    print(f"Skipping file: '{file_path}' - Parser error: {e}")
                    skipped_files.append(file_path)
                    
                    # Write skipped file to skipped_files.txt immediately
                    with open(os.path.join(folder_path, "skipped_files.txt"), "a") as f:
                        f.write(file_path + "\n")
                        
                except Exception as e:
                    print(f"Skipping file: '{file_path}' - An unexpected error occurred: {e}")
                    skipped_files.append(file_path)
                    
                    # Write skipped file to skipped_files.txt immediately
                    with open(os.path.join(folder_path, "skipped_files.txt"), "a") as f:
                        f.write(file_path + "\n")

    return skipped_files
                
if __name__ == "__main__":
    folder_path = ".\\ieee_vehicle_Speed_Dataset"
    process_csv_files(folder_path)