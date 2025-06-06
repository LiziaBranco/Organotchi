import os

def rename_inference_filtered_dirs(base_dir, project_dir, days, plates):
    for day in days:
        print(f"- Processing Day {day}")
        for plate in plates:
            root_path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}', plate)
            for root, dirs, files in os.walk(root_path):
                for dir_name in dirs:
                    if dir_name == "inference_filtered":
                        old_path = os.path.join(root, dir_name)
                        new_path = os.path.join(root, "inference_filtered_jpg")
                        os.rename(old_path, new_path)
                        print(f"Renamed: {old_path} → {new_path}")

# Example usage
base_dir = '../../../AGMayer/Lizia/Organoid_Segmentation'
project_dir = '2025_03_05-Organoid-Segmentation'


plates_info = [
    {"name": "QB01_2", "days_96": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], "days_24": [17, 18, 22, 23, 24, 25,  28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43], "num_plates": 4},
    {"name": "AB114", "days_96": [1, 2, 3, 4, 11, 12, 14, 15, 16, 17, 18], "days_24": [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], "num_plates": 3},
    {"name": "AB108", "days_96": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], "days_24": [22, 23, 24, 25,  28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43], "num_plates": 3},
    {"name": "QB01_1", "days_96": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], "days_24": [17, 18, 22, 23, 24, 25,  28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43], "num_plates": 4},
    {"name": "SB60", "days_96": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14], "days_24": [17, 18, 22, 23, 24, 25, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43], "num_plates": 4}
]

# Process each plate
for plate in plates_info:
    plate_name = plate["name"]
    
    # Process 96-well plates
    if "days_96" in plate and plate["days_96"]:
        print(f"\n🧠 Processing Plate {plate_name} (96-well plates)")
        rename_inference_filtered_dirs(base_dir, project_dir, plate["days_96"], [plate_name])
