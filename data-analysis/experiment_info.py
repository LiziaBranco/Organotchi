import questionary
import sys
import os
import json

def run_project_logic(project_dir):
    global batches_info, batches_complete_info

    # Convert the project_dir into a valid JSON filename
    json_filename = f"{project_dir.replace('/', '_')}.json"
    json_path = os.path.join("Organoid-Segmentation/data-analysis", json_filename)  # Adjust folder path as needed

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No project data found for: {project_dir} at {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    batches_info = data.get("batches_info", [])
    batches_complete_info = data.get("batches_complete_info", [])

    print("📦 Project data loaded successfully.")
    
# Define the available projects
AVAILABLE_PROJECT_DIRS = [
    '2025_03_05-Organoid-Segmentation',
    '2025_03_05-Theresa-SmallScale-Trial/AB108-AB112',
    '2025_03_05-Theresa-SmallScale-Trial/SB60-SB64',
    '2025_03_05-Theresa-SmallScale-Trial/QB01-QB05',
    '6_WNT_treatment/AB115-AB117',
    'Francesco',
    'Isabel',
    'Zeynep',
    'Kenzie'
]

# Shared config
checkpoint = 'Training-Results/work_dirs/split-0/iter_144000.pth'
base_dir = '../../../AGMayer/Lizia/Organoid_Segmentation'

pixel_to_micron = {
    "96_well": {"x": 0.8872, "y": 0.8874},
    "24_well": {"x": 2.8391, "y": 2.8396}
}

features = ['Area_um2', 'Diameter_um', 'Perimeter_um', 'Circularity',
            'Skeleton_Length_um', 'Solidity', 'Eccentricity', 'Roughness_Index']

# Prompt menu
print("🧠 Organoid Segmentation - Project Selector\n")

while True:
    action = questionary.select(
        "What would you like to do?",
        choices=[
            "Select a project directory",
            "List all project directories",
            "Exit"
        ]
    ).ask()

    if action == "List all project directories":
        print("\n📂 Available project directories:")
        for proj in AVAILABLE_PROJECT_DIRS:
            print(f"  - {proj}")
        print("")  # Blank line for readability
        # Instead of exiting, loop back to ask again

    elif action == "Select a project directory":
        project_dir = questionary.select(
            "Select a project directory:",
            choices=AVAILABLE_PROJECT_DIRS
        ).ask()

        if not project_dir:
            print("❌ No project selected. Try again.\n")
            continue  # Ask again
        print(f"\n✅ You selected: {project_dir}\n")
        run_project_logic(project_dir)
        break  # Exit the loop after successful selection

    elif action == "Exit":
        print("👋 Goodbye!")
        sys.exit(0)