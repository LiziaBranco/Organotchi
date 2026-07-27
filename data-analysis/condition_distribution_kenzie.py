import os
import pandas as pd

condition_map = {
    "P01": {
        "D35s GM+NL": ["Tile000001", "Tile000002","Tile000003","Tile000004","Tile000005","Tile000006","Tile000052","Tile000053","Tile000054"],
        "D35s GM": ["Tile000007", "Tile000008","Tile000009","Tile000010","Tile000011","Tile000012","Tile000058","Tile000059","Tile000060"],
	    "D35s NL": ["Tile000013", "Tile000014","Tile000015","Tile000016","Tile000017","Tile000018","Tile000064","Tile000065","Tile000066"],
        "D35s CTRL": ["Tile000019", "Tile000020","Tile000021","Tile000022","Tile000023","Tile000024","Tile000070","Tile000071","Tile000072"],
        "D35s DMSO": ["Tile000049","Tile000050","Tile000051","Tile000055", "Tile000056","Tile000057"],
        "D50s GM+NL": ["Tile000025","Tile000026","Tile000027","Tile000028", "Tile000029","Tile000030"],
        "D50s GM": ["Tile000031", "Tile000032","Tile000033","Tile000034","Tile000035","Tile000036"],
        "D50s NL": ["Tile000037","Tile000038","Tile000039","Tile000040","Tile000041", "Tile000042"],
        "D50s CTRL": ["Tile000043", "Tile000044","Tile000045","Tile000046","Tile000047","Tile000048"],
        "D50s DMSO": ["Tile000061","Tile000062","Tile000063","Tile000067", "Tile000068","Tile000069"]
    }
}

# === INPUT DIRECTORY ===
base_dir = "../../../AGMayer/Lizia/Organoid_Segmentation/Kenzie/processed_data"

for day in os.listdir(base_dir):
    day_path = os.path.join(base_dir, day)
    if not os.path.isdir(day_path):
        continue

    for batch in os.listdir(day_path):
        batch_path = os.path.join(day_path, batch)
        metrics_path = os.path.join(batch_path, "inference_metrics", "mask_metrics.csv")
        
        if not os.path.exists(metrics_path):
            continue

        df = pd.read_csv(metrics_path)

        if batch not in condition_map:
            continue

        for condition, tiles in condition_map[batch].items():
            df_cond = df[df["Organoid_ID"].isin(tiles)]
            if df_cond.empty:
                continue

            # Build output path: processed_data/DayXX/Condition/inference_metrics/
            out_dir = os.path.join(day_path, condition, "inference_metrics")
            os.makedirs(out_dir, exist_ok=True)

            out_file = os.path.join(out_dir, "mask_metrics.csv")

            if os.path.exists(out_file):
                # Append to existing condition file
                df_existing = pd.read_csv(out_file)
                df_combined = pd.concat([df_existing, df_cond], ignore_index=True)
                df_combined.to_csv(out_file, index=False)
            else:
                # Write new condition file
                df_cond.to_csv(out_file, index=False)

print("✅ Condition-specific mask_metrics.csv files created.")
