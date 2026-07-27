import os
import pandas as pd

condition_map = {
    "AB121": {
        "AB121_1_1 CTRL": ["Tile000001", "Tile000002","Tile000003","Tile000007","Tile000008","Tile000009","Tile000013","Tile0000014","Tile0000015","Tile0000019","Tile000020","Tile000021","Tile000025",
                 "Tile000026","Tile000027","Tile000031", "Tile000032","Tile000033","Tile000037","Tile000038","Tile000039","Tile000043","Tile000044",
                 "Tile000045","Tile000073","Tile000074","Tile000075","Tile000079",
                    "Tile000080", "Tile000081","Tile000085","Tile000086","Tile000087","Tile000091","Tile000092","Tile000093"],
        "AB121_1_1 VPA": ["Tile000004", "Tile000005","Tile000006","Tile000010","Tile000011","Tile000012","Tile000016","Tile000017","Tile000018","Tile000022","Tile000023","Tile000024",
                    "Tile000028", "Tile000029","Tile000030","Tile000034","Tile0000035","Tile0000036","Tile000040","Tile000041","Tile000042","Tile000046","Tile000047","Tile000048",
                    "Tile000076", "Tile000077","Tile000078","Tile000082","Tile000083","Tile000084", "Tile000088","Tile000089","Tile000090","Tile000094","Tile000095","Tile000096"],
        "AB121_1_2 CTRL": ["Tile000049","Tile000050","Tile000051","Tile000055","Tile000056", "Tile000057"],
        "AB121_1_2 VPA": ["Tile000052", "Tile000053","Tile000054","Tile000058","Tile000059","Tile000060"],
        "AB121_1_2 PIEZO": ["Tile000061","Tile000062","Tile000063","Tile000067", "Tile000068","Tile000069"],
        "AB121_1_2 VPA+PIEZO": ["Tile000064","Tile000065","Tile000066","Tile000070","Tile000071","Tile000072"]
    },
    "AB121dd": {
        "AB121_2 VPA": ["Tile000001", "Tile000002","Tile000003", "Tile000004", "Tile000005","Tile000006"],
        "AB121_2 PIEZO": ["Tile000007","Tile000008","Tile000009","Tile000010","Tile000011","Tile000012"],
        "AB121_2 VPA+PIEZO": ["Tile000013","Tile0000014","Tile0000015", "Tile000016","Tile000017","Tile000018"],
        "AB121_2 CTRL": ["Tile0000019","Tile000020","Tile000021","Tile000022","Tile000023","Tile000024"]
    },
    "P01": {
        "P01 CTRL": ["Tile000001", "Tile000002","Tile000003","Tile000007","Tile000008","Tile000009","Tile000013","Tile0000014","Tile0000015","Tile0000019","Tile000020","Tile000021","Tile000025","Tile000026","Tile000027",
                    "Tile000031", "Tile000032","Tile000033","Tile000037","Tile000038","Tile000039","Tile000043","Tile000044","Tile000045","Tile000049","Tile000050","Tile000051","Tile000055",
                    "Tile000056", "Tile000057","Tile000061","Tile000062","Tile000063","Tile000067", "Tile000068","Tile000069","Tile000073","Tile000074","Tile000075","Tile000079",
                    "Tile000080", "Tile000081","Tile000085","Tile000086","Tile000087","Tile000091","Tile000092","Tile000093"],
        "P01 VPA": ["Tile000004", "Tile000005","Tile000006","Tile000010","Tile000011","Tile000012","Tile000016","Tile000017","Tile000018","Tile000022","Tile000023","Tile000024",
                    "Tile000028", "Tile000029","Tile000030","Tile000034","Tile0000035","Tile0000036","Tile000040","Tile000041","Tile000042","Tile000046","Tile000047","Tile000048",
                    "Tile000052", "Tile000053","Tile000054","Tile000058","Tile000059","Tile000060","Tile000064","Tile000065","Tile000066","Tile000070","Tile000071","Tile000072",
                    "Tile000076", "Tile000077","Tile000078","Tile000082","Tile000083","Tile000084","Tile000088","Tile000089","Tile000090","Tile000094","Tile000095","Tile000096"]
    }
}

# === INPUT DIRECTORY ===
base_dir = "../../../AGMayer/Lizia/Organoid_Segmentation/Zeynep/processed_data"

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
