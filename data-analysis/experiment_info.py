checkpoint = 'Training-Results/work_dirs/split-0/iter_144000.pth'
base_dir = '../../../AGMayer/Lizia/Organoid_Segmentation'

# Define pixel-to-micron conversion factors
pixel_to_micron = {
    "96_well": {"x": 0.8872, "y": 0.8874},  # 4x objective
    "24_well": {"x": 2.8391, "y": 2.8396}   # 1.25x objective
}

features = ['Area_um2', 'Diameter_um', 'Perimeter_um', 'Circularity', 'Skeleton_Length_um', 'Solidity', 'Eccentricity', 'Roughness_Index']

project_dir = '2025_03_05-Organoid-Segmentation'
project_dir = '2025_03_05-Theresa-SmallScale-Trial/AB108-AB112'
#project_dir = '2025_03_05-Theresa-SmallScale-Trial/SB60-SB64'
#project_dir = '2025_03_05-Theresa-SmallScale-Trial/QB01-QB05'

if project_dir == '2025_03_05-Organoid-Segmentation':
    '''batches_info = [
        {"batch": "QB01_2", "days_96": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]},
        {"batch": "AB114", "days_96": [1, 2, 3, 4, 9, 11, 12, 13, 14, 15, 16, 17, 18], "days_24": [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35], "num_plates": 4},
        {"batch": "AB108", "days_96": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], "days_24": [22, 23, 24, 25,  28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50], "num_plates": 3},
        {"batch": "QB01_1", "days_96": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], "days_24": [17, 18, 22, 23, 24, 25, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50], "num_plates": 4},
        {"batch": "SB60", "days_96": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14], "days_24": [17, 18, 22, 23, 24, 25, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50], "num_plates": 4}
    ]'''
    
    batches_info = [
        {"batch": "AB114", "days_24": [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35], "num_plates": 4},
        {"batch": "AB108", "days_96": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], "days_24": [22, 23, 24, 25,  28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50], "num_plates": 3},
        {"batch": "QB01_1", "days_96": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], "days_24": [17, 18, 22, 23, 24, 25, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50], "num_plates": 4},
        {"batch": "SB60", "days_96": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14], "days_24": [17, 18, 22, 23, 24, 25, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50], "num_plates": 4}
    ]
    
    batches_complete_info = [
    {"batch": "AB108", "days": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 23, 24, 25, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45], "condition": "Protocol 1", "cell-line": "BIONi010-C"},
    {"batch": "QB01_1", "days": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 22, 23, 24, 25, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45], "condition": "Protocol 2", "cell-line": "BIONi010-C"},
    {"batch": "SB60", "days": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 22, 23, 24, 25, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45], "condition": "Protocol 3", "cell-line": "BIONi010-C"},
    {"batch": "AB114", "days": [1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], "condition": "Protocol 4", "cell-line": "BIONi010-C"}
    ]


elif project_dir == '2025_03_05-Theresa-SmallScale-Trial/AB108-AB112':
    batches_info = [
        {"batch": "AB108", "days_96": [1, 8, 10, 11, 13, 16],"days_24": [18, 22, 24, 28, 30, 32, 35, 50], "num_plates": 3},
        {"batch": "AB109", "days_96": [1, 8, 10, 13, 16],"days_24": [18, 22, 24, 28, 30, 32, 35, 50], "num_plates": 3},
        {"batch": "AB110", "days_96": [1, 8, 10, 13, 16],"days_24": [18, 22, 24, 28, 30, 32, 35, 50], "num_plates": 3},
        {"batch": "AB111", "days_96": [1, 8, 10, 13, 16],"days_24": [18, 22, 24, 28, 30, 32, 35, 50], "num_plates": 3},
        {"batch": "AB112", "days_96": [1, 8, 10, 13, 16],"days_24": [18, 22, 24, 28, 30, 32, 35, 50], "num_plates": 3}
    ]
    
    batches_complete_info = [  
    {"batch": "AB108", "days": [1, 8, 10, 11, 13, 16, 18, 22, 24, 28, 30, 32, 35, 50], "condition": "CTRL", "cell-line": "BIONi010-C"},
    {"batch": "AB109", "days": [1, 8, 10, 13, 16, 18, 22, 24, 28, 30, 32, 35, 50], "condition": "PCH", "cell-line": "PCH02"},
    {"batch": "AB110", "days": [1, 8, 10, 13, 16, 18, 22, 24, 28, 30, 32, 35, 50], "condition": "CTRL", "cell-line": "PCH02 Rescue"},
    {"batch": "AB111", "days": [1, 8, 10, 13, 16, 18, 22, 24, 28, 30, 32, 35, 50], "condition": "CTRL", "cell-line": "PCH03 Rescue"},
    {"batch": "AB112", "days": [1, 8, 10, 13, 16, 18, 22, 24, 28, 30, 32, 35, 50], "condition": "PCH", "cell-line": "PCH03"}
    ]


elif project_dir == '2025_03_05-Theresa-SmallScale-Trial/SB60-SB64':
    batches_info = [
    {"batch": "SB60", "days_96": [1, 2, 4, 7, 10, 13],"days_24": [17, 22, 24, 28, 32, 35, 50], "num_plates": 4},
    {"batch": "SB61", "days_96": [1, 2, 4, 7, 10, 13],"days_24": [17, 22, 24, 28, 32, 35, 50], "num_plates": 3},
    {"batch": "SB62", "days_96": [1, 2, 4, 7, 10, 13],"days_24": [17, 22, 24, 28, 32, 35, 50], "num_plates": 3},
    {"batch": "SB63", "days_96": [1, 2, 4, 7, 10, 13],"days_24": [17, 22, 24, 28, 32, 35, 50], "num_plates": 3},
    {"batch": "SB64", "days_96": [1, 4, 7, 10, 13],"days_24": [17, 22, 24, 28, 32, 35, 50], "num_plates": 3}
    ]

    batches_complete_info = [  
        {"batch": "SB60", "days": [1, 2, 4, 7, 10, 13, 17, 22, 24, 28, 32, 35, 50], "condition": "CTRL", "cell-line": "BIONi010-C"},
        {"batch": "SB61", "days": [1, 2, 4, 7, 10, 13, 17, 22, 24, 28, 32, 35, 50], "condition": "PCH", "cell-line": "PCH02"},
        {"batch": "SB62", "days": [1, 2, 4, 7, 10, 13, 17, 22, 24, 28, 32, 35, 50], "condition": "CTRL", "cell-line": "PCH02 Rescue"},
        {"batch": "SB63", "days": [1, 2, 4, 7, 10, 13, 17, 22, 24, 28, 32, 35, 50], "condition": "PCH", "cell-line": "PCH03"},
        {"batch": "SB64", "days": [1, 4, 7, 10, 13, 17, 22, 24, 28, 32, 35, 50], "condition": "CTRL", "cell-line": "PCH03 Rescue"}
    ]

elif project_dir == '2025_03_05-Theresa-SmallScale-Trial/QB01-QB05':
    batches_info = [
    {"batch": "QB01", "days_96": [1, 2, 4, 7, 8, 10, 12, 14],"days_24": [18, 22, 24, 28, 30, 32, 35, 50], "num_plates": 4},
    {"batch": "QB02", "days_96": [1, 2, 4, 7, 8, 10, 12, 14],"days_24": [18, 22, 24, 28, 30, 32, 35, 50], "num_plates": 3},
    {"batch": "QB03", "days_96": [1, 2, 4, 7, 8, 10, 12, 14],"days_24": [18, 22, 24, 28, 30, 32, 35, 50], "num_plates": 3},
    {"batch": "QB04", "days_96": [1, 2, 4, 7, 8, 10, 12, 14],"days_24": [18, 22, 24, 28, 30, 32, 35, 50], "num_plates": 3},
    {"batch": "QB05", "days_96": [1, 2, 4, 7, 8, 10, 12, 14],"days_24": [18, 22, 24, 28, 30, 32, 35, 50], "num_plates": 3}
    ]
    
    batches_complete_info = [  
    {"batch": "QB01", "days": [1, 2, 4, 7, 8, 10, 12, 14, 18, 22, 24, 28, 30, 32, 35, 50], "condition": "CTRL", "cell-line": "BIONi010-C"},
    {"batch": "QB02", "days": [1, 2, 4, 7, 8, 10, 12, 14, 18, 22, 24, 28, 30, 32, 35, 50], "condition": "PCH", "cell-line": "PCH02"},
    {"batch": "QB03", "days": [1, 2, 4, 7, 8, 10, 12, 14, 18, 22, 24, 28, 30, 32, 35, 50], "condition": "CTRL", "cell-line": "PCH02 Rescue"},
    {"batch": "QB04", "days": [1, 2, 4, 7, 8, 10, 12, 14, 18, 22, 24, 28, 30, 32, 35, 50], "condition": "PCH", "cell-line": "PCH03"},
    {"batch": "QB05", "days": [1, 2, 4, 7, 8, 10, 12, 14, 18, 22, 24, 28, 30, 32, 35, 50], "condition": "CTRL", "cell-line": "PCH03 Rescue"}
    ]
