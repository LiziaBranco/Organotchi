from utils import organoid_segmentation_inference, analyze_organoid_morphology

checkpoint = 'Training-Results/work_dirs/split-0/iter_144000.pth'
base_dir = '../../../AGMayer/Lizia/Organoid_Segmentation'

project_dir = '2025_03_05-Dasha-DMA'
project_dir = '2025_03_05-Theresa-SmallScale-Trial'
project_dir = '2025_03_05-Quality-Prediction'

# Get all projects dynamically from batch_dir

days = [1, 2, 3, 4, 6, 7, 8]
days = [10]

# Pixel-to-micron conversion factors
pixel_to_micron_x = 0.3798 #Dasha Campus Nord
pixel_to_micron_y = 0.3798

pixel_to_micron_x = 0.8872 #Our lab 4x objective
pixel_to_micron_y= 0.8874

plates = ['AB108', 'QB01_1', 'QB01_2', "SB60"]

organoid_segmentation_inference(checkpoint, base_dir, project_dir, days, plates)
analyze_organoid_morphology(base_dir, project_dir, days, plates, pixel_to_micron_x, pixel_to_micron_y)