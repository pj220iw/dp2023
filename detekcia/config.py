"""
#################################
# Configuration File
#################################
"""
pathVid_fire = 'video/2-Zenmuse_X4S_2.mp4'
pathVid_LakeMary = 'video/Matrice 200_phone_lakemary_X4S/Final.mp4'
pathVid_test_Fire = 'video/Testdata/fire.mp4'
pathVid_test_NoFire = 'video/Testdata/No_fire.mp4'

pathFrame_all = 'frames/all'
pathFrame_resize = 'frames/resize'

pathFrame_lakemary = 'frames/lakemary'
pathFrame_resize_lakemary = 'frames/resize_lakemary'

pathFrame_test = 'frames/Test_frame/'
pathFrame_resize_test = 'frames/Test/'

Flags = {'playVideoFlag': True, 'SaveRawFrameFlag': False, 'ResizeFlag': False, 'plot_center': True,
         'Debug_print': False}
new_size = {'width': 256, 'height': 256}
segmentation_new_size = {'width': 512, 'height': 512}
Config_classification = {"batch_size": 32, 'Save_Model': True, 'Epochs': 10, "TrainingPlot": True}
config_segmentation = {"batch_size": 16, 'Save_Model': False, 'Epochs': 1, "TrainingPlot": False,
                       "train_set_ratio": 0.85, "val_set_ratio": 0.15, "num_class": 2, "CHANNELS": 3}
Mode = 'Predikcia'
# Different Modes {"Fire", "Lake_Mary", "Test_Frame", "Training", "Classification", "Segmentation","Predikcia", "Resize",
# 'Video_to_frames}

config_uav = {"Num_uav": 2, "Init_flight_time": 40, 'Observation_time': 2, 'Observation_interval': 20,
              "Charge_time": 35, "Speed": 150, "Num_pile_fire": 5, "Event": 100, "uav_limit": 10}
