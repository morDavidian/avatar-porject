import os

samples_path = os.environ.get('SAMPLES_PATH') or '.\\samples'

x_data_file = os.environ.get('X_DATA_FILE') or 'jupyter_landmarks.csv'
x_cols_start_index = os.environ.get('X_COLS_START_INDEX') or 1
x_cols_end_index = os.environ.get('X_COLS_END_INDEX') or 209

y_data_file = os.environ.get('Y_DATA_FILE') or 'unity_blendshapes.csv'
y_cols_start_index = os.environ.get('Y_COLS_START_INDEX') or 2
y_cols_end_index = os.environ.get('Y_COLS_END_INDEX') or 74

output_filename = os.environ.get('OUTPUT_FILENAME') or 'expressions.csv'

quat_domain = [-1, 1]
blend_domain = [0, 100]
blend_range = [0, 68]