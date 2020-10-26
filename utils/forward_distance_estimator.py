import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import model_from_json


class ForwardDistanceEstimator():

    def __init__(self):
        self.x_min_threshold = 400
        self.x_max_threshold = 1500

    def load_scalers(self, scaler_x_path, scaler_y_path):
        self.scaler_x = joblib.load(scaler_x_path)
        self.scaler_y = joblib.load(scaler_y_path)

    def load_model(self, path_json, path_weights):
        with open(path_json) as json_file:
            loaded_json_file = json_file.read()
        self.estimation_model = model_from_json(loaded_json_file)
        self.estimation_model.load_weights(path_weights)
        self.estimation_model.compile(loss='mean_squared_error', optimizer='adam')

    def predict_distance(self, x_min, y_min, x_max, y_max):
        scaled_bbox = self.scaler_x.transform(np.array([x_min, y_min, x_max, y_max]).reshape(1, -1))
        y_pred = self.estimation_model.predict(scaled_bbox)
        dist = self.scaler_y.inverse_transform(y_pred)
        return dist
