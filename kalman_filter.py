import numpy as np
from filterpy.common import Q_discrete_white_noise

class KalmanFilter:
    def __init__(self, first_prediction: tuple[int, int], second_prediction: tuple[int, int]):
        # [2.x, 2.y, 2.x - 1.x, 2.y - 1.y]
        self.state_vector = np.array([second_prediction[0], second_prediction[1], second_prediction[0] - first_prediction[0], second_prediction[1] - first_prediction[1]])
        self.new_state_vector = np.array(0)
        dt = 1.0
        # State Transition Matrix (F)
        self.F = np.array([[1, 0, dt,  0],
                           [0, 1,  0, dt],
                           [0, 0,  1,  0],
                           [0, 0,  0,  1]])

        # Observation Matrix (H)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # Estimate Covariance (P)
        self.P = np.eye(4) * 500
        # Measurement Noise (R)
        self.R = np.eye(2) * 5
        # Process Noise (Q)
        self.Q = Q_discrete_white_noise(2, dt, 0.1, 2)

    def predict_state(self) -> None:
        self.new_state_vector = self.F @ self.state_vector

    def update_covariance(self):
        self.P = self.F @ self.P @ self.F.T + self.Q

    def predict(self, ml_prediction: tuple[int, int] | None) -> tuple[int, int]:
        if ml_prediction is None:
            self.state_vector = self.new_state_vector
            return int(self.state_vector[0]), int(self.state_vector[1])

        # measurement_residual
        y = ml_prediction - (self.H @ self.new_state_vector)
        # innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        # kalam gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # final prediction
        final = self.new_state_vector + K @ y
        # estimate covariance
        I = np.eye(4)
        self.P = (I - (K @ self.H)) @ self.P

        self.state_vector = final
        return final[0], final[1]





