import numpy as np
from typing import Optional, Tuple, List
import time
from collections import deque

class AsynchronousPredictiveKalmanFilter:
    """
    Implements the Asynchronous Predictive Kalman Filter (APKF)
    with multi-step prediction (N steps) and asynchronous measurement handling.
    Assumes a 3D constant velocity model.
    State vector x = [px, py, pz, vx, vy, vz] (Current best estimate, posterior x_0+)
    State Covariance P = Covariance of x (Posterior P_0+)
    """

    def __init__(self,
                 N: int,
                 dt: float = 0.1,
                 process_noise_std: float = 0.01,
                 measurement_noise_std: float = 0.02,
                 initial_state_estimate: Optional[np.ndarray] = None,
                 initial_estimate_covariance_diag: Optional[List[float]] = None
                 ):
        """
        Initializes the APKF.

        Args:
            N: Prediction horizon (number of steps, prediction is from 0 to N-1).
            dt: Default time step between filter updates (seconds).
            process_noise_std: Standard deviation for the process noise (acceleration uncertainty).
            measurement_noise_std: Standard deviation of the measurement noise for each pos axis (m).
            initial_state_estimate: Initial guess for the state vector [px, py, pz, vx, vy, vz]. (Optional)
            initial_estimate_covariance_diag: List of diagonal values for the initial estimate
                                              covariance matrix P. Order: [p_px..pz, p_vx..vz]. (Optional)
        """
        self.N = N
        self.dt = dt
        self.state_dim = 6
        self.meas_dim = 3

        # --- State Transition Model (Constant Velocity) ---
        self.F = np.eye(self.state_dim)
        self.F[:3, 3:] = np.eye(3) * self.dt

        # --- Measurement Model ---
        self.H = np.zeros((self.meas_dim, self.state_dim))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1

        # --- Noise Covariances ---
        # Measurement Noise Covariance R
        self.R = np.eye(self.meas_dim) * (measurement_noise_std ** 2)

        # Process Noise Covariance Q (Continuous White Noise Acceleration Model)
        # This assumes process_noise_std is the std dev of acceleration noise
        q_p = 0.5 * process_noise_std * (self.dt ** 2) # Position variance component
        q_v = process_noise_std * self.dt            # Velocity variance component
        # Simplified diagonal Q for this example structure.
        # A full CWNA Q is block-diagonal, but let's keep it simpler here.
        # More accurately Q should relate position and velocity noise based on dt^3, dt^2 etc.
        # Using diagonal Q based on approximate variances per step:
        q_diag = [q_p**2]*3 + [q_v**2]*3
        self.Q = np.diag(q_diag)
        # print("Process Noise Q:\n", self.Q)


        # --- Filter State Variables ---
        # x: Current best state estimate (posterior x_0+)
        self.x = np.zeros(self.state_dim) if initial_state_estimate is None else initial_state_estimate.copy()
        if self.x.shape != (self.state_dim,):
             raise ValueError(f"initial_state_estimate must have shape ({self.state_dim},)")

        # P: Current estimate covariance (posterior P_0+)
        self.P = np.eye(self.state_dim) if initial_estimate_covariance_diag is None else np.diag(initial_estimate_covariance_diag)
        if self.P.shape != (self.state_dim, self.state_dim):
             raise ValueError(f"Initial P from diag must be shape ({self.state_dim},{self.state_dim})")

        # --- Prediction Storage (Calculated during update) ---
        # x_star_predicted: N-step a priori state predictions [x0-, x1-, ..., x(N-1)-]
        # Initialized assuming initial x/P are posterior from step -1
        self.x_star_predicted = np.zeros(self.N * self.state_dim)
        # P_list_minus: List of N a priori covariance predictions [P0-, P1-, ..., P(N-1)-]
        self.P_list_minus: List[np.ndarray] = [np.zeros_like(self.P) for _ in range(self.N)]
        self._predict_horizon() # Perform initial prediction

        # --- Asynchronous Flags & Counters ---
        self.tslm = 0 # Time Since Last Measurement (in filter steps)
        self.measurement_processed_this_step = False

        self._update_timestamps = deque(maxlen=30)  # rolling window of timestamps
        self.update_rate_hz = None  # public attribute


    def _predict_step(self, x_in: np.ndarray, P_in: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """ Performs one prediction step. """
        F = np.eye(self.state_dim)
        F[:3, 3:] = np.eye(3) * dt # Use provided dt for transition
        x_pred = F @ x_in
        P_pred = F @ P_in @ F.T + self.Q # Add process noise Q
        return x_pred, P_pred

    def _predict_horizon(self):
        """ Predicts N steps ahead from the current posterior x, P.
            Stores results in self.x_star_predicted and self.P_list_minus.
        """
        x_curr, P_curr = self.x, self.P # Start from posterior
        for i in range(self.N):
            # Predict step i based on step i-1 (or initial for i=0)
            x_pred, P_pred = self._predict_step(x_curr, P_curr, self.dt)
            # Store as the i-th a priori prediction
            self.x_star_predicted[i*self.state_dim : (i+1)*self.state_dim] = x_pred
            self.P_list_minus[i] = P_pred
            # Use the predicted state/cov for the *next* prediction step in the horizon
            x_curr, P_curr = x_pred, P_pred

    def _update_step(self, z: np.ndarray):
        """ Performs the measurement update based on N-step predictions.
            Updates self.x and self.P (the posterior estimates).
        """
        # Use the *first* a priori prediction (step 0-) for the update
        x_priori_0 = self.x_star_predicted[0 : self.state_dim]
        P_priori_0 = self.P_list_minus[0]

        # Compute Kalman Gain (using P_priori_0, which is P1- from paper if N > 0)
        S = self.H @ P_priori_0 @ self.H.T + self.R  # Innovation covariance
        try:
            K = P_priori_0 @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        except np.linalg.LinAlgError:
            print("Warning: Innovation covariance matrix S is singular. Using pseudo-inverse.")
            K = P_priori_0 @ self.H.T @ np.linalg.pinv(S)

        # State update
        y = z - self.H @ x_priori_0  # Innovation (measurement residual)
        self.x = x_priori_0 + K @ y  # Update posterior state estimate x (x_0+)

        # Covariance update
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ P_priori_0  # Update posterior covariance P (P_0+)

    def _recalculate_predictions_after_update(self):
        """ Recalculates predictions for steps 1 to N-1 based on the updated
            posterior x (x_0+) after a measurement.
            Only updates self.x_star_predicted. Covariance prediction happens next cycle.
        """
        x_curr = self.x # Start from the new posterior x_0+
        # Update predictions in x_star_predicted for steps 1..N-1
        for i in range(1, self.N):
            # Predict state for step i based on state at step i-1
            F = np.eye(self.state_dim)
            F[:3, 3:] = np.eye(3) * self.dt
            x_curr = F @ x_curr # x_pred = F @ x_(i-1)+
            self.x_star_predicted[i*self.state_dim : (i+1)*self.state_dim] = x_curr

    def _shift_and_predict_last(self):
        """ Handles Case 3: Shifts predictions and calculates the new last step. """
        # Shift state predictions (like Eq 3.164 but for a priori)
        # x_[0..N-2]-(k) = x_[1..N-1]-(k-1)-
        self.x_star_predicted[:-self.state_dim] = self.x_star_predicted[self.state_dim:]

        # Shift covariance predictions (like Eq 3.166)
        # P_[0..N-2]-(k) = P_[1..N-1]-(k-1)-
        self.P_list_minus.pop(0) # Remove P0-(k-1)-

        # Get the state/covariance needed to predict the new last step (now N-2 step prediction)
        x_prev = self.x_star_predicted[(self.N-2)*self.state_dim : (self.N-1)*self.state_dim]
        P_prev = self.P_list_minus[-1] # Last element is now P_(N-2)-(k)

        # Predict the new last step (N-1) based on the shifted N-2 step prediction
        x_last, P_last = self._predict_step(x_prev, P_prev, self.dt)
        self.x_star_predicted[(self.N-1)*self.state_dim : ] = x_last
        self.P_list_minus.append(P_last)

        # Update posterior estimate (best guess is now the first predicted value)
        self.x = self.x_star_predicted[0 : self.state_dim].copy()
        self.P = self.P_list_minus[0].copy()


    # --- Public Methods ---

    def update(self, measurement: Optional[np.ndarray] = None):
        """
        Performs one step of the Asynchronous Predictive Kalman Filter.

        Args:
            measurement: A numpy array [px, py, pz] if a new measurement is
                         available this step, otherwise None.

        Returns:
            Tuple[np.ndarray, List[np.ndarray]]:
                - The N-step predicted states (a priori) for the *next* time step.
                - The list of N predicted covariances (a priori) for the *next* time step.
        """
        # --- Track timestamps ---
        now = time.time()
        self._update_timestamps.append(now)
        if len(self._update_timestamps) >= 2:
            intervals = np.diff(self._update_timestamps)
            avg_interval = np.mean(intervals)
            if avg_interval > 0:
                self.update_rate_hz = 1.0 / avg_interval  # public attribute

        process_measurement = measurement is not None
        self.measurement_processed_this_step = False # Reset flag for current step

        if process_measurement:
            # --- Case 1: New Measurement ---
            # Predict N steps based on *previous* posterior (self.x, self.P)
            self._predict_horizon()
            # Update posterior self.x, self.P based on measurement and step 0 prediction
            self._update_step(measurement)
            # Update the rest of the prediction vector based on the new posterior self.x
            self._recalculate_predictions_after_update()

            self.tslm = 0 # Reset time since last measurement
            self.measurement_processed_this_step = True

        elif self.tslm == 0: # This implies measurement was processed *last* step
            # --- Case 2: Measurement Acquired at Previous Iteration ---
            # Predict N steps based on *previous* posterior (self.x, self.P)
            self._predict_horizon()
            # No measurement, so posterior is just the first prediction step
            self.x = self.x_star_predicted[0 : self.state_dim].copy()
            self.P = self.P_list_minus[0].copy()

            self.tslm += 1 # Increment time since last measurement

        else:
            # --- Case 3: Measurement Acquired > 1 Iteration Ago ---
            # Shift existing predictions and predict only the new last step
            self._shift_and_predict_last()
            # self.x and self.P are updated inside _shift_and_predict_last

            self.tslm += 1 # Increment time since last measurement

        # The stored predictions are now the a priori predictions for the *next* step
        return self.x_star_predicted.copy(), [P.copy() for P in self.P_list_minus]


    def get_current_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Returns the current best estimate (posterior) of state and covariance. """
        return self.x.copy(), self.P.copy()

    def get_predicted_states(self) -> np.ndarray:
        """ Returns the N-step a priori state predictions calculated in the last update. """
        return self.x_star_predicted.copy()

    def get_predicted_covariances(self) -> List[np.ndarray]:
        """ Returns the list of N a priori covariance predictions calculated in the last update. """
        return [P.copy() for P in self.P_list_minus]

    def predict_pose_at_time(self, seconds_ahead: float) -> np.ndarray:
        """
        Predicts the position [x, y, z] of the object after a given number of seconds,
        using current state and constant velocity assumption.

        Args:
            seconds_ahead (float): Time ahead to predict (in seconds)

        Returns:
            np.ndarray: Predicted position [px, py, pz]
        """
        if self.update_rate_hz is None:
            raise RuntimeError("Update rate is not yet estimated. Wait for a few update() calls.")

        # Use latest state estimate (posterior)
        x_curr, _ = self.get_current_estimate()
        position = x_curr[0:3]
        velocity = x_curr[3:6]

        # Predict using constant velocity model: x = x0 + v*t
        future_position = position + velocity * seconds_ahead
        return future_position

# --- Example Usage (Similar to previous) ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # --- Simulation Parameters ---
    DT = 0.1       # Time step of the filter
    PREDICTION_HORIZON = 10 # Predict N=10 steps ahead
    MEASUREMENT_NOISE_STD = 0.5 # m
    PROCESS_NOISE_STD = 0.1   # std dev of acceleration noise (m/s^2)
    TOTAL_TIME = 20.0     # seconds
    MEASUREMENT_INTERVAL = 5 # Get measurement every X filter steps (e.g., 5 = 2 Hz if DT=0.1)
    OCCLUSION_START_STEP = 70
    OCCLUSION_END_STEP = 120

    # --- True System (Simulated) ---
    true_state = np.array([0.0, 5.0, 10.0, 1.0, -0.5, 0.2]) # px,py,pz, vx,vy,vz
    true_trajectory = []
    measurements = []
    measurement_times = []

    # --- Filter Initialization ---
    initial_guess = np.array([0.1, 5.1, 9.9, 0.9, -0.4, 0.25]) # Slightly off
    initial_cov_diag = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5] # Initial uncertainty

    apkf = AsynchronousPredictiveKalmanFilter(
        N=PREDICTION_HORIZON,
        dt=DT,
        process_noise_std=PROCESS_NOISE_STD,
        measurement_noise_std=MEASUREMENT_NOISE_STD,
        initial_state_estimate=initial_guess,
        initial_estimate_covariance_diag=initial_cov_diag
    )

    # --- Simulation Loop ---
    num_steps = int(TOTAL_TIME / DT)
    filter_estimates_step0_posterior = [] # Store x_0+ after update
    filter_predictions_stepN_apriori = [] # Store x_(N-1)- from prediction
    filter_time = []
    all_predicted_cov_diags_step0 = [] # Store diag(P0-)

    for k in range(num_steps):
        current_time = k * DT
        filter_time.append(current_time)

        # 1. Simulate true system advance
        F_sim = np.eye(6); F_sim[:3, 3:] = np.eye(3) * DT
        true_state = F_sim @ true_state
        # Add some simulated process noise to truth
        accel_noise = np.random.normal(0, PROCESS_NOISE_STD, 3)
        true_state[3:] += accel_noise * DT
        true_state[:3] += 0.5 * accel_noise * (DT**2)
        true_trajectory.append(true_state.copy())


        # 2. Simulate Measurement (Asynchronous)
        measurement = None
        if k % MEASUREMENT_INTERVAL == 0 and OCCLUSION_START_STEP <= k < OCCLUSION_END_STEP:
             # print(f"Step {k}: Occluded - No Measurement")
             pass # No measurement passed to filter
        elif k % MEASUREMENT_INTERVAL == 0:
            # Generate measurement with noise
            z = np.random.normal(0, MEASUREMENT_NOISE_STD, apkf.meas_dim)
            measurement = apkf.H @ true_state + z
            measurements.append(measurement)
            measurement_times.append(current_time)
            # print(f"Step {k}: Measurement Taken")


        # 3. Update Filter & Store Predicted Covariance *before* potential update
        predicted_states, predicted_covs = apkf.update(measurement)
        # Store step 0 a priori covariance diagonal
        all_predicted_cov_diags_step0.append(np.diag(predicted_covs[0]))


        # 4. Store Results (Posterior state & Last prediction)
        current_posterior_state, _ = apkf.get_current_estimate()
        filter_estimates_step0_posterior.append(current_posterior_state)

        last_prediction = predicted_states[(PREDICTION_HORIZON-1)*apkf.state_dim : PREDICTION_HORIZON*apkf.state_dim]
        filter_predictions_stepN_apriori.append(last_prediction)


    # --- Convert Results for Plotting ---
    true_trajectory = np.array(true_trajectory)
    measurements = np.array(measurements)
    filter_estimates_step0_posterior = np.array(filter_estimates_step0_posterior)
    filter_predictions_stepN_apriori = np.array(filter_predictions_stepN_apriori)
    all_predicted_cov_diags_step0 = np.array(all_predicted_cov_diags_step0)


    # --- Plotting ---
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes = ['X', 'Y', 'Z']
    pos_indices = [0, 1, 2]

    for i in range(3):
        ax = axs[i]
        ax.plot(filter_time, true_trajectory[:, pos_indices[i]], 'k-', lw=2, label=f'True {axes[i]} Pos')
        if len(measurement_times) > 0:
             ax.plot(measurement_times, measurements[:, i], 'rx', label='Measurements', markersize=6, mew=2)
        ax.plot(filter_time, filter_estimates_step0_posterior[:, pos_indices[i]], 'b--', label=f'Filtered {axes[i]} Pos (Posterior Step 0)')
        # Shift prediction time axis
        prediction_time = np.array(filter_time) + (PREDICTION_HORIZON -1) * DT
        ax.plot(prediction_time, filter_predictions_stepN_apriori[:, pos_indices[i]], 'g:', lw=2, label=f'Predicted {axes[i]} Pos (A Priori Step {PREDICTION_HORIZON-1})')

        # Plot covariance (standard deviation) for step 0 prediction
        std_dev = np.sqrt(all_predicted_cov_diags_step0[:, pos_indices[i]])
        ax.fill_between(filter_time,
                        filter_estimates_step0_posterior[:, pos_indices[i]] - std_dev,
                        filter_estimates_step0_posterior[:, pos_indices[i]] + std_dev,
                        color='blue', alpha=0.2, label=f'Filtered Pos ±1σ')


        ax.set_ylabel(f'{axes[i]} Position (m)')
        ax.legend(loc='upper left')
        ax.grid(True)
        ax.axvspan(OCCLUSION_START_STEP * DT, OCCLUSION_END_STEP * DT, color='grey', alpha=0.2, label='Occlusion Period')


    axs[-1].set_xlabel('Time (s)')
    fig.suptitle('Simplified Asynchronous Predictive KF Simulation (3D Constant Velocity)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
