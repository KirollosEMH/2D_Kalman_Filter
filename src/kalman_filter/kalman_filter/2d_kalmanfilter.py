import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
from numpy.linalg import inv

class KalmanFilter(Node):
    def __init__(self):
        super().__init__('kalman_filter_node')

        # Initialize Kalman Filter variables
        self.state = np.zeros(2)
        self.state_covariance = np.eye(2)
        self.process_noise = np.diag([0.01, 0.01])
        self.measurement_noise = np.diag([0.1, 0.1])

        # Subscribe to the /odom_noise topic
        self.subscription = self.create_subscription(Odometry,
                                                     '/odom_noise',
                                                     self.odom_callback,
                                                     1)
        
        # Publish the estimated reading
        self.estimated_pub = self.create_publisher(Odometry,
                                                   "/odom_estimated",
                                                   1)

    def odom_callback(self, msg):
        # Extract the position measurements from the Odometry message
        measured_position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

        # Prediction step
        
        A = np.array([[1, 0], [0, 1]])
        B = np.array([[1, 0], [0, 1]])
        control_input = np.zeros(2)
        
        # Predict the next state
        predicted_state = np.dot(A, self.state) + np.dot(B, control_input)
        
        # Update the state covariance
        self.state_covariance = np.dot(np.dot(A, self.state_covariance), A.T) + self.process_noise

        # Update step
        H = np.array([[1, 0], [0, 1]])
        residual = measured_position - np.dot(H, predicted_state)
        residual_covariance = np.dot(np.dot(H, self.state_covariance), H.T) + self.measurement_noise
        kalman_gain = np.dot(np.dot(self.state_covariance, H.T), inv(residual_covariance))
        self.state = predicted_state + np.dot(kalman_gain, residual)
        self.state_covariance = np.dot((np.eye(2) - np.dot(kalman_gain, H)), self.state_covariance)

        # Publish the estimated reading
        estimated_odom = Odometry()
        estimated_odom.header = msg.header
        estimated_odom.pose.pose.position.x = self.state[0]
        estimated_odom.pose.pose.position.y = self.state[1]
        self.estimated_pub.publish(estimated_odom)

def main(args=None):
    rclpy.init(args=args)
    node = KalmanFilter()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
