import rclpy
from rclpy.node import Node
import cv2
import tensorflow as tf
import numpy as np
import pyrealsense2 as rs
from tensorflow.keras.models import load_model
import math
import time
import pandas as pd
from tf_transformations import quaternion_from_euler
from geometry_msgs.msg import PoseArray, Pose
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

class VisionLegTracker(Node):
    def __init__(self):
        super().__init__("vision_leg_tracker_node")
        self.pipe = rs.pipeline()
        self.cfg = rs.config()

        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipe)
        self.pipeline_profile = self.cfg.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(
            self.device.get_info(rs.camera_info.product_line)
        )

        self.cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.label = "Warmup...."
        self.n_time_steps = 10
        self.lm_list = [[], [], [], [], [], []]
        self.prevAct = time.time()
        self.i = 0
        self.warmup_frames = 60

        # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.pipe.start(self.cfg)
        # Initialize prevtime for runtime calculation
        self.prevtime = time.time()

        # used to record the time when we processed last frame
        self.prev_frame_time = 0

        # used to record the time at which we processed current frame
        self.new_frame_time = 0

        self.fpsArr = []
        self.sumfps = 0

        self.confidence_threshold = 0.25
        self.bbox_threshold = 0.125

        self.EDGES = {
            (0, 1): "m",
            (0, 2): "c",
            (1, 3): "m",
            (2, 4): "c",
            (0, 5): "m",
            (0, 6): "c",
            (5, 7): "m",
            (7, 9): "m",
            (6, 8): "c",
            (8, 10): "c",
            (5, 6): "y",
            (5, 11): "m",
            (6, 12): "c",
            (11, 12): "y",
            (11, 13): "m",
            (13, 15): "m",
            (12, 14): "c",
            (14, 16): "c",
        }

        self.input_size = 128

        self.offset_x = 150.0
        self.offset_y = 37.0
        self.offset_z = 0.0
        self.offset = [self.offset_x, self.offset_y, self.offset_z]
        self.intrinsics = None

        self.prev_person_pos = [0.0, 0.0, 0.0]
        self.prev_person_marker_ids = set()

        try:
            self.interpreter = tf.lite.Interpreter(model_path="1.tflite")
            # self.interpreter = tf.lite.Interpreter(model_path="1.tflite")
            self.interpreter.allocate_tensors()
            self.get_logger().info("MoveNet loaded!")
        except:
            print("Can not access the MoveNet!")

        try:
            self.model = tf.saved_model.load('./mymodel')
        except:
            print("Can not access HAR model")

        self.get_logger().info("Vision Leg Tracker Node has started.")

        # Add a publisher for person coordinates
        self.coord_publisher = self.create_publisher(
            PoseArray, "/people_vision", 10
        )
        self.marker_publisher = self.create_publisher(MarkerArray, "/human_markers", 10)

        # Define the static transform broadcaster
        self.tf_broadcaster = StaticTransformBroadcaster(self)

        # Define the camera frame's static transform
        self.broadcast_camera_frame()

        # Frame width and height
        frame = self.pipe.wait_for_frames()
        color_frame = frame.get_color_frame()
        img = np.asanyarray(color_frame.get_data())
        self.HEIGHT, self.WIDTH, _ = img.shape

        # Align object - aligning depth to color
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

    def broadcast_camera_frame(self):
        # Create a TransformStamped message
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "base_link"  # Parent frame
        t.child_frame_id = "camera_frame"  # Camera frame

        # Define translation (position) of the camera
        t.transform.translation.x = -0.2
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.2

        # Define rotation (orientation) of the camera
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        # Broadcast the static transform
        self.tf_broadcaster.sendTransform(t)
        self.get_logger().info("Camera frame transform broadcasted.")

    def publish_human_marker(self, marker_array, id, x, y):
        # Create a marker for the sphere (human's head)
        sphere_marker = Marker()
        sphere_marker.header.frame_id = "camera_frame"
        sphere_marker.header.stamp = self.get_clock().now().to_msg()
        sphere_marker.ns = "humans"
        sphere_marker.id = id * 2
        sphere_marker.type = Marker.SPHERE
        sphere_marker.action = Marker.ADD
        sphere_marker.pose.position.x = x
        sphere_marker.pose.position.y = -y
        sphere_marker.pose.position.z = 0.4  # Sphere above the cylinder
        sphere_marker.scale.x = 0.15
        sphere_marker.scale.y = 0.15
        sphere_marker.scale.z = 0.15
        sphere_marker.color.a = 1.0  # Alpha (transparency)
        sphere_marker.color.r = 0.0
        sphere_marker.color.g = 1.0
        sphere_marker.color.b = 0.0  # Green color

        # Create a marker for the cylinder (human's body)
        cylinder_marker = Marker()
        cylinder_marker.header.frame_id = "camera_frame"
        cylinder_marker.header.stamp = self.get_clock().now().to_msg()
        cylinder_marker.ns = "humans"
        cylinder_marker.id = id * 2 + 1000
        cylinder_marker.type = Marker.CYLINDER
        cylinder_marker.action = Marker.ADD
        cylinder_marker.pose.position.x = x
        cylinder_marker.pose.position.y = -y
        cylinder_marker.pose.position.z = 0.2  # Cylinder's center
        cylinder_marker.scale.x = 0.1  # Diameter
        cylinder_marker.scale.y = 0.1  # Diameter
        cylinder_marker.scale.z = 0.4  # Height of the cylinder
        cylinder_marker.color.a = 1.0  # Alpha (transparency)
        cylinder_marker.color.r = 0.0
        cylinder_marker.color.g = 0.5
        cylinder_marker.color.b = 1.0  # Blue color

        marker_array.markers.append(sphere_marker)
        marker_array.markers.append(cylinder_marker)

        return marker_array

    def draw(self, frame, keypoints, bbox):
        if bbox[4] > self.bbox_threshold:
            startpoint = (int(bbox[1] * self.WIDTH), int(bbox[0] * self.HEIGHT))
            endpoint = (int(bbox[3] * self.WIDTH), int(bbox[2] * self.HEIGHT))
            thickness = 2

            # Blue color in BGR
            color = (255, 0, 0)
            cv2.rectangle(frame, startpoint, endpoint, color, thickness)

        if keypoints.shape[0] == 51:
            keypoints_k = keypoints.reshape(-1, 3)

        for kp in keypoints_k:
            ky, kx, kp_conf = kp

            # Draw the keypoint if confidence is above the threshold
            if kp_conf >= self.confidence_threshold:
                cv2.circle(
                    frame,
                    (int(kx * self.WIDTH), int(ky * self.HEIGHT)),
                    7,
                    (0, 255, 0),
                    -1,
                )

        for edge, _ in self.EDGES.items():
            p1, p2 = edge
            kp1 = keypoints_k[p1]
            kp2 = keypoints_k[p2]

            # Unpack the coordinates and confidence for both keypoints
            y1, x1, c1 = kp1
            y2, x2, c2 = kp2

            # Draw the connection if both points have a confidence above the threshold
            if c1 > self.confidence_threshold and c2 > self.confidence_threshold:
                cv2.line(
                    frame,
                    (int(x1 * self.WIDTH), int(y1 * self.HEIGHT)),
                    (int(x2 * self.WIDTH), int(y2 * self.HEIGHT)),
                    (255, 0, 0),
                    2,
                )

    def estimator(self, frame):
        img = frame.copy()
        img = tf.expand_dims(img, axis=0)
        resized_image, _ = self.keep_aspect_ratio_resizer(img, self.input_size)
        image_tensor = tf.cast(resized_image, dtype=tf.uint8)

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        is_dynamic_shape_model = input_details[0]["shape_signature"][2] == -1

        if is_dynamic_shape_model:
            input_tensor_index = input_details[0]["index"]
            input_shape = image_tensor.shape
            self.interpreter.resize_tensor_input(
                input_tensor_index, input_shape, strict=True
            )

        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(input_details[0]["index"], image_tensor.numpy())

        self.interpreter.invoke()

        keypoints_with_scores = self.interpreter.get_tensor(output_details[0]["index"])

        return keypoints_with_scores

    def keep_aspect_ratio_resizer(self, image, target_size):
        _, height, width, _ = image.shape
        scale = float(target_size / width)
        target_width = target_size
        scaled_height = math.ceil(height * scale)
        image = tf.image.resize(image, [scaled_height, target_width])
        target_height = int(math.ceil(scaled_height / 32) * 32)
        image = tf.image.pad_to_bounding_box(image, 0, 0, target_height, target_width)
        return (image, (target_height, target_width))

    def process_keypoints(self, intrinsics, keypoints, depth_array, depth_frame):
        nice_keypoints = []

        for i in range(0, len(keypoints), 3):
            y, x, confidence = keypoints[i], keypoints[i + 1], keypoints[i + 2]
            if confidence > self.confidence_threshold:
                # Scale the coordinates
                keypoints[i] = x * self.WIDTH
                keypoints[i + 1] = y * self.HEIGHT

                # Validate indices to avoid IndexError
                row = int(keypoints[i])
                col = int(keypoints[i + 1])

                patch_size = 5
                half_size = patch_size // 2
                positions = [(row, col)]

                for j in range(row - half_size, row + half_size + 1):
                    for k in range(col - half_size, col + half_size + 1):
                        positions.append((j, k))

                if 0 <= row < depth_array.shape[1] and 0 <= col < depth_array.shape[0]:
                    try:
                        depth_list = []
                        for x in range(len(positions)):
                            # print(positions[x][0])
                            current_depth = depth_frame.get_distance(
                                positions[x][0], positions[x][1]
                            )  # Access depth value
                            depth_list.append(current_depth)

                        depth = np.min(depth_list)
                        coordinate_camera = rs.rs2_deproject_pixel_to_point(
                            intrinsics, [keypoints[i], keypoints[i + 1]], depth
                        )
                        if (coordinate_camera[2] < 3) | (coordinate_camera[2] > 0.2):
                            # Remapping from camera to world
                            keypoints[i] = coordinate_camera[0]
                            keypoints[i + 1] = coordinate_camera[1]
                            keypoints[i + 2] = coordinate_camera[2]

                            xyz_keypoint = [
                                keypoints[i + 2],
                                keypoints[i],
                                keypoints[i + 1],
                            ]
                            nice_keypoints.append(xyz_keypoint)
                    except:
                        # Set default values or handle the out-of-bounds case
                        keypoints[i] = 0
                        keypoints[i + 1] = 0
                        keypoints[i + 2] = 0
                else:
                    # Set default values or handle the out-of-bounds case
                    keypoints[i] = 0
                    keypoints[i + 1] = 0
                    keypoints[i + 2] = 0

        return nice_keypoints

    def estimate_plane_pca(self, points):
        """
        Estimates a plane from a given set of 3D points using PCA.

        Parameters:
            points (numpy.ndarray): A (N, 3) array of 3D points.

        Returns:
            plane_normal (numpy.ndarray): A (3,) vector representing the plane's normal.
            plane_point (numpy.ndarray): A (3,) vector representing a point on the plane (centroid).
        """
        # points = gaussian_filter(points, sigma=1)

        try:
            # Compute the centroid (mean) of the points
            centroid = np.mean(points, axis=0)

            # Center the points by subtracting the centroid
            centered_points = points - centroid

            # Compute the covariance matrix
            covariance_matrix = np.cov(centered_points.T)

            # Perform eigen decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

            # The normal vector to the plane is the eigenvector associated with the smallest eigenvalue
            normal_vector = eigenvectors[:, np.argmin(eigenvalues)]
        except:
            normal_vector = [0, 0, 0]
            centroid = [0, 0, 0]

        return normal_vector, centroid

    def facing_direction(self, normal_vector, centroid):
        A, B, C = normal_vector
        D = np.dot(normal_vector, centroid)

        # Plane equation
        # print(f"Plane equation: {A:.4f}x + {B:.4f}y + {C:.4f}z + {D:.4f} = 0")

        normal_perpendicular = np.array([1, 0, 0])

        # print("NORMAL VECTOR:", normal )

        dot_product = np.dot(normal_vector, normal_perpendicular)

        # Calculate the magnitudes of the normal vectors
        magnitude_estimated = np.linalg.norm(normal_vector)
        magnitude_perpendicular = np.linalg.norm(normal_perpendicular)

        # Calculate the cosine of the angle
        cos_angle = dot_product / (magnitude_estimated * magnitude_perpendicular)

        # Calculate the angle in radians
        theta = np.arccos(cos_angle)

        # Convert the angle to degrees
        angle_degrees = np.degrees(theta)

        theta = np.radians(angle_degrees)

        if B < 0:
            theta = theta
        else:
            theta = 6.28 - theta

        return theta

    def processImage(self):
        current_ids = set()

        frame = self.pipe.wait_for_frames()

        aligned_frames = self.align.process(frame)

        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        img = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # img = cv2.medianBlur(img, 3)

        # Convert depth image to 3-channel grayscale for visualization
        depth_visual = cv2.normalize(
            depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        depth_visual = cv2.cvtColor(depth_visual, cv2.COLOR_GRAY2BGR)

        # Initialize camera intrinsics if not done
        if self.intrinsics is None:
            self.intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        keypoints_with_scores = self.estimator(img)

        # Localization to get the world coordinates
        person_world_coords = []
        poses = PoseArray()
        marker_array = MarkerArray()

        for i in range(6):
            bbox = keypoints_with_scores[0][i][51:57]
            keypoints_draw = keypoints_with_scores[0][i][:51]

            # if (keypoints_draw[17] > self.confidence_threshold):
            right_shoulder = tuple(
                [
                    int(keypoints_draw[16] * self.WIDTH),
                    int(keypoints_draw[15] * self.HEIGHT),
                ]
            )
            # else:
            #     right_shoulder = (20,0)
            # if (keypoints_draw[20] > self.confidence_threshold):
            left_shoulder = tuple(
                [
                    int(keypoints_draw[19] * self.WIDTH),
                    int(keypoints_draw[18] * self.HEIGHT),
                ]
            )

            # Rendering
            self.draw(depth_visual, keypoints_draw, bbox)
            self.draw(img, keypoints_draw, bbox)

            keypoints = self.process_keypoints(
                self.intrinsics, keypoints_draw, depth_image, depth_frame
            )

            # print(keypoints)

            if bbox[4] > self.bbox_threshold:
                # print(left_shoulder, right_shoulder)
                cv2.rectangle(
                    img, left_shoulder, right_shoulder, (0, 255, 0), 2
                )  # Fill with blue

                normal, centroid = self.estimate_plane_pca(keypoints)

                theta = self.facing_direction(normal, centroid)

                # print(theta)

                if left_shoulder[0] - right_shoulder[0] > 20:
                    theta = theta + 3.14

                x = float(centroid[0])
                y = float(centroid[1])

                self.prev_person_pos = [x, y, theta]

                person_world_coords.append([x, y, theta])
                # Add to PoseArray
                pose = Pose()
                pose.position.x = x
                pose.position.y = -y
                pose.position.z = 0.03
                q = quaternion_from_euler(0, 0, theta + 3.14)
                pose.orientation.x = q[0]
                pose.orientation.y = q[1]
                pose.orientation.z = q[2]
                pose.orientation.w = q[3]

                # pose.orientation.z = theta
                poses.poses.append(pose)

                current_ids.add(i)

                marker_array = self.publish_human_marker(marker_array, i, x, y)

                # self.get_logger().info(
                #     f"Pose -> x: {x:.2f}, y: {y:.2f}, theta: {theta:.2f} rad ({math.degrees(theta):.2f} deg)"
                # )
            
        # Add delete markers for people no longer detected
        for prev_id in self.prev_person_marker_ids - current_ids:
            delete_sphere_marker = Marker()
            delete_sphere_marker.header.frame_id = "base_laser"
            delete_sphere_marker.header.stamp = self.get_clock().now().to_msg()
            delete_sphere_marker.ns = "humans"
            delete_sphere_marker.id = prev_id
            delete_sphere_marker.action = Marker.DELETE

            delete_cylinder_marker = Marker()
            delete_cylinder_marker.header.frame_id = "base_laser"
            delete_cylinder_marker.header.stamp = self.get_clock().now().to_msg()
            delete_cylinder_marker.ns = "humans"
            delete_cylinder_marker.id = prev_id + 1000
            delete_cylinder_marker.action = Marker.DELETE

            marker_array.markers.append(delete_sphere_marker)
            marker_array.markers.append(delete_cylinder_marker)
        
        self.prev_person_marker_ids = current_ids

        # Publish the PoseArray if there are valid coordinates
        if poses.poses:
            poses.header.frame_id = "camera_frame"  # Replace with your camera frame
            poses.header.stamp = self.get_clock().now().to_msg()
            self.coord_publisher.publish(poses)
            # Publish MarkerArray
            self.marker_publisher.publish(marker_array)

        # Display FPS
        font = cv2.FONT_HERSHEY_SIMPLEX
        self.new_frame_time = time.time()
        fps = 1 / (self.new_frame_time - self.prev_frame_time)
        self.prev_frame_time = self.new_frame_time
        fps_str = "FPS: " + str(round(fps, 2))

        # Update fpsArr and sumfps
        self.fpsArr.append(fps)
        sumfps = sum(self.fpsArr)
        fpsAvg = sumfps / len(self.fpsArr)

        if len(self.fpsArr) == 10:  # Reset every 10 frames
            # print(f"Avg FPS: {fpsAvg}")
            self.fpsArr = []
            sumfps = 0

        cv2.putText(img, fps_str, (455, 30), font, 1, (100, 255, 0), 3, cv2.LINE_AA)

        # Stack both images horizontally
        images = np.hstack((img, depth_visual))

        cv2.imshow("People Detected", images)

        if cv2.waitKey(1) == ord("q"):
            self.pipe.stop()
            cv2.destroyAllWindows()
            rclpy.shutdown()

    def run(self):
        while rclpy.ok():
            self.processImage()


def main(args=None):
    rclpy.init(args=args)
    node = VisionLegTracker()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
