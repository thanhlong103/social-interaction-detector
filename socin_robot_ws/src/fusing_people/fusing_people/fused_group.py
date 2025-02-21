import rclpy
from rclpy.node import Node
from tf_transformations import euler_from_quaternion
from people_msgs.msg import People, MyPerson, PeopleGroupArray, PeopleGroup
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
import time

from shapely.geometry import LineString, Point, Polygon
import numpy as np
from sklearn.cluster import DBSCAN


class FusedPeopleSubscriber(Node):
    def __init__(self):
        super().__init__("fused_people_subscriber")

        self.subscription = self.create_subscription(
            People,
            "/people_vision",  # Topic name
            self.fused_people_callback,
            10,  # QoS
        )

        # Publisher for PeopleGroupArray
        self.publisher = self.create_publisher(PeopleGroupArray, "/people_groups", 10)

        self.runtime_publisher = self.create_publisher(Float32, '/group_runtime', 10)

        # Apply DBSCAN clustering
        # eps is the maximum distance between two points to be considered as neighbors
        # min_samples is the minimum number of points to form a cluster
        self.dbscan = DBSCAN(eps=3.0, min_samples=2)

        self.pose_array = []
        self.line_length = 20.0
        self.interest_area = 0.5
        self.area_near_threshold = 3.0

        self.get_logger().info("Fused People Subscriber Node has started.")

    def pose_process(self, pose):
        qx = pose.pose.orientation.x
        qy = pose.pose.orientation.y
        qz = pose.pose.orientation.z
        qw = pose.pose.orientation.w

        # Convert quaternion to Euler angles (roll, pitch, yaw)
        _, _, orientation = euler_from_quaternion([qx, qy, qz, qw])

        x = pose.pose.position.x
        y = pose.pose.position.y

        return [x, y, orientation]

    def create_line(self, point):
        """
        Create a line extending from a point in the direction of orientation.
        :param point: [x, y, theta] where theta is in radians
        :param length: Length of the line segment
        :return: LineString object
        """
        x, y, theta = point
        dx = np.cos(theta) * self.line_length  # Delta x
        dy = np.sin(theta) * self.line_length  # Delta y
        start = (x, y)  # Line starts at the input point
        end = (x + dx, y + dy)  # Line ends in the direction of orientation
        return LineString([start, end])

    def find_intersections(self, lines):
        """
        Find intersections between all pairs of lines.
        :param lines: List of LineString objects
        :return: List of intersection points
        """
        intersections = []
        for i, line1 in enumerate(lines):
            for j, line2 in enumerate(lines):
                if i < j:  # Avoid duplicate pairs
                    intersection = line1.intersection(line2)
                    if isinstance(
                        intersection, Point
                    ):  # Only consider point intersections
                        intersections.append(intersection)
        return intersections

    def detect_group(self, cluster):
        lines = [self.create_line(p) for p in cluster]
        intersection_points = self.find_intersections(lines)

        if len(intersection_points) == 1:
            distances = [
                Point(pt[0], pt[1]).distance(intersection_points[0])
                for pt in cluster
            ]
            return np.mean(distances) < self.area_near_threshold, intersection_points[0], 0.0, distances[0]/2

        elif len(intersection_points) > 1:
            coords = [(pt.x, pt.y) for pt in intersection_points]
            polygon = Polygon(coords).convex_hull
            area = polygon.area

            if area < self.interest_area:
                centroid = polygon.centroid
                distances = [
                    Point(pt[0], pt[1]).distance(centroid)
                    for pt in cluster
                ]
                return np.mean(distances) < self.area_near_threshold, centroid, area, distances[0]

        return False, None, None, None

    def create_fused_person(self, person_id, pose):
        person = MyPerson()
        person.id = person_id
        person.pose.position.x = pose[0]
        person.pose.position.y = pose[1]
        person.velocity.x = 0.0  # Default velocity; update if available
        person.velocity.y = 0.0
        return person

    def fused_people_callback(self, msg):

        start_time = time.time()
        """
        Callback to process data from /people_fused and publish grouped data.
        """
        if not msg:
            self.get_logger().info("No people detected in the /people_fused topic.")
            return

        self.pose_array = [self.pose_process(pose) for pose in msg.poses]
        self.pose_array = np.array(self.pose_array)

        if len(self.pose_array) == 0:
            self.get_logger().info("No people detected in the /people_fused topic.")
            return

        # DBSCAN clustering
        labels = self.dbscan.fit_predict(self.pose_array)

        groups = {}
        for label in set(labels):
            # if label != -1:  # Ignore outliers
            groups[label] = self.pose_array[labels == label]

        people_group_array = PeopleGroupArray()
        people_group_array.header = Header()
        people_group_array.header.frame_id = "laser_frame"

        for group_id, cluster in groups.items():
            people_group = PeopleGroup()

            isGroup, interest_point, area, radius = self.detect_group(cluster)

            print(interest_point)

            # Check if the group is valid
            if isGroup:
                activities_list = []
                for i in group_indices:
                    person_act = msg.people[i].activity
                    activities_list.append(person_act)
                    if i != 0:
                        if activities_list[i] != activities_list[i-1]:
                            self.get_logger().info(f"Group {group_id} is not a valid group.")
                            return

                people_group.centroid.x = interest_point.x
                people_group.centroid.y = interest_point.y
                people_group.area = area
                people_group.radius = radius

                # Extract members of this group
                group_indices = np.where(labels == group_id)[0]

                # Populate the group with FusedPerson data
                for idx in group_indices:
                    person = msg.poses[idx]
                    fused_person = MyPerson()
                    # fused_person.id = person.id
                    fused_person.pose = person
                    # fused_person.velocity = person.velocity
                    people_group.people.append(fused_person)

                if group_id == -1:
                    group_id = 10
                # people_group.activity = activities_list[0]
                people_group.id = int(group_id) 

                people_group_array.groups.append(people_group)

                self.get_logger().info(f"Group {group_id} detected!")
            else:
                self.get_logger().info(f"Group {group_id} is not a valid group.")

        runtime = Float32()

        runtime.data = round(time.time() - start_time,3)

        self.runtime_publisher.publish(runtime)

        # Publish the PeopleGroupArray message
        self.publisher.publish(people_group_array)
        self.get_logger().info("Published PeopleGroupArray message.")

def main(args=None):
    rclpy.init(args=args)
    node = FusedPeopleSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
