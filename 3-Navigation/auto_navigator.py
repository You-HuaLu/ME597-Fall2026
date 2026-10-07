#!/usr/bin/env python3

import sys
import os
import heapq
import math
import numpy as np
import yaml
from PIL import Image

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose, Twist
from std_msgs.msg import Float32


class Navigation(Node):
    """! Navigation node class.
    Plans a global path with A* on the saved occupancy-grid map and follows it
    with a simple look-ahead path follower. No Nav2 is used.
    """

    def __init__(self, node_name='Navigation'):
        """! Class constructor.
        @param  None.
        @return An instance of the Navigation class.
        """
        super().__init__(node_name)
        # Parameters
        self.declare_parameter('map_yaml', 'src/task_4/maps/classroom_map.yaml')
        self.declare_parameter('inflation_m', 0.20)      # safety margin around walls [m]
        self.declare_parameter('lookahead_m', 0.30)      # distance ahead on the path to aim at [m]
        self.declare_parameter('goal_tol_m', 0.10)       # stop when this close to the goal [m]
        self.declare_parameter('max_speed', 0.20)        # [m/s]
        self.declare_parameter('max_turn', 1.0)          # [rad/s]
        self.declare_parameter('k_heading', 1.5)         # P gain on yaw error

        # Path planner/follower related variables
        self.path = Path()
        self.goal_pose = PoseStamped()
        self.ttbot_pose = PoseStamped()
        self.start_time = 0.0
        self.new_goal = False
        self.have_pose = False
        self.at_goal = True

        self.__load_map(self.get_parameter('map_yaml').value,
                        self.get_parameter('inflation_m').value)

        # Subscribers
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.__goal_pose_cbk, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.__ttbot_pose_cbk, 10)

        # Publishers
        self.path_pub = self.create_publisher(Path, 'global_plan', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.calc_time_pub = self.create_publisher(Float32, 'astar_time',10) #DO NOT MODIFY

    # ------------------------------------------------------------------ map
    def __load_map(self, yaml_path, inflation_m):
        """! Read the .yaml/.pgm pair and build an inflated free-space grid."""
        with open(yaml_path, 'r') as f:
            meta = yaml.safe_load(f)
        img_path = meta['image']
        if not os.path.isabs(img_path):
            img_path = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), img_path)
        img = np.array(Image.open(img_path).convert('L'), dtype=np.float64)
        if meta.get('negate', 0):
            img = 255.0 - img
        self.res = float(meta['resolution'])
        self.origin = (float(meta['origin'][0]), float(meta['origin'][1]))
        # Same rule as the notebook: bright pixels are free, everything else is blocked.
        free = img > float(meta['occupied_thresh']) * 255.0
        self.grid = self.__inflate(~free, int(math.ceil(inflation_m / self.res)))
        self.get_logger().info('Map loaded: {}x{} cells, res {} m'.format(
            self.grid.shape[0], self.grid.shape[1], self.res))

    @staticmethod
    def __inflate(blocked, r):
        """! Dilate blocked cells by r cells (square kernel). True = blocked."""
        if r <= 0:
            return blocked.copy()
        padded = np.pad(blocked, r, constant_values=False)
        out = np.zeros_like(blocked)
        h, w = blocked.shape
        for di in range(2 * r + 1):
            for dj in range(2 * r + 1):
                out |= padded[di:di + h, dj:dj + w]
        return out

    def __world_to_grid(self, x, y):
        """! (x, y) in the map frame -> (row, col). Row 0 is the top of the image."""
        h = self.grid.shape[0]
        col = int((x - self.origin[0]) / self.res)
        row = h - 1 - int((y - self.origin[1]) / self.res)
        return row, col

    def __grid_to_world(self, row, col):
        h = self.grid.shape[0]
        x = self.origin[0] + (col + 0.5) * self.res
        y = self.origin[1] + (h - 1 - row + 0.5) * self.res
        return x, y

    def __nearest_free(self, cell):
        """! If a cell is inside the inflated walls, move to the closest free one."""
        h, w = self.grid.shape
        r0, c0 = cell
        r0 = min(max(r0, 0), h - 1)
        c0 = min(max(c0, 0), w - 1)
        if not self.grid[r0, c0]:
            return (r0, c0)
        free = np.argwhere(~self.grid)
        if len(free) == 0:
            return None
        d = (free[:, 0] - r0) ** 2 + (free[:, 1] - c0) ** 2
        r, c = free[int(np.argmin(d))]
        return (int(r), int(c))

    # ------------------------------------------------------------ callbacks
    def __goal_pose_cbk(self, data):
        """! Callback to catch the goal pose.
        @param  data    PoseStamped object from RVIZ.
        @return None.
        """
        self.goal_pose = data
        self.new_goal = True
        self.get_logger().info(
            'goal_pose: {:.4f}, {:.4f}'.format(self.goal_pose.pose.position.x, self.goal_pose.pose.position.y))

    def __ttbot_pose_cbk(self, data):
        """! Callback to catch the position of the vehicle.
        @param  data    PoseWithCovarianceStamped object from amcl.
        @return None.
        """
        self.ttbot_pose = PoseStamped()
        self.ttbot_pose.header = data.header
        self.ttbot_pose.pose = data.pose.pose
        self.have_pose = True

    # -------------------------------------------------------------- planner
    def a_star_path_planner(self, start_pose, end_pose):
        """! A Start path planner.
        @param  start_pose    PoseStamped object containing the start of the path to be created.
        @param  end_pose      PoseStamped object containing the end of the path to be created.
        @return path          Path object containing the sequence of waypoints of the created path.
        """
        path = Path()
        self.get_logger().info(
            'A* planner.\n> start: {},\n> end: {}'.format(start_pose.pose.position, end_pose.pose.position))
        self.start_time = self.get_clock().now().nanoseconds*1e-9 #Do not edit this line (required for autograder)
        cells = self.__astar_cells(
            self.__world_to_grid(start_pose.pose.position.x, start_pose.pose.position.y),
            self.__world_to_grid(end_pose.pose.position.x, end_pose.pose.position.y))
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        for (r, c) in cells:
            x, y = self.__grid_to_world(r, c)
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        # Do not edit below (required for autograder)
        self.astarTime = Float32()
        self.astarTime.data = float(self.get_clock().now().nanoseconds*1e-9-self.start_time)
        self.calc_time_pub.publish(self.astarTime)

        return path

    def __astar_cells(self, start, goal):
        """! 8-connected A* on the inflated grid using a heapq open list.
        @return list of (row, col) from start to goal, or [] if there is no path.
        """
        start = self.__nearest_free(start)
        goal = self.__nearest_free(goal)
        if start is None or goal is None:
            return []
        h, w = self.grid.shape
        blocked = self.grid
        sqrt2 = math.sqrt(2.0)
        moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, sqrt2), (-1, 1, sqrt2), (1, -1, sqrt2), (1, 1, sqrt2)]
        gr, gc = goal
        g = {start: 0.0}
        parent = {}
        closed = set()
        open_heap = [(math.hypot(gr - start[0], gc - start[1]), 0, start)]
        counter = 0
        while open_heap:
            __, __, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            if cur == goal:
                break
            closed.add(cur)
            r, c = cur
            for dr, dc, cost in moves:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= h or nc < 0 or nc >= w or blocked[nr, nc]:
                    continue
                # do not cut corners between two walls
                if dr != 0 and dc != 0 and (blocked[r + dr, c] or blocked[r, c + dc]):
                    continue
                nxt = (nr, nc)
                ng = g[cur] + cost
                if ng < g.get(nxt, math.inf):
                    g[nxt] = ng
                    parent[nxt] = cur
                    counter += 1
                    heapq.heappush(open_heap, (ng + math.hypot(gr - nr, gc - nc), counter, nxt))
        if goal not in parent and goal != start:
            return []
        cells = [goal]
        while cells[-1] != start:
            cells.append(parent[cells[-1]])
        cells.reverse()
        return cells

    # ------------------------------------------------------------- follower
    @staticmethod
    def __yaw(pose):
        q = pose.pose.orientation
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def get_path_idx(self, path, vehicle_pose):
        """! Path follower.
        @param  path                  Path object containing the sequence of waypoints of the created path.
        @param  vehicle_pose     PoseStamped object containing the current vehicle position.
        @return idx                   Position in the path pointing to the next goal pose to follow.
        """
        pts = np.array([[p.pose.position.x, p.pose.position.y] for p in path.poses])
        vx, vy = vehicle_pose.pose.position.x, vehicle_pose.pose.position.y
        nearest = int(np.argmin((pts[:, 0] - vx) ** 2 + (pts[:, 1] - vy) ** 2))
        look = self.get_parameter('lookahead_m').value
        idx = nearest
        while idx < len(pts) - 1 and math.hypot(pts[idx, 0] - vx, pts[idx, 1] - vy) < look:
            idx += 1
        return idx

    def path_follower(self, vehicle_pose, current_goal_pose):
        """! Path follower.
        @param  vehicle_pose           PoseStamped object containing the current vehicle pose.
        @param  current_goal_pose      PoseStamped object containing the current target from the created path. This is different from the global target.
        @return speed, heading         Desired forward speed and desired yaw angle [rad].
        """
        dx = current_goal_pose.pose.position.x - vehicle_pose.pose.position.x
        dy = current_goal_pose.pose.position.y - vehicle_pose.pose.position.y
        heading = math.atan2(dy, dx)
        err = math.atan2(math.sin(heading - self.__yaw(vehicle_pose)),
                         math.cos(heading - self.__yaw(vehicle_pose)))
        # Slow down while turning; rotate in place if the target is far off to the side.
        max_speed = self.get_parameter('max_speed').value
        speed = max_speed * max(0.0, math.cos(err)) if abs(err) < math.radians(60) else 0.0
        return speed, heading

    def move_ttbot(self, speed, heading):
        """! Function to move turtlebot passing directly a heading angle and the speed.
        @param  speed     Desired speed.
        @param  heading   Desired yaw angle.
        @return None.
        """
        cmd_vel = Twist()
        err = math.atan2(math.sin(heading - self.__yaw(self.ttbot_pose)),
                         math.cos(heading - self.__yaw(self.ttbot_pose)))
        max_turn = self.get_parameter('max_turn').value
        cmd_vel.linear.x = float(speed)
        cmd_vel.angular.z = float(max(-max_turn, min(max_turn, self.get_parameter('k_heading').value * err)))

        self.cmd_vel_pub.publish(cmd_vel)

    def stop_ttbot(self):
        self.cmd_vel_pub.publish(Twist())

    def run(self):
        """! Main loop of the node. Wait for a pose and a goal, plan once per new goal,
        then drive along the path until the goal is reached.
        @param none
        @return none
        """
        while rclpy.ok():
            # Call the spin_once to handle callbacks (also paces the loop at ~10 Hz)
            rclpy.spin_once(self, timeout_sec=0.1)

            if not self.have_pose:
                continue

            # 1. Create the path to follow (only when a new goal arrives)
            if self.new_goal:
                self.new_goal = False
                self.path = self.a_star_path_planner(self.ttbot_pose, self.goal_pose)
                if len(self.path.poses) == 0:
                    self.get_logger().warn('No path found to the goal.')
                    self.at_goal = True
                    self.stop_ttbot()
                    continue
                self.path_pub.publish(self.path)
                self.at_goal = False

            if self.at_goal:
                continue

            # 2. Loop through the path and move the robot
            gx = self.path.poses[-1].pose.position.x
            gy = self.path.poses[-1].pose.position.y
            dist_goal = math.hypot(gx - self.ttbot_pose.pose.position.x,
                                   gy - self.ttbot_pose.pose.position.y)
            if dist_goal < self.get_parameter('goal_tol_m').value:
                self.get_logger().info('Goal reached.')
                self.at_goal = True
                self.stop_ttbot()
                continue
            idx = self.get_path_idx(self.path, self.ttbot_pose)
            current_goal = self.path.poses[idx]
            speed, heading = self.path_follower(self.ttbot_pose, current_goal)
            self.move_ttbot(speed, heading)


def main(args=None):
    rclpy.init(args=args)
    nav = Navigation(node_name='Navigation')

    try:
        nav.run()
    except KeyboardInterrupt:
        pass
    finally:
        nav.stop_ttbot()
        nav.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
