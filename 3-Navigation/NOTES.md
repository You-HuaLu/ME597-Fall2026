# Lab 3 notes

- `navigation_astar_f24.ipynb`: course notebook with `AStar` completed (heapq open list). Map name set to `classroom_map`
  (files in this folder), so it runs without Google Drive. Remove the Drive-mount cell if present in your copy.
- `auto_navigator.py`: A* (heapq, 8-connected, inflated map) + look-ahead path follower. Plans once per new goal.
  Map is a ROS parameter: `--ros-args -p map_yaml:=<path>/classroom_map.yaml` (also `inflation_m`, `lookahead_m`, `max_speed`).
- Checked off-ROS only: A* matches Dijkstra path cost on the classroom map, and the follower reaches the goal in a
  kinematic simulation. Not yet run in Gazebo / on the TurtleBot.
