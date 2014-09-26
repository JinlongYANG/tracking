#define BOOST_SIGNALS_NO_DEPRECATION_WARNING

#include "ros/ros.h"
#include <tracking_node.hpp>

int main(int argc, char **argv)
{
  // Set up ROS.
  ros::init(argc, argv, "tracking",ros::init_options::NoRosout);
  ros::NodeHandle nh;

    tracking_Node tracking_Node(nh);

  // Main loop.
  while (nh.ok())
  {
    ros::spinOnce();
  }
  return 0;
}
