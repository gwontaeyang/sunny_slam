#include <exception>
#include <iostream>

#include "slam_3d_map_cleanup/map_cleanup.hpp"

int main(int argc, char ** argv)
{
  try {
    return gole::slam_3d_map_cleanup::runTemporalMapCleanupCli(argc, argv, std::cout, std::cerr);
  } catch (const std::exception & error) {
    std::cerr << "temporal_map_cleanup: " << error.what() << '\n';
    return 1;
  }
}
