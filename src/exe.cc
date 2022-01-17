#include <iostream>

#include "meshing.h"

int main(int argc, char*argv[])
{
  if(argc < 3) {
    std::cerr << "Usage: " << argv[0] << " input.ply output.ply" << std::endl;
    return EXIT_FAILURE;
  }
  const char* input_file = argv[1];
  const char* output_file = argv[2];
  const double max_edge_length = 1.0;

  if(argc == 3) {
    meshing_from_paths(input_file, output_file, max_edge_length);
  } else {
    if(argc < 5) {
      return EXIT_FAILURE;
    }
    const char* endpoints_file = argv[3];
    const char* observations_file = argv[4];
    const int max_visibility = 0;
    bool post_filtering = true;
    if(argc >= 6) {
      if(std::string(argv[6]) == "pre")
        post_filtering = false;
    }
    meshing_with_visibility_from_paths(input_file,
                                       output_file,
                                       endpoints_file,
                                       observations_file,
                                       max_edge_length,
                                       max_visibility,
                                       post_filtering);
  }
  return 0;
}
