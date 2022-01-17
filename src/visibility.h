#pragma once

#include "types.h"
#include "log_exceptions.h"

// Ray caster through the cells of a Delaunay triangulation. The tracing locates
// the cell of the ray origin and then iteratively intersects the ray with all
// facets of the current cell and advances to the neighboring cell of the
// intersected facet. Note that the ray can also pass through outside of the
// hull of the triangulation, i.e. lie within the infinite cells/facets.
// The ray caster collects the intersected facets along the ray.
typedef Triangulation_3 Delaunay;
typedef Delaunay::Segment_simplex_iterator Simplex_iterator;
struct DelaunayTriangulationRayCaster {
  DelaunayTriangulationRayCaster(const Delaunay& triangulation)
      : triangulation_(triangulation) {}

  void CastRaySegment(const Kernel::Segment_3& ray_segment,
                      std::vector<Delaunay::Facet>* intersections) const {
    intersections->clear();
    Delaunay::Segment_traverser_simplices it = triangulation_.segment_traverser_simplices(
        ray_segment.start(), ray_segment.end());
    for (const Delaunay::Simplex& s :it) {
      if(s.dimension() == 2) {  // it's a facet
        intersections->push_back(Delaunay::Facet(s));
      }
    }
  }

 private:
  const Delaunay& triangulation_;
};

void read_visibility_from_paths(const std::string endpoints_file,
                                const std::string observations_file,
                                std::vector<Segment>& rays
) {
  Point_set endpoints;
  //std::cout << "Reading " << endpoints_file << "..." << std::endl;
  std::ifstream in(endpoints_file);
  THROW_CHECK_MSG(
    CGAL::IO::read_PLY(in, endpoints),
    std::string("Can't read input file ") + endpoints_file);
  size_t num_points = endpoints.size();
  //std::cout << "Found " << num_points << " observations" << std::endl;

  Point_set observations;
  //std::cout << "Reading " << observations_file << "..." << std::endl;
  std::ifstream in_obs(observations_file);
  THROW_CHECK_MSG(
    CGAL::IO::read_PLY(in_obs, observations),
    std::string("Can't read input file ") + observations_file);
  size_t num_obs = observations.size();
  THROW_CHECK_MSG(
    num_points == num_obs,
    std::string("Different number of points: ") + std::to_string(num_points)
    + " vs " + std::to_string(num_obs));

  const double view_offset = 0.1;
  rays.clear();
  rays.reserve(num_points);
  for (size_t i = 0; i < num_points; i++) {
    Segment ray(observations.point(i), endpoints.point(i));
    Vector_3 vec = ray.to_vector();
    Segment ray_trimmed = Segment(
        ray.source(),
       ray.target() - (vec * (view_offset / std::sqrt(vec.squared_length()))));
    rays.push_back(ray_trimmed);
  }
}
