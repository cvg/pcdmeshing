#pragma once

#include <vector>
#include <iostream>
#include <fstream>

#include <CGAL/array.h>
#include <CGAL/property_map.h>
#include <CGAL/IO/read_ply_points.h>

#include <CGAL/Advancing_front_surface_reconstruction.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Cartesian_converter.h>

#include <boost/functional/hash.hpp>

#include "types.h"
#include "log_exceptions.h"
#include "visibility.h"


struct Priority {
  Priority(double max_edge_length_sq, const VisibilityCounter& vis_counter, int max_vis)
    : max_edge_length(max_edge_length_sq), vis_counter(vis_counter), max_vis(max_vis)
  {}

  Priority(double max_edge_length_sq)
    : max_edge_length(max_edge_length_sq)
  {}

  template <typename AdvancingFront, typename Cell_handle>
  double operator() (const AdvancingFront& adv, Cell_handle& c,
                     const int& index) const
  {
    // if the facet has a visibility count: discard if higher than a threshold
    auto vis = vis_counter.find(Triangulation_3::Facet(c, index));
    if(vis != vis_counter.end())
      if(vis->second > max_vis)
        return adv.infinity();

    // discard if any of the edges is longer than a threshold
    if(max_edge_length > 0.) {
      double d  = 0;
      d = squared_distance(c->vertex((index+1)%4)->point(),
                           c->vertex((index+2)%4)->point());
      if(d > max_edge_length)
        return adv.infinity();
      d = squared_distance(c->vertex((index+2)%4)->point(),
                           c->vertex((index+3)%4)->point());
      if(d > max_edge_length)
        return adv.infinity();
      d = squared_distance(c->vertex((index+1)%4)->point(),
                           c->vertex((index+3)%4)->point());
      if(d > max_edge_length)
        return adv.infinity();
    }

    return adv.smallest_radius_delaunay_sphere (c, index);
  }

  double max_edge_length;
  const VisibilityCounter vis_counter;
  int max_vis = 0;
};

typedef CGAL::Advancing_front_surface_reconstruction<Triangulation_3, Priority> Reconstruction;

// Mesh management: colored vertices and iterative face addition
struct Construct{
  Mesh& mesh;
  Construct(Mesh& mesh, const ColoredPoints& points)
    : mesh(mesh)
  {
    Mesh::Property_map<Mesh::Vertex_index, CGAL::Color>
          vcolors = mesh.add_property_map<Mesh::Vertex_index, CGAL::Color>("v:color").first;
    for(std::size_t i = 0; i < points.size (); ++ i)
    {
      const Point_3& p = get<0>(points[i]);
      const Color& c = get<2>(points[i]);
      boost::graph_traits<Mesh>::vertex_descriptor v = mesh.add_vertex(p);
      vcolors[v] = CGAL::Color(c[0], c[1], c[2]);
    }
  }
  void add_facet(const Triangle f)
  {
    typedef boost::graph_traits<Mesh>::vertex_descriptor vertex_descriptor;
    typedef boost::graph_traits<Mesh>::vertices_size_type size_type;
    mesh.add_face(vertex_descriptor(static_cast<size_type>(f[0])),
                  vertex_descriptor(static_cast<size_type>(f[1])),
                  vertex_descriptor(static_cast<size_type>(f[2])));
  }

  void add_facets(const Reconstruction R)
  {
    typedef typename Reconstruction::TDS_2::Face_iterator Face_iterator;
    const Reconstruction::TDS_2& tds = R.triangulation_data_structure_2();
    for(Face_iterator fit = tds.faces_begin(); fit != tds.faces_end(); ++fit){
      if(fit->is_on_surface()){
        add_facet(CGAL::make_array(fit->vertex(0)->vertex_3()->id(),
                                   fit->vertex(1)->vertex_3()->id(),
                                   fit->vertex(2)->vertex_3()->id()));
      }
    }
  }
};


ColoredPoints read_full_pointcloud(std::string input_file) {
  ColoredPoints points;
  std::cout << "Reading input pointcloud " << input_file << std::endl;
  std::ifstream in(input_file);
  THROW_CHECK_MSG(
      CGAL::IO::read_PLY_with_properties(in, std::back_inserter(points),
                                         CGAL::make_ply_point_reader(Point_map()),
                                         std::make_tuple(Color_map(),
                                                         CGAL::Construct_array(),
                                                         CGAL::IO::PLY_property<unsigned char>("red"),
                                                         CGAL::IO::PLY_property<unsigned char>("green"),
                                                         CGAL::IO::PLY_property<unsigned char>("blue")),
                                         CGAL::IO::make_ply_normal_reader(Normal_map())),
      std::string("Cannot read file") + input_file);
  return points;
}


Mesh pointcloud_meshing(const ColoredPoints& points, double max_edge_length) {
  Mesh mesh;
  Construct construct(mesh, points);

  //std::cout << "Building the Delaunay triangulation..." << std::endl;
  typedef CGAL::Cartesian_converter<Kernel,Kernel> CC;
  CC cc = CC();
  Triangulation_3 triangulation(
      boost::make_transform_iterator(mesh.points().begin(),
                                     CGAL::AFSR::Auto_count_cc<Point_3, CC>(cc)),
      boost::make_transform_iterator(mesh.points().end(),
                                     CGAL::AFSR::Auto_count_cc<Point_3, CC>(cc)));

  //std::cout << "Starting the meshing..." << std::endl;
  Priority priority(max_edge_length*max_edge_length);
  Reconstruction reconstruction(triangulation, priority);
  reconstruction.run();

  //std::cout << "Exporting facets to a mesh..." << std::endl;
  construct.add_facets(reconstruction);

  //std::cout << "Removing unused vertices..." << std::endl;
  CGAL::Polygon_mesh_processing::remove_isolated_vertices(mesh);

  return mesh;
}


Mesh pointcloud_meshing_with_visibility(const ColoredPoints& points,
                                        const std::vector<Segment>& rays,
                                        double max_edge_length,
                                        int max_visibility,
                                        bool post_filtering) {
  Mesh mesh;
  Construct construct(mesh, points);

  //std::cout << "Building the Delaunay triangulation..." << std::endl;
  typedef CGAL::Cartesian_converter<Kernel,Kernel> CC;
  CC cc = CC();
  Triangulation_3 triangulation(
      boost::make_transform_iterator(mesh.points().begin(),
                                     CGAL::AFSR::Auto_count_cc<Point_3, CC>(cc)),
      boost::make_transform_iterator(mesh.points().end(),
                                     CGAL::AFSR::Auto_count_cc<Point_3, CC>(cc)));

  VisibilityCounter visibility_counter;
  if(!post_filtering) {
    //std::cout << "Building the visibility count..." << std::endl;
    const DelaunayTriangulationRayCaster ray_caster(triangulation);
    #pragma omp parallel for
    for(size_t i = 0; i < rays.size(); i++) {
      //if(i % 1000000 == 0)
        //std::cout << "Ray casting " << i << " / " << rays.size() << std::endl;
      std::vector<Triangulation_3::Facet> intersections;
      ray_caster.CastRaySegment(rays[i], &intersections);
      for (const auto& intersection : intersections) {
        #pragma omp critical
        visibility_counter[intersection]++;
      }
    }
  }

  //std::cout << "Starting the meshing..." << std::endl;
  Priority priority(max_edge_length*max_edge_length,
                    visibility_counter,
                    max_visibility);
  Reconstruction reconstruction(triangulation, priority);
  reconstruction.run();

  //std::cout << "Exporting facets to a mesh..." << std::endl;
  construct.add_facets(reconstruction);

  if(post_filtering) {
    //std::cout << "Looking for mesh-visibility intersections..." << std::endl;
    AABBTree tree(faces(mesh).first, faces(mesh).second, mesh);
    std::unordered_set<size_t> faces_to_remove;
    #pragma omp parallel for
    for(size_t i = 0; i < rays.size(); i++) {
      std::list<AABBTree::Primitive_id> primitives;
      tree.all_intersected_primitives(rays[i], std::back_inserter(primitives));
      #pragma omp critical
      faces_to_remove.insert(primitives.begin(), primitives.end());
    }

    //std::cout << "Removing " << faces_to_remove.size() << " faces based on visibility." << std::endl;
    for(auto idx: faces_to_remove) {
      mesh.remove_face(Mesh::Face_index(idx));
    }
    mesh.collect_garbage();
  }

  //std::cout << "Removing unused vertices..." << std::endl;
  CGAL::Polygon_mesh_processing::remove_isolated_vertices(mesh);

  return mesh;
}

void meshing_from_paths(const std::string input_file,
                        const std::string output_file,
                        double max_edge_length
) {
  ColoredPoints points = read_full_pointcloud(input_file);
  //std::cout << "Read " << points.size() << " point(s)" << std::endl;
  Mesh mesh = pointcloud_meshing(points, max_edge_length);

  std::cout << "Writing output mesh to file " << output_file << std::endl;
  std::ofstream f (output_file, std::ios_base::binary);
  CGAL::IO::set_binary_mode (f);
  CGAL::IO::write_PLY(f, mesh);
  f.close();
}

void meshing_with_visibility_from_paths(const std::string input_file,
                                        const std::string output_file,
                                        const std::string endpoints_file,
                                        const std::string observations_file,
                                        double max_edge_length,
                                        int max_visibility,
                                        bool post_filtering) {
  ColoredPoints points = read_full_pointcloud(input_file);
  //std::cout << "Read " << points.size() << " point(s)" << std::endl;

  std::vector<Segment> rays;
  read_visibility_from_paths(endpoints_file, observations_file, rays);

  Mesh mesh = pointcloud_meshing_with_visibility(
      points, rays, max_edge_length, max_visibility, post_filtering);

  std::cout << "Writing output mesh to file " << output_file << std::endl;
  std::ofstream f (output_file, std::ios_base::binary);
  CGAL::IO::set_binary_mode (f);
  CGAL::IO::write_PLY(f, mesh);
  f.close();
}
