#pragma once
#include <ostream>
class Vertex;
// simple class which can be built from any coord vector of length 3.
// provides methods for calculating basic vec-ops (cross/dot product)
// without having to think hard about memory management.
class Vec3
{
  friend std::ostream& operator<<(std::ostream& out, const Vec3& vec);
  public:
    // can build a vec object from a pointer to the start of a double
	// array of length 3
	Vec3(double* coords);
	// or directly from three discrete values
	Vec3(double x, double y, double z);
	// or coersion from a vector object (such as in assignment)
	Vec3(Vertex& v);
  // write out to a double array at position given
  void writeOutTo(double* array);
	// internally, values stored in three doubles
	double x,y,z;
	// * == ELEMENT product
	Vec3 operator*(Vec3 v2);
	// ^ == CROSS product
	double dot(Vec3 v2);
	Vec3 operator-(Vec3 v2);
	Vec3 operator+(Vec3 v2);
	Vec3 operator^(Vec3 v2);
	Vec3 operator*(double scale);
	Vec3 operator/(double scale);
	double mag();
	double sum();
	void normalize();
};
