#include "vec3.h"
#include "vertex.h"
#include <cmath>
#include <iostream>
#include <ostream>

Vec3::Vec3(double* coords)
{
  x = coords[0];
  y = coords[1];
  z = coords[2];
}

Vec3::Vec3(double xIn, double yIn, double zIn)
{
  x = xIn;
  y = yIn;
  z = zIn;
}

Vec3::Vec3(Vertex& v)
{
  x = v.coords[0];
  y = v.coords[1];
  z = v.coords[2];
}

Vec3 Vec3::operator-(Vec3 v2)
{
  double xOut = x - v2.x;
  double yOut = y - v2.y;
  double zOut = z - v2.z;
  return Vec3(xOut,yOut,zOut);
}

// dot product
Vec3 Vec3::operator*(Vec3 v2)
{
  double xout = x * v2.x;
  double yout = y * v2.y;
  double zout = z * v2.z;
  return Vec3(xout,yout,zout);
}

Vec3 Vec3::operator*(double scale)
{
  double xout = x * scale;
  double yout = y * scale;
  double zout = z * scale; 
  return Vec3(xout,yout,zout);
}

Vec3 Vec3::operator/(double scale)
{
  double xout = x / scale;
  double yout = y / scale;
  double zout = z / scale; 
  return Vec3(xout,yout,zout);
}

// cross product
Vec3 Vec3::operator^(Vec3 v2)
{
  double xout = (y*v2.z) - (z*v2.y);
  double yout = (z*v2.x) - (x*v2.z);
  double zout = (x*v2.y) - (y*v2.x);
  return Vec3(xout,yout,zout);
}

double Vec3::dot(Vec3 v2)
{
  Vec3 prod = (*this) * v2;
  return prod.sum();
}

double Vec3::sum()
{
  return x + y + z;
}

double Vec3::mag()
{
  double mag2 = x*x + y*y + z*z;
  if(mag2 > 0)
	return sqrt(mag2);
  else
	return 0.;

}

void Vec3::normalize()
{
  double magnitude = mag();
  if(magnitude > 0)
  {
	x= (x*1.0)/magnitude;
	y= (y*1.0)/magnitude;
	z= (z*1.0)/magnitude;
  }
  else
	std::cout << "oooh no, zero vec! Can't normalize!" << std::endl;
}


std::ostream& operator<<(std::ostream& out, const Vec3& vec)
{
  out << "(" << vec.x << "," << vec.y << "," << vec.z << ")";
  return out;

}
