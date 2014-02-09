/**********************************************************
     Starter code for Assignment 3

     This code was originally written by Jack Wang for
		    CSC418, SPRING 2005

		implements scene_object.h

***********************************************************/

#include <cmath>
#include <iostream>
#include "scene_object.h"
#include <iostream>







bool UnitSquare::intersect( Ray3D& ray, const Matrix4x4& worldToModel,
		const Matrix4x4& modelToWorld ) {
	// TODO: implement intersection code for UnitSquare, which is
	// defined on the xy-plane, with vertices (0.5, 0.5, 0), 
	// (-0.5, 0.5, 0), (-0.5, -0.5, 0), (0.5, -0.5, 0), and normal
	// (0, 0, 1).
	//
	// Your goal here is to fill ray.intersection with correct values
	// should an intersection occur.  This includes intersection.point, 
	// intersection.normal, intersection.none, intersection.t_value.   
	//
	// HINT: Remember to first transform the ray into object space  
	// to simplify the intersection test.
	

	// Transform ray into object space:

	Point3D origin = worldToModel * ray.origin;
	Vector3D dir = worldToModel * ray.dir;
	Ray3D ray_obj = Ray3D(origin, dir);

	double t = -ray_obj.origin[2]/ray_obj.dir[2];

	if (t <= 0){
		// the intersection must occur in the given directions	
		return false;
	}
	Vector3D ray_normal = Vector3D(0, 0, 1);
	double x = ray_obj.origin[0] + t * ray_obj.dir[0];
	double y = ray_obj.origin[1] + t * ray_obj.dir[1];
	Point3D intersectionPoint = Point3D(x,y,0);
	
	//Intersecion
	if ((x <= 0.5) && (x >= -0.5) && (y <= 0.5) && (y >= -0.5)) 
	{
		if (ray.intersection.none || (t < ray.intersection.t_value))
		{
			ray.intersection.t_value = t;
			ray.intersection.point = modelToWorld * intersectionPoint;
			ray.intersection.normal = transNorm(worldToModel, ray_normal);
			ray.intersection.none = false;
			return true;
		}
	
	}

	return false;
}

bool UnitSphere::intersect( Ray3D& ray, const Matrix4x4& worldToModel,
		const Matrix4x4& modelToWorld ) {
	// TODO: implement intersection code for UnitSphere, which is centred 
	// on the origin.  
	//
	// Your goal here is to fill ray.intersection with correct values
	// should an intersection occur.  This includes intersection.point, 
	// intersection.normal, intersection.none, intersection.t_value.   
	//
	// HINT: Remember to first transform the ray into object space  
	// to simplify the intersection test.
	Point3D origin = Point3D(0,0,0);
	Point3D ray_origin = worldToModel * ray.origin;
	Vector3D ray_dir = worldToModel * ray.dir;
	Ray3D ray_obj = Ray3D(ray_origin, ray_dir);
	
	double A = ray_dir.dot(ray_dir);
	double B = (ray_origin - origin).dot(ray_dir);
	double C = (ray_origin - origin).dot(ray_origin - origin) - 1;

	double D = (B * B) - (A * C);
	
	if (D < 0) { // No intersections
		return false;
	}

	double lambda1 = -(B/A) + ((sqrt(D)/A));
	double lambda2 = -(B/A) - ((sqrt(D)/A));

	if ((lambda1 < 0) && (lambda2 < 0)) {
		return false;
	}

	if  ( ((lambda1 < 0) && (lambda2 > 0)) || ((lambda1 > lambda2) && (lambda2 > 0)) || (lambda1 == lambda2)) {
		double x = ray_obj.origin[0] + lambda2 * ray_obj.dir[0];
		double y = ray_obj.origin[1] + lambda2 * ray_obj.dir[1];
		double z = ray_obj.origin[2] + lambda2 * ray_obj.dir[2];

		Point3D intersection_point = Point3D(x,y,z);
		Vector3D ray_normal = Vector3D(2*x, 2*y, 2*z); // gradient
	
		if (ray.intersection.none || (lambda2 < ray.intersection.t_value)) {
			ray.intersection.t_value = lambda2;
			ray.intersection.point = modelToWorld * intersection_point;
			ray.intersection.normal = transNorm(worldToModel, ray_normal);
			ray.intersection.none = false;
			return true;	
		}
	}else if  ( ((lambda2 < 0) && (lambda1 > 0)) || ((lambda2 > lambda1) && (lambda1 > 0)) ) {
		double x = ray_obj.origin[0] + lambda1 * ray_obj.dir[0];
		double y = ray_obj.origin[1] + lambda1 * ray_obj.dir[1];
		double z = ray_obj.origin[2] + lambda1 * ray_obj.dir[2];

		Point3D intersection_point = Point3D(x,y,z);
		Vector3D ray_normal = Vector3D(2*x, 2*y, 2*z); // gradient
	
		if (ray.intersection.none || (lambda1 < ray.intersection.t_value)) {
			ray.intersection.t_value = lambda1;
			ray.intersection.point = modelToWorld * intersection_point;
			ray.intersection.normal = transNorm(worldToModel, ray_normal);
			ray.intersection.none = false;
			
			return true;	
		}
		
	}
	
	return false;
}

