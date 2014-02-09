/***********************************************************
     Starter code for Assignment 3

     This code was originally written by Jack Wang for
		    CSC418, SPRING 2005

		implements light_source.h

***********************************************************/

#include <cmath>
#include "light_source.h"
#include "bmp_io.h"

void PointLight::shade( Ray3D& ray ) {
	// TODO: implement this function to fill in values for ray.col 
	// using phong shading.  Make sure your vectors are normalized, and
	// clamp colour values to 1.0.
	//
	// It is assumed at this point that the intersection information in ray 
	// is available.  So be sure that traverseScene() is called on the ray 
	// before this function. 
	
	Colour KA = (*ray.intersection.mat).ambient;
	Colour KD = (*ray.intersection.mat).diffuse;
	Colour KS = (*ray.intersection.mat).specular;

	Colour IA = _col_ambient;
	Colour ID = _col_diffuse;
	Colour IS = _col_specular;

	Vector3D N = (ray.intersection.normal);
	N.normalize();

	Vector3D L = (_pos - ray.intersection.point);
	L.normalize();
	
	Vector3D V = (-1 * ray.dir);
	V.normalize();

	Vector3D R = ((2 * (L.dot(N)) * N) - L);
	R.normalize();

	double alpha = (*ray.intersection.mat).specular_exp;
	double maxNdotL = (0.0<N.dot(L))?N.dot(L):0.0;
	double maxVdotR = (0.0<pow(V.dot(R), alpha))?pow(V.dot(R), alpha):0.0;
	Colour shade = KA * IA + KD * (maxNdotL * ID) + KS * (maxVdotR * IS);
	shade.clamp();
	ray.col = shade;
	//ray.col = (*ray.intersection.mat).diffuse +  (*ray.intersection.mat).ambient;

	//ray.col in the shading function simply to ray.intersection.mat->diffuse
	//ray.col = (*ray.intersection.mat).diffuse;
	

	
}

