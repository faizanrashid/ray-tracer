/************************************************************
     Starter code for Assignment 3

     This code was originally written by Jack Wang for

		    CSC418, SPRING 2005

		Implementations of functions in raytracer.h, 
		and the main function which specifies the 
		scene to be rendered.	

***********************************************************/


#include "raytracer.h"
#include "bmp_io.h"
#include <cmath>
#include <iostream>
#include <cstdlib>

extern const int NUM_RAYS = 10;
extern const int NUM_SHADOW_RAYS = 1;
extern const int NUM_REFLECT_RAYS = 9;
Raytracer::Raytracer() : _lightSource(NULL) {
	_root = new SceneDagNode();
}

Raytracer::~Raytracer() {
	delete _root;
}

SceneDagNode* Raytracer::addObject( SceneDagNode* parent, 
		SceneObject* obj, Material* mat ) {
	SceneDagNode* node = new SceneDagNode( obj, mat );
	node->parent = parent;
	node->next = NULL;
	node->child = NULL;
	
	// Add the object to the parent's child list, this means
	// whatever transformation applied to the parent will also
	// be applied to the child.
	if (parent->child == NULL) {
		parent->child = node;
	}
	else {
		parent = parent->child;
		while (parent->next != NULL) {
			parent = parent->next;
		}
		parent->next = node;
	}
	
	return node;;
}

LightListNode* Raytracer::addLightSource( LightSource* light ) {
	LightListNode* tmp = _lightSource;
	_lightSource = new LightListNode( light, tmp );
	return _lightSource;
}

void Raytracer::rotate( SceneDagNode* node, char axis, double angle ) {
	Matrix4x4 rotation;
	double toRadian = 2*M_PI/360.0;
	int i;
	
	for (i = 0; i < 2; i++) {
		switch(axis) {
			case 'x':
				rotation[0][0] = 1;
				rotation[1][1] = cos(angle*toRadian);
				rotation[1][2] = -sin(angle*toRadian);
				rotation[2][1] = sin(angle*toRadian);
				rotation[2][2] = cos(angle*toRadian);
				rotation[3][3] = 1;
			break;
			case 'y':
				rotation[0][0] = cos(angle*toRadian);
				rotation[0][2] = sin(angle*toRadian);
				rotation[1][1] = 1;
				rotation[2][0] = -sin(angle*toRadian);
				rotation[2][2] = cos(angle*toRadian);
				rotation[3][3] = 1;
			break;
			case 'z':
				rotation[0][0] = cos(angle*toRadian);
				rotation[0][1] = -sin(angle*toRadian);
				rotation[1][0] = sin(angle*toRadian);
				rotation[1][1] = cos(angle*toRadian);
				rotation[2][2] = 1;
				rotation[3][3] = 1;
			break;
		}
		if (i == 0) {
		    node->trans = node->trans*rotation; 	
			angle = -angle;
		} 
		else {
			node->invtrans = rotation*node->invtrans; 
		}	
	}
}

void Raytracer::translate( SceneDagNode* node, Vector3D trans ) {
	Matrix4x4 translation;
	
	translation[0][3] = trans[0];
	translation[1][3] = trans[1];
	translation[2][3] = trans[2];
	node->trans = node->trans*translation; 	
	translation[0][3] = -trans[0];
	translation[1][3] = -trans[1];
	translation[2][3] = -trans[2];
	node->invtrans = translation*node->invtrans; 
}

void Raytracer::scale( SceneDagNode* node, Point3D origin, double factor[3] ) {
	Matrix4x4 scale;
	
	scale[0][0] = factor[0];
	scale[0][3] = origin[0] - factor[0] * origin[0];
	scale[1][1] = factor[1];
	scale[1][3] = origin[1] - factor[1] * origin[1];
	scale[2][2] = factor[2];
	scale[2][3] = origin[2] - factor[2] * origin[2];
	node->trans = node->trans*scale; 	
	scale[0][0] = 1/factor[0];
	scale[0][3] = origin[0] - 1/factor[0] * origin[0];
	scale[1][1] = 1/factor[1];
	scale[1][3] = origin[1] - 1/factor[1] * origin[1];
	scale[2][2] = 1/factor[2];
	scale[2][3] = origin[2] - 1/factor[2] * origin[2];
	node->invtrans = scale*node->invtrans; 
}

Matrix4x4 Raytracer::initInvViewMatrix( Point3D eye, Vector3D view, 
		Vector3D up ) {
	Matrix4x4 mat; 
	Vector3D w;
	view.normalize();
	up = up - up.dot(view)*view;
	up.normalize();
	w = view.cross(up);

	mat[0][0] = w[0];
	mat[1][0] = w[1];
	mat[2][0] = w[2];
	mat[0][1] = up[0];
	mat[1][1] = up[1];
	mat[2][1] = up[2];
	mat[0][2] = -view[0];
	mat[1][2] = -view[1];
	mat[2][2] = -view[2];
	mat[0][3] = eye[0];
	mat[1][3] = eye[1];
	mat[2][3] = eye[2];

	return mat; 
}

void Raytracer::traverseScene( SceneDagNode* node, Ray3D& ray ) {
	SceneDagNode *childPtr;

	// Applies transformation of the current node to the global
	// transformation matrices.
	_modelToWorld = _modelToWorld*node->trans;
	_worldToModel = node->invtrans*_worldToModel; 
	if (node->obj) {
		// Perform intersection.
		if (node->obj->intersect(ray, _worldToModel, _modelToWorld)) {
			ray.intersection.mat = node->mat;
		}
	}
	// Traverse the children.
	childPtr = node->child;
	while (childPtr != NULL) {
		traverseScene(childPtr, ray);
		childPtr = childPtr->next;
	}

	// Removes transformation of the current node from the global
	// transformation matrices.
	_worldToModel = node->trans*_worldToModel;
	_modelToWorld = _modelToWorld*node->invtrans;
}

void Raytracer::computeShading( Ray3D& ray ) {
	LightListNode* curLight = _lightSource;

	Colour sum_shade(0.0, 0.0, 0.0);
	for (;;) {
		if (curLight == NULL) break;
		// Each lightSource provides its own shading function.
	      	// Implement shadows here if needed.

		/*Point3D light_position = curLight->light->get_position();
		Vector3D dir = (light_position -  ray.intersection.point);
		dir.normalize();
		Point3D origin = ray.intersection.point;
		origin = origin + 0.01*dir;
		Ray3D shadowRay = Ray3D(origin, dir); 
		traverseScene(_root, shadowRay); 
		if (shadowRay.intersection.none) {
			curLight->light->shade(ray);
		}
		//Soft shadows
		sum_shade = sum_shade + 0.5*ray.col;
		for (int k=0; k < NUM_SHADOW_RAYS; k++) {
			double varx = ((((double)rand() / (double)RAND_MAX)*2) - 1) * 0.10;
			double vary = ((((double)rand() / (double)RAND_MAX)*2) - 1) * 0.10;
			double varz = ((((double)rand() / (double)RAND_MAX)*2) - 1) * 0.10;
			Point3D tempShadowOrigin = origin;
			tempShadowOrigin[0] = tempShadowOrigin[0] + varx;
			tempShadowOrigin[1] = tempShadowOrigin[1] + vary;
			tempShadowOrigin[2] = tempShadowOrigin[2] + varz;			
			tempShadowOrigin = tempShadowOrigin + 0.2*dir; 	
			shadowRay = Ray3D(tempShadowOrigin, dir);
			traverseScene(_root, shadowRay);
			if (shadowRay.intersection.none) {
				curLight->light->shade(ray);				
				sum_shade = sum_shade + 0.1*ray.col;
			}
			
		}*/	
		
		//Uncomment the line below if you want to turn shadows off
		curLight->light->shade(ray);
		curLight = curLight->next;
		
	}
	//sum_shade = (1.0/( (0.5) + (0.1)*(NUM_SHADOW_RAYS-1)))*sum_shade;
	//sum_shade.clamp();
	//ray.col = sum_shade;	
}

void Raytracer::initPixelBuffer() {
	int numbytes = _scrWidth * _scrHeight * sizeof(unsigned char);
	_rbuffer = new unsigned char[numbytes];
	_gbuffer = new unsigned char[numbytes];
	_bbuffer = new unsigned char[numbytes];
	for (int i = 0; i < _scrHeight; i++) {
		for (int j = 0; j < _scrWidth; j++) {
			_rbuffer[i*_scrWidth+j] = 0;
			_gbuffer[i*_scrWidth+j] = 0;
			_bbuffer[i*_scrWidth+j] = 0;
		}
	}
}

void Raytracer::flushPixelBuffer( char *file_name ) {
	bmp_write( file_name, _scrWidth, _scrHeight, _rbuffer, _gbuffer, _bbuffer );
	delete _rbuffer;
	delete _gbuffer;
	delete _bbuffer;
}

Colour Raytracer::shadeRay( Ray3D& ray, int recursive_depth) {
	Colour col(0.0, 0.0, 0.0); 
	traverseScene(_root, ray); 
	Colour reflColour(0.0, 0.0, 0.0);
	Colour refractColour(0.0, 0.0, 0.0);
	// Don't bother shading if the ray didn't hit 
	// anything.
	if (!ray.intersection.none) {
		computeShading(ray);
		col = ray.col;
		if (recursive_depth < 2) {
			//Reflection code
       			Vector3D norm = ray.intersection.normal;
			norm.normalize();
			Vector3D direction = ray.dir;
			direction.normalize();	
			Point3D origin_reflRay = ray.intersection.point;
			Vector3D dir_reflRay = (direction - ((2*(direction.dot(norm)))*norm));
			dir_reflRay.normalize();
			Ray3D reflRay = Ray3D(origin_reflRay + 0.01*dir_reflRay, dir_reflRay);
			double dampFactor = ray.intersection.mat->damp_factor;
		//	reflColour = dampFactor*shadeRay(reflRay , recursive_depth+1);
		//	for (int k = 0; k < NUM_REFLECT_RAYS; k++) {
		//		double varx = ((((double)rand() / (double)RAND_MAX)*2) - 1) * 0.10;
		//		double vary = ((((double)rand() / (double)RAND_MAX)*2) - 1) * 0.10;
		//		double varz = ((((double)rand() / (double)RAND_MAX)*2) - 1) * 0.10;
		//		Point3D variedOrigin = origin_reflRay;				
		//		variedOrigin[0] += varx;
		//		variedOrigin[1] += vary;
		//		reflRay = Ray3D(variedOrigin + 0.01*dir_reflRay, dir_reflRay);
		//		reflColour = reflColour + dampFactor*shadeRay(reflRay , recursive_depth+1); 
		//	}
		//	reflColour = (1.0/(NUM_REFLECT_RAYS))*reflColour;
			
			


			//refraction code
			double n2 = ray.intersection.mat->refraction_factor;
			Point3D origin_refractRay = ray.intersection.point;
			
			double n1n2 = 1/n2; //n1 is air
			if (recursive_depth == 1) {
				n1n2 = 1/n1n2;
			}
			double costhetai = norm.dot(direction);
			double sinSqrdThetaT = (n1n2)*(n1n2)*(1 - costhetai*costhetai);
			if (sinSqrdThetaT > 1){  // Don't need to refract. Critical angle condition
				col.clamp();
				return col;
			}
			Vector3D dir_refractRay = n1n2*direction + ((n1n2*costhetai - sqrt(1 - sinSqrdThetaT))*norm)  ;
			dir_refractRay.normalize();
			Ray3D refractRay = Ray3D(origin_refractRay + 0.01*dir_refractRay, dir_refractRay);
			refractColour = shadeRay(refractRay, recursive_depth+1);
		
		}
	
		col = col + reflColour + refractColour;
		col.clamp();
	}			
	// You'll want to call shadeRay recursively (with a different ray, 
	// of course) here to implement reflection/refraction effects.  

	return col; 
}	

void Raytracer::render( int width, int height, Point3D eye, Vector3D view, 
		Vector3D up, double fov, int focal_dist,  char* fileName ) {
	Matrix4x4 viewToWorld;
	_scrWidth = width;
	_scrHeight = height;
	double factor = (double(height)/2)/tan(fov*M_PI/360.0);

	initPixelBuffer();
	viewToWorld = initInvViewMatrix(eye, view, up);
	// Construct a ray for each pixel.
	for (int i = 0; i < _scrHeight; i++) {
		for (int j = 0; j < _scrWidth; j++) {
			Colour col(0.0, 0.0, 0.0);
			//sets up pixel  ray origin and direction in view space, 
			// image plane is at z = -1.
			Point3D origin(0, 0, 0);
			Point3D imagePlane;
			imagePlane[0] = (-double(width)/2 + 0.5 + j)/factor;
			imagePlane[1] = (-double(height)/2 + 0.5 + i)/factor;
			imagePlane[2] = -1;
 
			// TODO: Convert ray to world space and call 
			// shadeRay(ray) to generate pixel colour. 	
			    // r¯(λ) = (¯pWi,j) + λ~(di,j)
			   // viewToWorld*imageplane  will give us the point in world coords
			  
			Vector3D rayDirection = (viewToWorld*imagePlane) - eye;
			rayDirection.normalize();
			Point3D rayOrigin = viewToWorld * origin;	
			Point3D focal_point = origin + focal_dist*(imagePlane - origin);

			
			for (int z = 0; z < 6; z++) {
				double varx = ((((double)rand() / (double)RAND_MAX)*3) - 1)*0.05  ;
				double vary = ((((double)rand() / (double)RAND_MAX)*3) - 1)*0.05 ;
				Point3D origin_temp = origin;
				origin_temp[0] += varx;
				origin_temp[1] += vary;
				rayOrigin = viewToWorld*origin_temp;
				rayDirection = viewToWorld*focal_point - rayOrigin;
				//montecarlo for distributed ray tracing
						
				//Ray3D ray_objects[NUM_RAYS];
				Ray3D ray = Ray3D(rayOrigin, rayDirection);
				// ray_objects[0] = ray;  incase you need to store the rays
				Colour temp_col = shadeRay(ray);			
			/*	for (int k=1; k < NUM_RAYS; k++) {
					double varx = ((((double)rand() / (double)RAND_MAX)*2) - 1) * 0.05;
					double vary = ((((double)rand() / (double)RAND_MAX)*2) - 1) * 0.05;
					Point3D variedOrigin = rayOrigin;				
					variedOrigin[0] += varx;
					variedOrigin[1] += vary;
					Ray3D ray = Ray3D(variedOrigin, rayDirection);
					// ray_objects[k] = ray; incase you need to store the rays
					temp_col = tempcol + shadeRay(ray);
				}
				temp_col = (1.0/NUM_RAYS) * temp_col ; 
			*/
				col = col + temp_col;
			}	
			col = (1.0/6.0)*col;
			_rbuffer[i*width+j] = int(col[0]*255);
			_gbuffer[i*width+j] = int(col[1]*255);
			_bbuffer[i*width+j] = int(col[2]*255);
		}
	}

	flushPixelBuffer(fileName);
}

int main(int argc, char* argv[])
{	
	// Build your scene and setup your camera here, by calling 
	// functions from Raytracer.  The code here sets up an example
	// scene and renders it from two different view points, DO NOT
	// change this if you're just implementing part one of the 
	// assignment.  
	Raytracer raytracer;
	int width = 320; 
	int height = 240; 

	if (argc == 3) {
		width = atoi(argv[1]);
		height = atoi(argv[2]);
	}

	// Camera parameters.
	Point3D eye(1, 0, 1);
	Vector3D view(0, 0, -1);
	Vector3D up(0, 1, 0);
	double fov = 60;

	// Defines a material for shading.
	Material gold( Colour(0.3, 0.3, 0.3), Colour(0.86, 0.088888, 0.23529), 
			Colour(0.628281, 0.555802, 0.366065),
			25.0, 0.0, 0.0);

	Material glass( Colour(0.3, 0.3, 0.3), Colour(0.86, 0.0888888, 0.23529),
				Colour(0.628281, 0.555802, 0.366065),
				25.0, 0.0, 0.0);

	Material purpleWall( Colour(0.1, 0.1, 0.1), Colour(0.6, 0.6, 0.6), 
			Colour(0.4, 0.4, 0.5), 
			5.0, 1.0, 0.00);


	// Defines a point light source.
	raytracer.addLightSource( new PointLight(Point3D(0, 0, 1 ), 
				Colour(0.9, 0.9, 0.9) ) );

	//raytracer.addLightSource( new PointLight(Point3D(-1, 0, 5), 
	//			Colour(0.9, 0.9, 0.9) ) );



	// Add a unit square into the scene with material mat.
	SceneDagNode* sphere = raytracer.addObject( new UnitSphere(), &gold );
//	SceneDagNode* plane = raytracer.addObject( new UnitSquare(), &purpleWall);
	SceneDagNode* sphere2 = raytracer.addObject(new UnitSphere(), &glass);
	
	// Apply transformations to unit sphere and square
	double factor1[3] = { 0.5, 0.5, 0.5 };
	double factor2[3] = { 6.0, 6.0, 6.0 };
	double factor3[3] = {0.15, 0.15, 0.15};


	//sphere2 transformations
	raytracer.translate(sphere2, Vector3D(1.0, 0, 0));
	raytracer.scale(sphere2, Point3D(0,0,0), factor3);

	//sphere transformations
	raytracer.translate(sphere, Vector3D(3, 0, -6.5));	
	raytracer.rotate(sphere, 'x', -45); 
	raytracer.rotate(sphere, 'z', 45); 
	raytracer.scale(sphere, Point3D(0, 0, 0), factor1);

	//plane transformations
	//raytracer.translate(plane, Vector3D(0, 0, -7));	
	//raytracer.rotate(plane, 'z', 45); 
	//raytracer.scale(plane, Point3D(0, 0, 0), factor2);


	
	// Render the scene, feel free to make the image smaller for
	// testing purposes.	
	raytracer.render(width, height, eye, view, up, fov, 1,  "view1.bmp");
	
	// Render it from a different point of view.
	Point3D eye2(-6, 0, -6);
	Vector3D view2(1, 0, -0.1);
	raytracer.render(width, height, eye2, view2, up, fov, 7, "view2.bmp");
	
	return 0;
}
