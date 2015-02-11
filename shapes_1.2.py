#!/usr/bin/env python

import sys
import json 
import math
import numpy as np
import collections

def ccw(A, B, C):
	"""Tests whether the line formed by A, B, and C is counter clockwise"""
	return (B['x'] - A['x'])*(C['y'] - A['y']) > (B['y'] - A['y'])*(C['x'] - A['x'])

def intersect(A, B, C, D):
	"""Tests whether line segments AB and CD intersect"""
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

class Shape():

	def __init__(self, shapejson):
		if 'id' in shapejson:
			self.name = shapejson['id']
		else:
			self.name = None

		if 'point' in shapejson:
			self.points = shapejson['point']
		elif 'x' not in shapejson[0]:
			self.points = [{'x':point[0], 'y':point[1]} for point in shapejson]
		self.n = len(self.points)


	def is_convex(self):

		"""
			is_convex works by checking that all consecutive line segments in the
			shape rotate either clockwise or counter clockwise. If there is even
			one case where the rotation doesn't match the rest, the shape is not convex.
			NOTE: This assumes the shape has no holes
			
			Read more here:
			http://www.toptal.com/python/computational-geometry-in-python-from-theory-to-implementation
		"""

		#Check if there aren't enought points to represent a polygon
		if self.n < 3:
			print "'{}' has too few points to represent a polygon!".format(self.name)
			return False
		
		target = None
		for i in range(self.n):
		    #Check every triplet of points
		    A = self.points[i % self.n]
		    B = self.points[(i + 1) % self.n]
		    C = self.points[(i + 2) % self.n]
		    #ccw() checks to see if a bend between 3 points is clockwise or counter cw.
		    if target == None:
		        target = ccw(A, B, C)
		    elif ccw(A, B, C) != target:
		        return False
		    
		return True


class Memoize(dict):
	#This memoize decorator class is based of the class seen at url below:
	#https://wiki.python.org/moin/PythonDecoratorLibrary

	def __init__(self, func):
		self.func = func
    
	def __call__(self, *args):
		return self[args]
    
	def __missing__(self, key):
		result = self[key] = self.func(*key)
		#return "memoizing {}".format(result)
		return result

				
@Memoize
def sep_ax(shapeA, shapeB):
	#I chose to model the sep_ax function like a service that other functions 
	#call and so then have sep_ax memoized so that the projections don't need 
	#to be recomputed

	"""
		This algorithm is written by implementing the Separating Axis Theorum (SAT), 
		see url for in depth explanation: http://tinyurl.com/p45k6ga
		the algorithm is generalized such that other functions call it
		and use it's results as necessary. 

		This algorithm projects the points of each shape onto an axis found by
		first finding a vector representing the line segment of an edge in a shape,
		then finding the perpedicular of that segment ([x,y] -> [-y,x]).
		We do this for each edge on a shape, or both shapes depending on 
		whether we're testing for intersection or containment. The conditionals 
		for those two cases live in other functions.

		To think of this algorithm visually: if you had two objects and a 
		flashlight, and shined the flashlight on the objects to see thier 
		shadows on the wall, if you could find an angle to shine the light
		such that you saw light between the two shadows, you could conclude 
		that the shapes do not intersect. Also if you only saw one shadow
		in every angle, you could conclude that one object is inside the other. 
	"""

	#We use a deque because it has the convenient rotate method from 
	#pythons's collection module
	pointsA = [shapeA.name, collections.deque(shapeA.points)]
	pointsB = [shapeB.name, collections.deque(shapeB.points)]

	
	shapes = [pointsA, pointsB]

	out = []
	for shape in shapes:	
		rotshape = shape[1]
		for rot in xrange(len(rotshape)):
			rotshape.rotate(rot)
			#Find vector representing line segment between two points, we'll 
			#then use this to compute a perpedicular vector which we'll use as 
			lineVecA = [rotshape[1]['x'] - rotshape[0]['x'], rotshape[1]['y'] - rotshape[0]['y']]
			#An easy way to find the perpendicular of a vector [x,y] is to take [-y,x]
			perpVec = np.array([-lineVecA[1], lineVecA[0]])
			#Normalize the vector
			perpVec = perpVec/np.linalg.norm(perpVec)

			#Perform projection onto axis for all point in both shapes
			projsA = [(np.dot([point['x'], point['y']], perpVec)/(np.linalg.norm(perpVec)**2)) * perpVec for point in pointsA[1]]
			projsB = [(np.dot([point['x'], point['y']], perpVec)/(np.linalg.norm(perpVec)**2)) * perpVec for point in pointsB[1]]

			#Project onto one dimensional line so we can begin to compare points easily
			oneDProjA = [np.dot(perpVec, point) for point in projsA]
			oneDProjB = [np.dot(perpVec, point) for point in projsB]

			out.append((oneDProjA, oneDProjB, shape[0]))

	return out

@Memoize
def intersecting_shapes(shapeA, shapeB):
	#This is the intersect condition, it checks that there is no overlap in the 
	#projection ranges. If there is even one case of no overlap, the shapes don't intersect
	sepax = sep_ax(shapeA, shapeB)

	for oneDProjA, oneDProjB, name in sepax:
		if  ((max(oneDProjA) < min(oneDProjB)) and (min(oneDProjA) < max(oneDProjB)) 
			or (max(oneDProjB) < min(oneDProjA)) and (min(oneDProjB) < max(oneDProjA))):
			return False
	if is_shape_in_other(shapeA, shapeB) in (1, -1):
		return False
	return True


@Memoize
def is_shape_in_other(shapeA, shapeB):
	
	"""
		These conditional check if shapeA's projection fits inside the range of 
		shapeB's projection. If all projections fit, then shapeA is def in shapeB.
		If not, we assume B is in A until that fails. If both A in B fails and
		B in A fails, then neither is contained in the other.
	"""
	
	sepax = sep_ax(shapeA, shapeB)
	
	truth = 1 
	for oneDProjA, oneDProjB, name in sepax:
		#If A not in B, we assume B in A
		if not ((min(oneDProjB) <= min(oneDProjA)) \
			   and (max(oneDProjA) <= max(oneDProjB))):
			truth = -1
			if name == shapeB.name:
				#we assume B in A until proven otherwise. If below conditional
				#says B not in A even once
				if not ((min(oneDProjB) >= min(oneDProjA)) \
					   and (max(oneDProjA) >= max(oneDProjB))):
					truth = 0
					return truth
	return truth


def scan_shapes(shapes):
	
	#Get only the convex shapes
	valid_shapes = []
	for shape in shapes:
		if shape.is_convex():
			valid_shapes.append(shape)
		else:
			print "'{}' is not a polygon".format(shape.name)

	for shapeA in valid_shapes:
		for shapeB in valid_shapes:
			if shapeA == shapeB:
				continue

			if is_shape_in_other(shapeA, shapeB) == -1:
				print "'{}' surrounds '{}'".format(shapeA.name, shapeB.name)
			
			elif is_shape_in_other(shapeA, shapeB) == 1:
				print "'{}' is inside '{}'".format(shapeA.name, shapeB.name)

			#This basically makes sure everything is memoized while being the final conditional.
			else:
				if intersecting_shapes(shapeA, shapeB):
					print "'{}' intersects '{}'".format(shapeA.name, shapeB.name)
				else: 
					print "'{}' is separate from '{}'".format(shapeA.name, shapeB.name)


if __name__ == '__main__':
	f = open(str(sys.argv[1]), 'r')
	jshapes = json.load(f)
	shapes = [Shape(shape) for shape in jshapes['geometry']['shape']]
	scan_shapes(shapes)


	#Here I was testing the shape that the first iteration failed on. Seems fine now.
	#Called it tetris cuz it looks like a piece from tetris if you draw it out.
	#tetris = {
	#	"id":"tetris test" ,
	#	"point":[
	#		{ "x": 0, "y": 0 },
	#		{ "x": 2, "y": 0 },
	#		{ "x": 2, "y": 1 }, 
	#		{ "x": 1, "y": 1 },
	#		{ "x": 1, "y": 2 },
	#		{ "x": 0, "y": 2 }  
	#	]
	#}
	#tetris = Shape(tetris)


