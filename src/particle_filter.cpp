/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#define _USE_MATH_DEFINES
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static std::default_random_engine RANDOM_GEN;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles.
	num_particles = 100;
	// Initialize all particle weights 
	weights.resize(num_particles);
	// Add random Gaussian noise based to each particle
	normal_distribution<double> dist_x(0, std[0]);
	normal_distribution<double> dist_y(0, std[1]);
	normal_distribution<double> dist_theta(0, std[2]);
	// Initialize all particles to first position (based on estimates of 
	// x, y, theta and their uncertainties from GPS sampled from a gaussian)
	for (int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = x + dist_x(RANDOM_GEN);
		p.y = y + dist_y(RANDOM_GEN);
		p.theta = theta + dist_theta(RANDOM_GEN);
		p.weight = 1.0;
		particles.push_back(p);
		weights[i] = 1.0;
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise
	// (mean = updated particle position, stddev = uncertainty of mesasurement captured by std_pos).
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);
	// Move particles
	for (int i = 0; i < num_particles; i++) {
		double theta_particle = particles[i].theta;
		// no change of yaw -> straight line movement, catch division by zero
		if (fabs(yaw_rate) < 0.000001) {
			particles[i].x += cos(theta_particle) * velocity * delta_t;
			particles[i].y += sin(theta_particle) * velocity * delta_t;
			// particles[i].theta remains unchanged
		} else {
			double delta_theta = yaw_rate * delta_t;
			particles[i].x += (velocity / yaw_rate) * (sin(theta_particle + delta_theta) - sin(theta_particle));
			particles[i].y += (velocity / yaw_rate) * (cos(theta_particle) - cos(theta_particle + delta_theta));
			particles[i].theta += yaw_rate * delta_t;
		}
		// Add noise
		particles[i].x += dist_x(RANDOM_GEN);
		particles[i].y += dist_y(RANDOM_GEN);
		particles[i].y += dist_theta(RANDOM_GEN);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement (ground truth from lidar)
	// and assign the observed measurement to this particular landmark.
	for (LandmarkObs& observation: observations) {
		double dist_to_predict = -1; 
		int predict_id = -1;
		for (LandmarkObs predict : predicted) {
			double distance = dist(observation.x, observation.y, predict.x, predict.y);
			if (dist_to_predict == -1 || distance < dist_to_predict) {
				dist_to_predict = distance;
				predict_id = predict.id;
			}
		}
		observation.id = predict_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// * Predict measurements to all map landmarks within sensor range
	// * Associate sensor measurements to map landmarks using ParticleFilter::dataAssociation
	// * Calculate new weights of particles with multi variate Gaussian probability density function
	// * Normalize weights into range 0..1
	for (int i = 0; i < num_particles; i++) {
		// get particle measurements
		double particle_x = particles[i].x;
		double particle_y = particles[i].y;
		double particle_theta = particles[i].theta;
		// clear debug information
		particles[i].sense_x.clear();
		particles[i].sense_y.clear();
		particles[i].associations.clear();
		// generating list of predicted observations, landmarks are already in map coordinate system
		vector<LandmarkObs> predicted;
		for (Map::single_landmark_s landmark : map_landmarks.landmark_list) {
			// assuming sensor range squarely in both x and y direction
			if (fabs(landmark.x_f-particle_x) <= sensor_range && fabs(landmark.y_f-particle_y) <= sensor_range) {
				LandmarkObs predict;
				predict.id = landmark.id_i;
				predict.x = landmark.x_f;
				predict.y = landmark.y_f;
				predicted.push_back(predict);
			}
		}
		// list of actual observations, transformed into map coordinates
		vector<LandmarkObs> observations_transformed;
		for (LandmarkObs observation: observations) {
			LandmarkObs observation_transformed;
			observation_transformed.x = cos(particle_theta)*observation.x - sin(particle_theta)*observation.y + particle_x;;
			observation_transformed.y = sin(particle_theta)*observation.x + cos(particle_theta)*observation.y + particle_y;
			observation_transformed.id = observation.id;
			observations_transformed.push_back(observation_transformed);
		}
		// associate predicted landmarks with observed (measurement) landmarks
		dataAssociation(predicted, observations_transformed);
		// start with particle weight 1.0
		particles[i].weight = 1.0;
		// find actual x and y coordinates for each observed landmark
		for (LandmarkObs observation: observations_transformed) {
			// find corresponding predicted landmark (for current particle)
			double predict_x, predict_y;
			for (LandmarkObs predict: predicted) {
				if (predict.id == observation.id) {
					predict_x = predict.x;
					predict_y = predict.y;
					break;
				}
			}
			// calculate new weight with multi variate Gaussian probability density function
			double sigma_x = std_landmark[0];
			double sigma_y = std_landmark[1];
			double normalizer = (1 / (2 * M_PI * sigma_x * sigma_y));
			double exponent = pow(predict_x-observation.x, 2) / (2 * pow(sigma_x, 2)) + 
								pow(predict_y-observation.y, 2) / (2 * pow(sigma_y, 2));
			double weight = normalizer * exp(-exponent);
			// landmark measurements are independent, so the weight is a product 
			// for each observation_transformed inside of this for loop
			particles[i].weight *= weight;
			weights[i] = particles[i].weight;
			// add debug information
			particles[i].associations.push_back(observation.id);
			particles[i].sense_x.push_back(observation.x);
			particles[i].sense_y.push_back(observation.y);
		}
	}
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 
	// --> use "resample wheel" trick
	vector<Particle> new_particles;
	uniform_int_distribution<int> int_dist(0, num_particles-1);
	auto index = int_dist(RANDOM_GEN);
	double beta = 0.0;
	double max_weight = *max_element(weights.begin(), weights.end());
	uniform_real_distribution<double> real_dist(0.0, 2.0*max_weight);
	for (int i = 0; i < num_particles; i++) {
		beta += real_dist(RANDOM_GEN);
		while (beta > weights[index]) {
			beta -= weights[index];
			index = (index+1) % num_particles;
		}
		new_particles.push_back(particles[index]);
	}
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
