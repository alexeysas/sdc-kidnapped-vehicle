/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	num_particles = 50;
	particles = vector<Particle>(num_particles);

	//cout << std[0] << ' ' << std[1] << ' ' << std[2] << endl;

	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	// Add random Gaussian noise to each particle.
	for (unsigned int i = 0; i < num_particles; i++) {
		Particle particle = Particle();
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1;
		particles[i] = particle;
	}

	is_initialized = true;
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	//cout << "Prediction start" << endl;

	default_random_engine gen;
	
	for (unsigned int i = 0; i < num_particles; i++) {

		Particle particle = particles[i];

		//cout << "Move parameters: v=" << velocity << " yaw_rate =" << yaw_rate << "  dt=" << delta_t << endl;
		//cout << "Before Move: x=" << particle.x << " y=" << particle.y << "  tata" << particle.theta << endl;

		if (fabs(yaw_rate) > 0.0001) {
			particle.x = particle.x + velocity / yaw_rate * (sin(particle.theta + delta_t * yaw_rate) - sin(particle.theta));
			particle.y = particle.y + velocity / yaw_rate * (cos(particle.theta) - cos(particle.theta + delta_t * yaw_rate));
		}
		else {
			particle.x = particle.x + velocity * cos(particle.theta) * delta_t;
			particle.y = particle.y + velocity * sin(particle.theta) * delta_t;
		}
		particle.theta = particle.theta + yaw_rate * delta_t;
		
		// Add control noise
		normal_distribution<double> dist_x(particle.x, std_pos[0]);
		normal_distribution<double> dist_y(particle.y, std_pos[1]);
		normal_distribution<double> dist_theta(particle.theta, std_pos[2]);

		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta= dist_theta(gen);

		particles[i] = particle;


		//cout << "After Move: x=" << particle.x << " y=" << particle.y << "  tata" << particle.theta << endl;
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (unsigned int i = 0; i < observations.size(); i++) {
		double min_distance = numeric_limits<double>::max();
		int min_id = 0;
		double ox = observations[i].x;
		double oy = observations[i].y;
		int index = 0;

		for (unsigned int j = 0; j < predicted.size(); j++) {
			double px = predicted[j].x;
			double py = predicted[j].y;

			double distance = sqrt((ox - px) * (ox - px) + (oy - py) * (oy - py));

			if (distance < min_distance) {
				min_distance = distance;
				min_id = predicted[j].id;
				index = j;
			}
		}

		observations[i].id = min_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
	const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	
	for (unsigned int i = 0; i < num_particles; i++) {
		
		//transform observations from vehicle to map coordinate
		std::vector<LandmarkObs> transformed_observations(observations.size());

		for (unsigned int j = 0; j < observations.size(); j++) {
			LandmarkObs transformed_observation = LandmarkObs();
			std::pair<double, double> xy_pair = ParticleFilter::transform(particles[i].x, particles[i].y, particles[i].theta, observations[j].x, observations[j].y);
			transformed_observation.x = xy_pair.first;
			transformed_observation.y = xy_pair.second;
			transformed_observations[j] = transformed_observation;
		}

		// prepare landmarks
		std::vector<LandmarkObs> landmarks = std::vector<LandmarkObs>();
		for (unsigned int i = 0; i < map_landmarks.landmark_list.size(); i++) {
			LandmarkObs landmark = LandmarkObs();
			landmark.x = map_landmarks.landmark_list[i].x_f;
			landmark.y = map_landmarks.landmark_list[i].y_f;
			landmark.id = map_landmarks.landmark_list[i].id_i;

			double distance = sqrt((landmark.x - particles[i].x) * (landmark.x - particles[i].x) + (landmark.y - particles[i].y) * (landmark.y - particles[i].y));

			//try to restrict landmarks to verify
			if (distance < sensor_range + 3 * sqrt(std_landmark[0] * std_landmark[0] + std_landmark[1] * std_landmark[1]))
			{
				landmarks.push_back(landmark);
			}
		}

		//create assotiations and sent them to debug engine
		dataAssociation(landmarks, transformed_observations);

		std::vector<int> assotiations = std::vector<int>(observations.size());
		std::vector<double> sense_x = std::vector<double>(observations.size());
		std::vector<double> sense_y = std::vector<double>(observations.size());

		for (unsigned int j = 0; j < observations.size(); j++) {
			assotiations[j] = transformed_observations[j].id;
			sense_x[j] = transformed_observations[j].x;
			sense_y[j] = transformed_observations[j].y;
		}

		SetAssociations(particles[i], assotiations, sense_x, sense_y);

		// initialize weight
		particles[i].weight = 1.0;
		
		//Update the weights of each particle using a mult-variate Gaussian distribution.
		for (unsigned int j = 0; j < transformed_observations.size(); j++) {
			LandmarkObs closestLandmark = find_landmark(landmarks, transformed_observations[j].id);
		
		ParticleFilter::
				particles[i].weight *= multivariate_gaussian(transformed_observations[j].x, transformed_observations[j].y, closestLandmark.x, closestLandmark.y, std_landmark[0], std_landmark[1]);
		
		}
	}
}

std::pair<double, double> ParticleFilter::transform(double x_particle, double y_particle, double  heading_particle, double x_obs, double y_obs) {
	std::pair<double, double> res;
	res.first = x_particle + x_obs * cos(heading_particle) - y_obs * sin(heading_particle);
	res.second = y_particle + x_obs * sin(heading_particle) + y_obs * cos(heading_particle);
	return res;
}

double ParticleFilter::multivariate_gaussian(double x, double y, double mx, double my, double sx, double sy) {
	double res = 1.0 / (2.0 * M_PI * sx * sy) * exp(-((x - mx) * (x - mx) / (2.0 * sx * sx) + (y - my) * (y - my) / (2.0 * sy * sy)));
	return res;
}

LandmarkObs ParticleFilter::find_landmark(std::vector<LandmarkObs> landmarks, int id) {
	
	for (unsigned int i = 0; i < landmarks.size(); i++) {
		if (id == landmarks[i].id) {
			return landmarks[i];
		}
	}

	throw std::invalid_argument("landmark with providede id does not exist");
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::random_device rd;
	std::mt19937 gen(rd());

	std::vector<double> weights = std::vector<double>(num_particles);
	
	for (unsigned int i = 0; i < num_particles; i++) {
		weights[i] = particles[i].weight;
	}

	std::discrete_distribution<> d(weights.begin(), weights.end());
	std::vector<Particle> newParticles = std::vector<Particle>(num_particles);

	for (unsigned int i = 0; i < num_particles; i++) {
		int index = d(gen);
		newParticles[i] = particles[index];
	}

	particles = newParticles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
	const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;

}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
