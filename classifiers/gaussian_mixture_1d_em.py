"""1D Gaussian mixture model for random points solved with EM algorithm ."""
from __future__ import division, print_function
import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(44)

K = 4
NB_POINTS = 50
NB_ITERATIONS = 100

def main():
    """
    Generate random points and fit 1D gaussians to them.
    Basic formulas are from https://www.youtube.com/watch?v=iQoXFmbXRJA .
    """
    # generate random points
    points_x = np.array([random.random() * 100 for _ in range(NB_POINTS)])
    points = [(x, 0) for x in points_x]

    # initialize gaussians randomly
    gaussians = [(random.random()*100, random.random()*10) for _ in range(K)]
    print(gaussians)

    plot(points, gaussians)

    # perform EM for N iterations
    for _ in range(NB_ITERATIONS):
        gaussians = em_iteration(points_x, gaussians)
        print(gaussians)

    plot(points, gaussians)

    print("Finished")

def em_iteration(points_x, gaussians):
    """Perform one iteration of EM given some points and gaussians."""
    # Estimate for each point a and cluster/gaussian c the probability p(a|c)
    clustering = np.zeros((points_x.shape[0], len(gaussians)))
    for i in range(points_x.shape[0]):
        xi = points_x[i]
        p_xi_clusters = gauss_probs(xi, gaussians)
        total_prob = sum(p_xi_clusters)

        for k in range(len(gaussians)):
            p_xi_cluster = p_xi_clusters[k]
            p_cluster = 1
            p_cluster_xi = (p_xi_cluster * p_cluster) / total_prob
            clustering[i][k] = p_cluster_xi

    # sum up for each cluster/gaussian its total probability of all points
    cluster_prob_sums = np.sum(clustering, axis=0)

    # update mu/mean and sigma of each cluster/gaussian
    new_gaussians = []
    for k in range(len(gaussians)):
        cluster = clustering[:, k]
        cluster_prob = cluster_prob_sums[k]
        new_mu = np.dot(cluster, points_x) / cluster_prob
        new_sig = np.sqrt(np.dot(cluster, (points_x - new_mu)**2) / cluster_prob)
        new_gaussians.append((new_mu, new_sig))

    return new_gaussians

def gauss_probs(xi, gaussians):
    """Estimate for a point a and gaussians C the probabilities p(a|c)."""
    probs = []
    for mu, sig in gaussians:
        probs.append(gauss_prob(xi, mu, sig))
    return probs

def gauss_prob(xi, mu, sig):
    """For a single point a and a gaussian c estimate p(a|c)."""
    return gaussian(np.array([xi]), mu, sig)[0]

def gaussian(x, mu, sig):
    """For a set of points X and a gaussian c estimate p(xi|c) (for each point)."""
    return (1/np.sqrt(2*np.pi*sig**2)) * np.exp((-(x - mu)**2) / (2 * sig**2))

def plot(points, gaussians):
    """Plot the example points and gaussians."""
    plt.scatter([x for (x, y) in points], [y for (x, y) in points], color="black")

    for mu, sig in gaussians:
        plt.plot(gaussian(np.linspace(0, 100, 100), mu, sig))

    plt.show()

if __name__ == "__main__":
    main()
