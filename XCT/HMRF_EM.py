import numpy as np
from skimage.restoration import estimate_sigma
from scipy.ndimage import correlate
from skimage.filters import threshold_otsu

def simple_binary_algorithm(data, thresh, num_dims):
    """
        Only useful for binary case
        Input:
        ------------------------------------------------------
        data: numpy array, flattened input image
        num_dims: number of dimension
        thresh: threshold 
        Output:
        ------------------------------------------------------
        label: initial labelled 
        mu: update centroid of each cluster
        sigma: update variance of each cluster
    """
    binary_clusters = 2
    mu = np.zeros((binary_clusters, num_dims))
    sigma  = np.zeros((binary_clusters, num_dims, num_dims))
    
    mu[0] = np.mean(data[data<=thresh], axis=0)
    mu[1] = np.mean(data[data>thresh], axis=0)
    sigma[0] = np.cov(data[data<=thresh].T)
    sigma[1] = np.cov(data[data>thresh].T)
    label = data>thresh
    
    return label, mu, sigma

def otsu_binary_algorithm(data, num_dims):
    """
        Only useful for binary case
        Input:
        ------------------------------------------------------
        data: numpy array, flattened input image
        num_dims: number of dimension
        Output:
        ------------------------------------------------------
        label: initial labelled 
        mu: update centroid of each cluster
        sigma: update variance of each cluster
    """
    otsu_clusters = 2
    thresh = threshold_otsu(data)
    mu = np.zeros((otsu_clusters, num_dims))
    sigma  = np.zeros((otsu_clusters, num_dims, num_dims))
    
    mu[0] = np.mean(data[data<=thresh], axis=0)
    mu[1] = np.mean(data[data>thresh], axis=0)
    sigma[0] = np.cov(data[data<=thresh].T)
    sigma[1] = np.cov(data[data>thresh].T)
    label = data>thresh
    
    return label, mu, sigma

def kmeans_initial_guess(data, num_samples, num_clusters):
    """
        Input:
        ------------------------------------------------------
        data: numpy array, flattened input image
        num_samples: number of pixels in image
        num_clusters: number of clusters (i.e. 2 in binary segementation)
        Output:
        ------------------------------------------------------
        mu: numpy array, randomly pick num_clusters value from data as
            initial guess
    """
    mu = data[np.random.choice(num_samples, num_clusters, False),:]
    return mu

def kmeans_get_label(data, num_samples, num_clusters, mu):
    """
        Input:
        ------------------------------------------------------
        data: numpy array, flattened input image
        num_samples: number of pixels in image
        num_clusters: number of clusters (i.e. 2 in binary segementation)
        mu: numpy array, centroid of each cluster
        Output:
        ------------------------------------------------------
        label: set all dataset points (pixels) to the best cluster according
            to minimal distance from centroid of each cluster
    """
    dist = np.zeros((num_samples, num_clusters))
    for k in range(num_clusters):
        dist[:,k] = np.linalg.norm(data - mu[k], axis=1)

    label = np.argmin(dist, axis=1)
    return label

def kmeans_get_means(data, num_dims, num_clusters, label):
    """
        Input:
        ------------------------------------------------------
        data: numpy array, flattened input image
        num_dims: number of dimension
        num_clusters: number of clusters (i.e. 2 in binary segementation)
        label: label for dataset points (pixels) to the best cluster according
            to minimal distance from centroid of each cluster
        Output:
        ------------------------------------------------------
        mu: update centroid of each cluster
        sigma: update variance of each cluster
    """
    mu = np.zeros((num_clusters, num_dims))
    sigma  = np.zeros((num_clusters, num_dims, num_dims))
    for k in range(num_clusters):
        idx_list = np.where(label == k)[0]
        if(len(idx_list) == 0):
            mu[k] = data[np.random.randint(len(data)), :]
            sigma[k] = np.eye(num_dims)
        else:
            mu[k] = np.mean(data[idx_list], axis = 0)
            sigma[k] = np.cov(data[idx_list].T) # Remember to check this
    return mu, sigma

def kmeans_calc_loss(data, num_samples, num_clusters, mu, label):
    """
        Input:
        ------------------------------------------------------
        data: numpy array, flattened input image
        num_samples: number of pixels in image
        num_clusters: number of clusters (i.e. 2 in binary segementation)
        label: label for dataset points (pixels) to the best cluster according
            to minimal distance from centroid of each cluster
        mu: updated centroid of each cluster
        Output:
        ------------------------------------------------------
        return loss to determine convergence criterion
    """
    loss = 0
    for k in range(num_clusters):
        idx_list = np.where(label == k)[0]
        loss += np.sum(np.linalg.norm(data[idx_list]-mu[k], axis=1))
    return loss

def k_means_algorithm(data, num_samples, num_clusters, num_dims, verbose=False):
    """
        Input:
        ------------------------------------------------------
        data: numpy array, flattened input image
        num_samples: number of pixels in image
        num_clusters: number of clusters (i.e. 2 in binary segementation)
        num_dims: number of dimension of data, always 1 for pixels or voxels
        Output:
        ------------------------------------------------------
        labels: a list contains all segementation label from each iteration
        losses: a list contains all loss from each iteration
        mu: a list contains centroid of each cluster from each iteration
        sigma: a list contains variance of each cluster from each iteration
    """
    losses = []
    iter_cnt = 0
    epsilon = 1e-4
    max_iters = 100
    update = 2*epsilon

    #initial guess
    mus = [kmeans_initial_guess(data, num_samples, num_clusters)]
    sigmas = [np.eye(num_dims)*num_clusters]

    while (update > epsilon) and (iter_cnt < max_iters):
        iter_cnt += 1
        # Assign labels to each datapoint based on cnetroid
        label = kmeans_get_label(data, num_samples, num_clusters, mus[-1])

        # Assign centroid based on labels
        mu, sigma = kmeans_get_means(data, num_dims, num_clusters, label)
        mus.append(mu)
        sigmas.append(sigma)
        
        # Calculate loss
        losses.append(kmeans_calc_loss(data, num_samples, num_clusters, mus[-1], label))

        # Check convergence
        if iter_cnt >= 2:
            update = np.abs(losses[-1]-losses[-2])

        # Logging
        if verbose:
            print('iteration {}, update {}'.format(iter_cnt, update))
    if verbose:
        print('mu = {}, losses = {}'.format(mus[-1].flatten(), losses))

    return label, mus[-1], sigmas[-1], losses[-1]

def gaussian_U(data, mu, sigma, w):
    """
        Input:
        ------------------------------------------------------
        data: numpy array, flattened input image
        mu: centroid of each cluster
        sigma: variance of each cluster
        w: weight ratio between void and particle
        Output:
        ------------------------------------------------------
        return gaussian energy on each pixels
    """
    return 0.5*np.einsum('i,ij,ji->i',w,data-mu,np.linalg.inv(sigma)@(data-mu).T)+0.5*np.log(np.linalg.det(sigma))

def clique_U(label, c, img_size):
    """
        Input:
        ------------------------------------------------------
        data: numpy array, flattened input image
        label: hidden configuration corresponds to cluster labels
        c: a particular cluster (i.e. 0 for background, 1 for forground)
        Output:
        ------------------------------------------------------
        return clique potential energy on each pixel
    """
    return np.array([clique_potential(i, label.reshape(img_size), c) for i in np.arange(len(label))])

def clique_U2(label, c, img_size):
    temp = label.reshape(img_size)
    tag1 = temp == c
    tag2 = temp != c
    constraints = np.zeros(img_size)
    constraints[tag1] = 0.
    constraints[tag2] = 0.5
    # kernel = np.array([[[0.,0.,0.],[0.,1.,0.],[0.,0.,0.]],[[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]],[[0.,0.,0.],[0.,1.,0.],[0.,0.,0.]]])
    kernel = np.array([[[1.,1.,1.],[1.,1.,1.],[1.,1.,1.]],[[1.,1.,1.],[1.,0.,1.],[1.,1.,1.]],[[1.,1.,1.],[1.,1.,1.],[1.,1.,1.]]])
    return correlate(constraints, kernel, mode='nearest').flatten()

# find a way to speed up this step
def clique_potential(idx, label, c):
    """
        This put spatial constraints on pixel, pixels are more likely to have
        same labels as their surrounding pixels.
        Input:
        ------------------------------------------------------
        idx: index of pixel of interest
        label: hidden configuration corresponds to cluster labels
        c: a particular cluster (i.e. 0 for background, 1 for forground)
        Output:
        ------------------------------------------------------
        return clique potential energy on that pixel
    """
    m, n, l = label.shape
    i = idx // (n*l)
    j = idx % (n*l) // l
    k = idx % (n*l) % l
    u = 0
    if i >= 1:
        u += (c != label[i-1, j, k])/2.
    if i < m-1:
        u += (c != label[i+1, j, k])/2.
    if j >= 1:
        u += (c != label[i, j-1, k])/2.
    if j < n-1:
        u += (c != label[i, j+1, k])/2.
    if k >= 1:
        u += (c != label[i, j, k-1])/2.
    if k < l-1:
        u += (c != label[i, j, k+1])/2.
    return u

def EM_initial_guess(data, num_samples, num_clusters, num_dims):
    """
        run this for multiple time, make sure good initialization
        EM alogirthm is very sensitive to initialization
        use K-means to determine initial means and variances
        Input:
        ------------------------------------------------------
        data: numpy array, flattened input image
        num_samples: number of pixels in image
        num_clusters: number of clusters (i.e. 2 in binary segementation)
        num_dims: number of dimension of data, always 1 for pixels or voxels
        Output:
        ------------------------------------------------------
        label: final segementation label from each iteration
        mu: final centroid of each cluster from each iteration
        sigma: final variance of each cluster from each iteration
    """
    label, mu, sigma = otsu_binary_algorithm(data, num_dims)
    #label, mu, sigma = simple_binary_algorithm(data, 30, num_dims)
    #label, mu, sigma, loss = k_means_algorithm(data, num_samples, num_clusters, num_dims, verbose=False)
    return label, mu, sigma

def EM_E_step(num_clusters, num_samples, data, label, mu, sigma, beta, img_size, w):
    """
        Main goal: compute posterior of each cluster
        Input:
        ------------------------------------------------------
        data: numpy array, flattened input image
        num_samples: number of pixels in image
        num_clusters: number of clusters (i.e. 2 in binary segementation)
        label: hidden label configuration
        mu: means for each cluster in finite mixture model
        sigma: variance for each cluster in finite mixture model
        beta: ratio between Gaussian energy and clique potential energy
        img_size: image size, used to reconstruct spatial relationship
        Output:
        ------------------------------------------------------
        Uarg: cluster label corresponds to minimal energy for each pixel
        Umin: minimum energy of current configuration
        Q: posterior corresponds to each cluster
    """
    U1 = np.zeros((num_samples, num_clusters)) # Gaussian Energy
    U2 = np.zeros((num_samples, num_clusters)) # clique potential Energy

    for k in range(num_clusters):
        U1[:,k] = gaussian_U(data, mu[k], sigma[k], w)
        U2[:,k] = clique_U2(label, k, img_size)
        #U2[:,k] = clique_U(label, k, img_size)

    U = U1 + U2*beta
    Umin = np.min(U, axis = 1)
    Uarg = np.argmin(U, axis = 1)
    Uenergy = np.sum(Umin)

    # Consier Q2 as prior
    Q1 = np.zeros((num_samples, num_clusters)) # Gaussian probability
    Q2 = np.zeros((num_samples, num_clusters)) # clique potential probability

    for k in range(num_clusters):
        Q1[:,k] = (2*np.pi)**(-data.shape[1]/2.) * np.exp(-U1[:,k])
        Q2[:,k] = np.exp(-U2[:,k])
    Q2 = (Q2.T / np.sum(Q2, axis = 1)).T

    Q = Q1 * Q2
    Q = (Q.T / np.sum(Q, axis = 1)).T
    return Uarg, Uenergy, Q

def EM_M_step(num_clusters, num_dims, num_samples, Q, data, w):
    """
        Main goal: update parameters
        Input:
        ------------------------------------------------------
        data: numpy array, flattened input image
        num_dims: number of dimension
        num_samples: number of pixels in image
        num_clusters: number of clusters (i.e. 2 in binary segementation)
        Q: posterior corresponds to each cluster
        w: weight ratio between void and particle
        Output:
        ------------------------------------------------------
        mu: updated mean
        sigma: updated variance
    """
    # M Step
    ## calculate the new mean and covariance for each gaussian by
    ## utilizing the new responsibilities
    mu      = np.zeros((num_clusters, num_dims))
    sigma   = np.zeros((num_clusters, num_dims, num_dims))

    ## The number of datapoints belonging to each gaussian
    num_samples_per_cluster = np.sum(Q, axis=0)
    weighted_sum = np.einsum('i,ij->j',w,Q)

    for k in range(num_clusters):
        ## means
        #mu[k] = 1./weighted_sum[k] * np.sum(w*Q[:,k]*data.T, axis = 1).T
        mu[k] = 1./weighted_sum[k] * np.einsum('i,i,ij -> j', w, Q[:,k], data)
        centered_data = np.matrix(data-mu[k])

        ## covariances
        #sigma[k] = np.array(1./num_samples_per_cluster[k]*np.dot(np.multiply(centered_data.T, w*Q[:,k]), centered_data))
        sigma[k] = 1./num_samples_per_cluster[k] * np.multiply(centered_data.T, w*Q[:,k])@centered_data

    return mu, sigma

def EM_calc(num_dims, num_samples, num_clusters, data, img_size, void_weight=1., beta = 1., verbose=False):
    """
        This is an almost antomotic method except one might need to select void_weight;
        void_weight could be meaningfully interpreted as sample density;
        Input:
        ------------------------------------------------------
        data: numpy array, flattened input image
        num_dims: number of dimension
        num_samples: number of pixels in image
        num_clusters: number of clusters (i.e. 2 in binary segementation)
        Q: posterior corresponds to each cluster
        void_weight: weight ratio between void and particle
        Output:
        ------------------------------------------------------
        label: hidden configuration
        energy_sum: energy_sum, used to determine convergence
        mu: updated mean
        sigma: updated variance
    """
    energy_sums     = []
    iter_cnt        = 0
    epsilon         = 1
    max_iters       = 30
    update          = 2*epsilon
    #beta            = 1.
    w               = np.ones(len(data))

    # initial guess
    label, mu, sigma = EM_initial_guess(data, num_samples, num_clusters, num_dims)
    label = label.astype(np.bool)
    mus    = [mu]
    sigmas = [sigma]

    while (update > epsilon) and (iter_cnt < max_iters):
        iter_cnt += 1

        # E - Step
        label, Uenergy, Q = EM_E_step(num_clusters, num_samples, data, label, mu, sigma, beta, img_size, w)
        label = label.astype(np.bool)
        energy_sums.append(Uenergy)

        # M - Step
        mu, sigma = EM_M_step(num_clusters, num_dims, num_samples, Q, data, w)

        mus.append(mu)
        sigmas.append(sigma)


        # check convergence
        if iter_cnt >= 2 :
            update = np.abs(energy_sums[-1] - energy_sums[-2])

        # logging
        if verbose:
            print("iteration {}, update {}, mean {}".format(iter_cnt, update, mu.flatten()))

        w = np.ones(len(data))
        # label == np.argmin(mu) corresponds to void
        w[label == np.argmin(mu)] = void_weight

    print('required {} steps to finish, mu = {}'.format(iter_cnt, mus[-1].flatten()))
    energy = energy_sums[-1]
    mu = mus[-1]
    sigma = sigmas[-1]
    label = label != np.argmin(mu)
    #label = label == np.argmax(mu)
    return label, energy_sums, {'mu': mu, 'sigma': sigma}

