import numpy

def row_norms(matrix, squared=False):
    norms = numpy.einsum('ij,ij->i', matrix, matrix)
    if not squared: numpy.sqrt(norms, norms)
    return norms

def safe_dot(x, y):
    if x.ndim > 2 or y.ndim > 2:
        result = numpy.dot(x, y)
    else:
        result = x @ y
    return result

def generate_batches(n, batch_size, min_batch_size=0):
    start = 0
    for _ in range(int(n // batch_size)):
        end = start + batch_size
        if end + min_batch_size > n: continue
        yield slice(start, end)
        start = end
    if start < n:
        yield slice(start, n)

def euclidean_distance(x, y=None, squared=False):
    if y is None: y = x
    distances = euclidean_distances_upcast(x, y)
    numpy.maximum(distances, 0, out=distances)
    if x is y: numpy.fill_diagonal(distances, 0)
    return distances if squared else numpy.sqrt(distances, out=distances)

def euclidean_distances_upcast(x, y=None):
    n_samples_x = x.shape[0]
    n_samples_y = y.shape[0]
    n_features = x.shape[1]

    distances = numpy.empty((n_samples_x, n_samples_y), numpy.float32)

    x_density, y_density = 1, 1

    maxmem = max(
        ((x_density * n_samples_x + y_density * n_samples_y) * n_features
         + (x_density * n_samples_x * y_density * n_samples_y)) / 10,
        10 * 2 ** 17)
    tmp = (x_density + y_density) * n_features
    batch_size = (-tmp + numpy.sqrt(tmp ** 2 + 4 * maxmem)) / 2
    batch_size = max(int(batch_size), 1)
    x_batches = generate_batches(n_samples_x, batch_size)

    for i, x_slice in enumerate(x_batches):
        x_chunk = x[x_slice].astype(numpy.float64)
        xx_chunk = row_norms(x_chunk, squared=True)[:, numpy.newaxis]

        y_batches = generate_batches(n_samples_y, batch_size)

        for j, y_slice in enumerate(y_batches):
            if x is y and j < i:
                d = distances[y_slice, x_slice].T
            else:
                y_chunk = y[y_slice].astype(numpy.float64)
                yy_chunk = row_norms(y_chunk, squared=True)[numpy.newaxis, :]
                d = -2 * safe_dot(x_chunk, y_chunk.T)
                d += xx_chunk
                d += yy_chunk
            distances[x_slice, y_slice] = d.astype(numpy.float32, copy=False)
    return distances

def k_init(self, dataset, n_clusters, random_state):
    '''
    :param dataset: array
    :param n_clusters: int
    :param random_state: int
    :return: None
    '''
    n_data, n_features = dataset.shape
    centers = numpy.empty((n_clusters, n_features), dataset.dtype)
    n_local_trials = 2 + int(numpy.log(n_clusters))
    center_id = random_state.randint(n_data)

    centers[0] = dataset[center_id]
    closest_distance = self.euclidean_distance(centers[0, numpy.newaxis], x, squared=True)
    current_potential = closest_distance.sum()

    for point in range(1, n_clusters):
        random_values = random_state.random_sample(n_local_trials) * current_potential
        candidate_ids = numpy.searchsorted(numpy.cumsum(closest_distance, dtype=numpy.float64), random_values)

        numpy.clip(candidate_ids, None, closest_distance.size - 1, out=candidate_ids)

        distance_to_candidates = self.euclidean_distance(x[candidate_ids], x, squared=True)

        numpy.minimum(closest_distance, distance_to_candidates, out=distance_to_candidates)
        candidates_potential = distance_to_candidates.sum(axis=1)

        best_candidate = numpy.argmin(candidates_potential)
        current_potential = candidates_potential[best_candidate]
        closest_distance = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        centers[point] = x[best_candidate]
    return centers

def _init_center(data, k, init='k-means++', random_state=None):
    n_samples = data.shape[0]
    if init == 'k-means++':
        centers = k_init(data, k, random_state)
    elif init == 'random':
        seeds = random_state.permutation(n_samples)[:k]
        centers = data[seeds]
    return centers

def _lables_inertia(data, centers, distances=None):
    n_samples = data.shape[0]
    labels = numpy.full(n_samples, -1, numpy.int32)
    if distances == None: distances = numpy.zeros(shape=(0,), dtype=data.dtype)
    inertia = 1
    return labels, inertia

class KMeans(object):
    def __init__(self, n_clusters=2, init='k-means++', n_init=10, max_iteration=300, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iteration = max_iteration
        self.random_state = random_state

    def _kmeans_single_lloyd(self, data, n_clusters, max_iteration=300, init='k-means++', random_state=None):
        best_lables, best_inertia, best_centers = None, None, None
        centers = _init_center(data, n_clusters, init, random_state)
        distances = numpy.zeros(shape=(data.shape[0],), dtype=data.dtype)

        for i in range(max_iteration):
            centers_old = centers.copy()
            labels, inertia = 

    def fit(self, x, y=None):
        n_init = self.n_init
        init = self.init
        random_state = self.random_state

        x_mean = x.mean(axis=0)
        x -= x_mean

        x_squared_norms = row_norms(x, squared=True)

        best_lables, best_inertia, best_centers = None, None, None
        if self.n_clusters == 1:
            kmeans_single = _kmeans_single_lloyd
        else:
            kmeans_single = _kmeans_single_dlkan

        seeds = random_state.randint(numpy.iinfo(numpy.int32).max, size=n_init)
        for seed in seeds:
            lables, inertia, centers, n_iteration = kmeans_single()



if __name__ == '__main__':
    X = [[0, 1], [1, 1], [1, 2]]
    x = numpy.asarray(X)
    print(euclidean_distance(x, x))
