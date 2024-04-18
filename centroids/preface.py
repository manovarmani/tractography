import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import center_of_mass
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import dendrogram, linkage
from skbio.stats.ordination import pcoa
import phate

class color:
    # ANSI escape codes for various colors
    COLORS = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'end': '\033[0m', 
    }

    @staticmethod
    def cprint(text, c):
        color_code = color.COLORS.get(c, color.COLORS['end'])
        print(f"{color_code}{text}{color.COLORS['end']}")

def get_com(nifti, **kwargs):
    """
    get the center of mass of a 3D image

    PARAMETERS
    mri_img: nibabel image object
    mri_img_txt: precise names of each labels (optional)
    """
    global mri_img_data
    img = nib.load(nifti)
    mri_img_data = img.get_fdata()

    unique_labels = np.unique(mri_img_data)
    unique_labels = unique_labels[unique_labels != 0] #eliminate background
    centroids = {}
    for i, label in enumerate(unique_labels):
        mask = (mri_img_data == label)
        centroid = center_of_mass(mask)
        centroids[label] = centroid

    if 'txt' in kwargs:
        with open(kwargs['txt'], 'r') as file:
            labelnames = [line.strip().replace('\t', ' ') for line in file.readlines()]
            
        if len(labelnames) == len(centroids):
            new_centroids = {labelnames[i]: centroid for i, (key, centroid) in enumerate(centroids.items())}
            centroids = new_centroids
        else:
            color.cprint("ERR: # of labels in the TXT file does not match # of unique labels. Ignoring TXT file.", 'red')

    return centroids


def visualize(axis='z'):
    """
    visualize MRI mosaic along a specified axis (axis = x, y, or z)
    """
    if axis == 'x':
        print('visualizing x axis')
        num_slices = mri_img_data.shape[0]
        slice_selection = lambda i: mri_img_data[i, :, :]
    elif axis == 'y':
        print('visualizing y axis')
        num_slices = mri_img_data.shape[1]
        slice_selection = lambda i: mri_img_data[:, i, :]
    else:
        color.cprint('no axis specified, defaulting to z axis', 'yellow')
        num_slices = mri_img_data.shape[2]
        slice_selection = lambda i: mri_img_data[:, :, i]

    grid_size = int(np.ceil(np.sqrt(num_slices)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()

    for i in range(num_slices):
        ax = axes[i]
        ax.imshow(slice_selection(i), cmap='inferno')
        ax.set_title(f'Slice {i}')
        ax.axis('off')

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def get_distance_matrix(centroids):
    """
    get distance matrix between centroids
    """
    labels = list(centroids.keys())
    coords = list(centroids.values())

    dist_matrix = squareform(pdist(coords), 'euclidean')

    return pd.DataFrame(dist_matrix, index=labels, columns=labels)


def perform_mds(dist_matrix):
    """
    perform Multidimensional Scaling (MDS) on a given distance matrix.
    """
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    mds_coords = mds.fit_transform(dist_matrix)

    plt.figure(figsize=(8, 6))
    plt.scatter(mds_coords[:, 0], mds_coords[:, 1], color='blue')
    plt.title("MDS")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.show()


def perform_hierarchical_clustering(dist_matrix):
    """
    perform Hierarchical Clustering on a given distance matrix
    """
    linked = linkage(dist_matrix, 'single')

    plt.figure(figsize=(10, 7))
    dendrogram(linked, orientation='top', labels=np.arange(dist_matrix.shape[0]), 
               distance_sort='descending', show_leaf_counts=True)
    plt.title("Hierarchical Clustering")
    plt.xlabel("Index")
    plt.ylabel("Distance")
    plt.show()


def perform_pcoa(dist_matrix):
    """
    perform Principal Coordinate Analysis (PCoA) on a given distance matrix
    """
    pcoa_results = pcoa(dist_matrix)

    plt.figure(figsize=(8, 6))
    plt.scatter(pcoa_results.samples['PC1'], pcoa_results.samples['PC2'], color='green')
    plt.title("PCoA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.show()

def plot_heatmap(dist_matrix, labels=None):
    """
    plot heat map of a distance matrix
    """
    plt.figure(figsize=(15, 10))
    sns.heatmap(dist_matrix, annot=False, fmt=".2f", cmap='coolwarm', cbar_kws={'label': 'Distance'})
    plt.title('Heat Map of Distance Matrix')
    plt.show()

def get_phate(dist_matrix):

    dist_matrix_copy = dist_matrix.copy()
    n_neighbors = 25
    #gamma=1
    #t=10

    phate_operator = phate.PHATE(n_components = 3, knn = n_neighbors, n_jobs=-1)
    #phate_operator = phate.PHATE(n_components = 2, knn = n_neighbors, gamma=gamma, t=t, a=decay)
    ph_res_3d = phate_operator.fit_transform(dist_matrix_copy)
    ph_res_3d = pd.DataFrame(ph_res_3d, index = dist_matrix_copy.index)
    ph_res_3d = ph_res_3d.rename(columns={0: "x", 1: "y"})

    dist_matrix_copy["PHATE_1"] = ph_res_3d.x
    dist_matrix_copy["PHATE_2"] = ph_res_3d.y

    plt.figure(figsize=(8, 6))
    plt.scatter(ph_res_3d['x'], ph_res_3d['y'], c='blue', marker='o', alpha=0.5)
    plt.xlabel('PHATE 1')
    plt.ylabel('PHATE 2')
    plt.title('2D Scatter Plot of PHATE Coordinates')
    plt.grid(True)
    plt.show()

def centroid_nifti(centroids, original_image_path, output_image_path, block_size=2):
    """
    Create a new nifti image with centroids marked

    *PARAMETERS*
    centroids: dictionary of centroids
    original_image_path: path to the original nifti image
    output_image_path: path to the output nifti image
    block_size: size of the block around the centroid to be marked
    """
    
    #later see if should check if mri_img_data is None and load it
    original_img = nib.load(original_image_path)
    original_data = original_img.get_fdata()

    label_data = np.zeros_like(original_data)

    label = 1001

    half_block = block_size // 2

    for centroid in centroids.values():
        x, y, z = map(int, centroid)
        for dx in range(-half_block, half_block + 1):
            for dy in range(-half_block, half_block + 1):
                for dz in range(-half_block, half_block + 1):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < label_data.shape[0] and 0 <= ny < label_data.shape[1] and 0 <= nz < label_data.shape[2]:
                        label_data[nx, ny, nz] = label
        label += 1

    new_img = nib.Nifti1Image(label_data, original_img.affine, original_img.header)
    
    nib.save(new_img, output_image_path)


def compare():
    pass