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
    Calculate the center of mass for each labeled region in a 3D image without modifying the original data.

    PARAMETERS
    nifti: path to nibabel image object
    mri_img_txt: file path for precise names of each label (optional)
    """
    # Load the image and ensure data is treated as integers
    global img, mri_img_data, labelnames
    img = nib.load(nifti)
    mri_img_data = img.get_fdata()
    mri_img_data = np.round(mri_img_data).astype(int)  # Round and convert to int to ensure label integrity

    # Find all unique labels excluding the background (0)
    unique_labels = np.unique(mri_img_data)
    unique_labels = unique_labels[unique_labels != 0]
    
    centroids = {}
    for label in unique_labels:
        mask = (mri_img_data == label)
        centroid = center_of_mass(mask)
        centroids[label] = tuple(centroid)  # Store centroids as tuples

    # Handle optional text file for label names
    if 'txt' in kwargs:
        try:
            with open(kwargs['txt'], 'r') as file:
                labelnames = [line.strip().replace('\t', ' ') for line in file.readlines()]
                if len(labelnames) == len(centroids):
                    # Map labels to names if counts match
                    new_centroids = {labelnames[i]: centroids[key] for i, key in enumerate(sorted(centroids.keys()))}
                    centroids = new_centroids
                else:
                    print("ERROR: # of labels in the TXT file does not match # of unique labels. Ignoring TXT file.")
        except Exception as e:
            print(f"Error reading text file: {e}")

    return centroids


def visualize(axis):
    """
    visualize MRI mosaic along a specified axis (axis = x, y, or z)
    """
    try:
        mri_img_data
    except NameError:
        color.cprint('MRI data not loaded. Please run get_com() first.', 'red')
        return
    if axis == 'x':
        print('visualizing x axis')
        num_slices = mri_img_data.shape[0]
        slice_selection = lambda i: mri_img_data[i, :, :]
    elif axis == 'y':
        print('visualizing y axis')
        num_slices = mri_img_data.shape[1]
        slice_selection = lambda i: mri_img_data[:, i, :]
    elif axis == 'z':
        print('visualizing z axis')
        num_slices = mri_img_data.shape[2]
        slice_selection = lambda i: mri_img_data[:, :, i]
    else:
        color.cprint('Invalid axis. Please enter x, y, or z.', 'red')
        return

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

def mark_centroids(output_nifti_path, output_label_file, output_color_file, block_size=1):
    """
    Marks centroids on an MRI image with a visible block around each centroid for better visualization

    Parameters:
    output_nifti_path: Path to save the modified NIfTI file.
    output_label_file: Path to save new labels.
    output_color_file: Path to save color codes.
    block_size: Determines the size of the marked block around each centroid (default 1).
    """
    updated_labels = []
    color_map = {}
    white_intensity = 255 

    for i, label in enumerate(labelnames):
        region_mask = mri_img_data == (i + 1)
        if np.any(region_mask):
            centroid = np.round(np.mean(np.argwhere(region_mask), axis=0)).astype(int)
            x, y, z = centroid

            new_label = f'{label}_CTX'
            updated_labels.append(new_label)
            color = np.random.randint(0, 256, size=3)
            color_map[new_label] = color

            for dx in range(-block_size, block_size + 1):
                for dy in range(-block_size, block_size + 1):
                    for dz in range(-block_size, block_size + 1):
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if 0 <= nx < mri_img_data.shape[0] and 0 <= ny < mri_img_data.shape[1] and 0 <= nz < mri_img_data.shape[2]:
                            mri_img_data[nx, ny, nz] = white_intensity

    new_img = nib.Nifti1Image(mri_img_data, img.affine, img.header)
    nib.save(new_img, output_nifti_path)

    with open(output_label_file, 'w') as f:
        for label in updated_labels:
            f.write(f"{label}\n")

    with open(output_color_file, 'w') as f:
        for label, color in color_map.items():
            f.write(f"{color[0]} {color[1]} {color[2]} 100\n")
        f.write("255 255 255 255\n")

def compare():
    pass