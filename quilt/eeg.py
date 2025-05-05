import warnings
import pickle 

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import signal
from scipy.stats import zscore

from sklearn.feature_selection import mutual_info_regression
from rich.progress import track

from mne.filter import filter_data
from mne.viz import plot_topomap

from sklearn.neighbors import NearestNeighbors
from quilt.builder import OscillatorNetwork

def entropy_knn(data, k=3):
    """Entropy for continuous data (Kraskov, 2004) using kNN"""
    n = len(data)
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(data.reshape(-1, 1))
    distances, _ = nbrs.kneighbors(data.reshape(-1, 1))
    avg_log_dist = np.mean(np.log(distances[:, -1]))
    return -avg_log_dist + np.log(n) + np.log(2)

def entropy(prob_dist):
    """Brutal entropy by homogeneous binning"""
    return -np.sum(prob_dist * np.log2(prob_dist + 1e-10))

def mutual_information(x, y, bins=20, normalize=True):
    """Brutal MI by homogeneous binning"""
    hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)  
    py = np.sum(pxy, axis=0)
    Hx = entropy(px)
    Hy = entropy(py)
    Hxy = entropy(pxy.flatten())
    MI = Hx + Hy - Hxy
    if normalize:
        MI = 2*MI/(Hx+Hy)
    return MI

def azimuthal_equidistant_projection(positions_3d):
        positions_3d = np.array(positions_3d)
        positions_3d /= np.linalg.norm(positions_3d, axis=1, keepdims=True)
        x, y, z = positions_3d[:, 0], positions_3d[:, 1], positions_3d[:, 2]
        
        theta = np.arccos(z) 
        phi = np.arctan2(y, x)
        
        r = theta / np.pi
        x_2d = r * np.cos(phi)
        y_2d = r * np.sin(phi)
        return np.column_stack((x_2d, y_2d))



def sort_electrodes(electrode_names, position_map=None, return_dims=False):
    left = []
    center = []
    right = []

    for electrode in electrode_names:
        if electrode[-1] == 'z':
            center+= [electrode]
        else:
            n = int(electrode[-1])
            if n%2 ==0:
                right += [electrode]
            else:
                left += [electrode]
    
    if position_map is not None:
        ## Sorted by rostro-caudal dimension
        order = lambda x: position_map[x][1] 
        left = sorted(left, key=order, reverse=True)
        center = sorted(center, key=order, reverse=True)
        right = sorted(right, key=order, reverse=True)
    else:
        # Sorts by alphabetical order
        left = sorted(left)
        center = sorted(center)
        right = sorted(right)
    
    result = (right + center + left,)

    # Adds the counts if requested
    if return_dims:
        result += (dict(right=len(right), center=len(center), left=len(left)),)

    return result


class EEGPSDholder:
    def __init__(self, psd, f, channel_names, position_map):
        self.psd = psd
        self.channel_names = channel_names
        self.position_map = position_map
        self.positions = np.array([self.position_map[ch] for ch in self.channel_names])
        self.positions2Dazim = azimuthal_equidistant_projection(self.positions)
        self.positions2Dproj = self.positions[:, 0:2]

        self.positions2Dproj /= np.linalg.norm(self.positions2Dproj, keepdims=True)
        self.positions2Dazim /= np.linalg.norm(self.positions2Dazim, keepdims=True)

        self.f = f

        # Normalization
        for i in range(len(channel_names)):
            self.psd[i] = self.psd[i] / np.trapz(self.psd[i], x=f)

    def bandpower(self, band):
        
        bandpowers = np.zeros(self.psd.shape[0])
        fmask = (self.f>band[0])&(self.f<band[1])
        for i in range(len(self.channel_names)):
            bandpowers[i] = np.trapz(self.psd[i][fmask],x=self.f[fmask] )

        return bandpowers
    
    def plot_band_topomap(self, band, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        
        return plot_topomap(self.bandpower(band), self.positions2Dazim, 
                            axes=ax, 
                            contours=0,
                            sphere=0.15, image_interp='cubic')
    

class EEGholder:
    """Aggregate class for EEG in raw format. This class is Deprecated.

    Attributes:
        - data: the raw eeg object (n_channels, n_datapoints), can be both electric field or estimated scalp current density (Laplacian filter)
        - t: time of the timeseries
        - f: frequency in PSD
        - psds: PSD for each channel (n_channles, n_fourier_points)
    """
    def __init__(self, file, channel_names, channel_map, fs, zscore=False):
        """Initializes the EEGholder. 
        
        Arguments:
            - file: file (.npy) containing the raw recordings. Do not specify file extension.
            - channel_names: the file (.npy) containing the names of the channels in order. Do not specify file extension. 
            - channel_map: a pickled (.pkl) dictionary that maps each channel name to a position in 3d space. See Oostenveld's blog at https://robertoostenveld.nl/electrode/
        """
        self.file = file
        self.channel_names = channel_names 
        self.channel_map = channel_map
        self.fs = fs
        self.zscore = zscore

        # Data
        self.data = np.load(f"{file}.npy")
        self.t = np.linspace(0, len(self.data[0])/fs, len(self.data[0]))
        self.n_channels = len(self.data)

        # Channel names
        if isinstance(channel_names, str):
            self.channel_names = np.load(f"{self.channel_names}.npy")
        else:
            self.channel_names = channel_names
        print("Channel names:", self.channel_names)
        if len(self.channel_names) != self.n_channels:
            raise TypeError(f"Shape mismatch in data: names are different than data\nchannels: {len(self.data)} - names: {len(self.channel_names)}")
        
        # Channel map position
        if isinstance(channel_map, str):
            with open(channel_map+".pkl", "rb") as f:
                self.channel_map = pickle.load(f)
        else:
            self.channel_map = channel_map

        # Get channels positions
        self.channel_positions = np.array([self.channel_map[n] for n in self.channel_names])

        # Predeclare spectral attributes
        self.f = None
        self.psds = None

        # Predeclare mutual info
        self.MI = None
        self.entropy = None

        # Info
        print(f"Loaded a {self.n_channels}-channels rec")
        print(f"Datapoints: {self.data.shape[1]}")
        print(f"Total time: {self.t[-1]} seconds")
        if self.zscore:
            self.data = StandardScaler().fit_transform(self.data.T).T

    def cut_time(self, t_start, t_end, renormalize=True):

        mask = (self.t>=t_start)&(self.t<t_end)
        self.t = self.t[mask]
        self.data = self.data[:,mask]
        print(f"Cut EEG in [{t_start}, {t_end}] ({len(self.t)} datapoints)")
        if renormalize and self.zscore:
            self.data = StandardScaler().fit_transform(self.data.T).T
    
    def take_central_window(self, window_T, reset_time=True):
        t_mid = self.t[0] + 0.5*(self.t[-1] - self.t[0])
        t_up = t_mid + 0.5*window_T
        t_down = t_mid - 0.5*window_T
        mask = (self.t>t_down)&(self.t<t_up)
        self.data = self.data[:, mask]
        self.t = self.t[mask]

        if reset_time:
            self.t -= self.t[0]

        if self.zscore:
            self.data = StandardScaler().fit_transform(self.data.T).T
            
        print(f"Cut eeg in [{t_down:.1f},{t_up:.1f}]")
    
    def channel_data(self, channel):
        mask = self.channel_names == channel
        return self.data[mask][0]

    def drop_channel(self, channel):
        mask = self.channel_names != channel
        self.data = self.data[mask]
        self.n_channels -= 1

    def compute_psd(self):
        """Computes the PSD of each channel. By default the spectral resolution is 0.5Hz."""
        self.psds = []
        for i in range(self.n_channels):
            f, psd = signal.welch(self.data[i], fs=self.fs, nperseg=2*self.fs, noverlap=self.fs)
            self.psds.append(psd)
        self.psds = np.array(self.psds)
        self.f = f
    
    def normalize_psds(self):
        for i in range(len(self.psds)):
            self.psds[i] /= np.trapz(self.psds[i], x=self.f)

    def get_power_in_band(self, band):
        mask = (self.f>band[0])&(self.f<band[1])
        pows = np.zeros(self.n_channels)
        for i in range(self.n_channels):
            pows[i] = np.trapz(self.psds[i, mask], x=self.f[mask])
        return pows

    @property
    def channel_powers(self):
        tot_power = np.zeros(len(self.psds))
        for i in range(len(self.psds)):
            tot_power[i] = np.trapz(self.psds[i], x=self.f)
        return tot_power
    
    @property
    def pos_azimuthal(self):
        return azimuthal_equidistant_projection(self.channel_positions)
    
    @property
    def pos_proj(self):
        return (self.channel_positions.T[:2]).T

    
    def compute_mutual_info(self, n_neighbors=None, thinning=None, bins=None, normalize=True, entropy_diagonal=True):
        """Computes the mutual information between each channel. 
        
        If n_neighbors and thinning are given, returns the Kraskov knn mutual info over a thinned shuffled subset of the timeseries.
        if bins are given, returns the brutal MI done by contingency table.
        """
        if n_neighbors is None and bins is None:
            raise ValueError("Specify n_neighbors or bins to select knn or default")
        if n_neighbors is not None and bins is not None:
            raise ValueError("Only one between n_neighbors and bins must be specified")
        
        MI = np.zeros((self.n_channels, self.n_channels))

        if n_neighbors:
            if thinning is None:
                print("k-nn mutual info without thinning may take a while")
                thin_list = np.arange(len(self.t))
            else:
                order = np.arange(len(self.t))
                np.random.shuffle(order)
                thin_list = order[::thinning]
                # print("thinned list", thin_list)
                
            for i in track(range(self.n_channels)):
                for j in range(i, self.n_channels):
                    if j==i:
                        if entropy_diagonal:
                            MI[i, i] = entropy_knn(self.data[i, thin_list], k=n_neighbors)
                        else:
                            MI[i,i] = 0
                    else:
                        MI[i,j] = mutual_info_regression( self.data[i, thin_list].reshape(-1, 1), self.data[j, thin_list], n_neighbors=n_neighbors)[0]
                        MI[j,i] = MI[i,j]
            

            if normalize:
                for i in range(self.n_channels):
                    for j in range(i, self.n_channels):
                        MI[i,j] = 2*MI[i, j]/(MI[i,i] + MI[j,j])
                        MI[j,i] = 2*MI[j, i]/(MI[i,i] + MI[j,j])

        if bins:
            for i in track(range(self.n_channels)):
                for j in range(i, self.n_channels):
                    if not entropy_diagonal and i==j:
                        MI[i,j] = 0
                    else:
                        MI[i,j] = mutual_information(self.data[i], self.data[j], bins=bins, normalize=normalize)
                        MI[j,i] = MI[i,j]
        self.MI = MI 
        return MI

    def compute_entropy(self, bins=50):
        self.entropy = np.zeros(self.n_channels)
        for i in range(self.n_channels):
            p, _ = np.histogram(self.data[i], bins=bins, density=False)
            p = np.array(p, dtype=float)
            p /= np.sum(p)

            self.entropy[i] = entropy(p)

        return self.entropy
    

class EEGcap:
    '''Class to compute EEG from an OscillatorNetwork
    
    Arguments
    ---------
        node_based_lfm_file: str
            the file of the nlfm data. Must be a dict having fields 'nodes' (list[str]), 'electrodes' (lis[str]) and 'nlfm' (2D array)
        position_map_file: str
            the file of the channel-position map (e.g. Oostenveld's map)
    
            
    Attributes
    ----------
        nodes: list[str]
            list of the names of the nodes
        electrodes: list[str]
            list of the names of the electrodes
        nlfm_gain: np.ndarray
            the gain matrix with shape (n_electrodes, n_nodes)

    '''
    def __init__(self, 
                node_based_lfm_file: str,
                position_map_file: str
                 ):
        # Load the NLFM data
        with open(node_based_lfm_file, 'rb') as f:
            nlfm_data = pickle.load(f)

        # Set nodes and regions
        self.nodes = nlfm_data['nodes']
        self.electrodes = nlfm_data['electrodes']
        self.nlfm_gain = nlfm_data['nlfm']

        # Load position map
        with open(position_map_file, 'rb') as f:
            self.position_map = pickle.load(f)
        
        # Compute positions
        self.electrodes_position = np.array([self.position_map[electrode] for electrode in self.electrodes])

        self.signals = None

    def eeg(self, network: OscillatorNetwork,
            filter_kwargs,
            init_to_skip=1,
            zscore_signals=True
            ):
        """Returns the EEG of the oscillator network.
        
        Arguments
        ---------
        filter_kwargs: dict
            the arguments of mne.filter.filter_data to process the EEG
        """

        # Check if the nodes of the OscillatorNetwork are the same
        difference_nodes = set(self.nodes) - set(network.oscillators.keys())
        if (len(difference_nodes)):
            warnings.warn(f"Nodes of the NLFM does not correspond to the network oscillators:\nDifferences: {difference_nodes}")


        T = len(network.oscillators[list(network.oscillators.keys())[0]].history)
        signals = np.zeros((len(self.nodes), T))

        # Takes the timeseries once to avoid overhead due to data request
        time_series = np.zeros((len(self.nodes), T))

        for i,node in enumerate(self.nodes):
            time_series[i] = network.oscillators[node].eeg # Uses the eeg method of oscillator that gives the right VOI

        signals = np.dot(self.nlfm_gain, time_series)

        if init_to_skip > 0:
            signals = signals[:, init_to_skip*int(network.tau_init/network.dt):]

        if "sfreq" in filter_kwargs:
            del  filter_kwargs['sfreq']
        signals = filter_data(signals, 
                             sfreq=1000.0/network.dt, # Time is in ms
                             **filter_kwargs)
        if zscore_signals:
            signals = zscore(signals, axis=1)
        self.signals = signals
        return signals
    
    def get_electrode_projection(self, method='equidist'):
        positions_3d = self.electrodes_position/np.linalg.norm(self.electrodes_position, axis=1, keepdims=True)

        if method == 'equidist':
            x, y, z = positions_3d[:, 0], positions_3d[:, 1], positions_3d[:, 2]
            
            theta = np.arccos(z) 
            phi = np.arctan2(y, x)
            
            r = theta / np.pi
            x_2d = r * np.cos(phi)
            y_2d = r * np.sin(phi)
            return np.column_stack((x_2d, y_2d))
        if method == 'cartesian':
            return positions_3d[:, :2]
        else:
            raise ValueError("Method must be either 'equidist' or 'cartesian'")
    
    def compute_normalized_psd(self, fmax=45):
        """Computes the PSD of each channel. By default the spectral resolution is 0.5Hz."""
        if self.signals is None:
            raise RuntimeError("Compute the eeg of a network first")
        self.psds = []

        for i in range(len(self.electrodes)):
            f, psd = signal.welch(self.signals[i], fs=self.fs, nperseg=2*self.fs, noverlap=self.fs)
            self.psds.append(psd)
        psds = np.array(self.psds)

        psds = psds[f<=fmax]
        f = f[f<=fmax]

        for i in range(len(psds)):
            self.psds[i] /= np.trapz(self.psds[i], x=self.f)

        return f, psds

