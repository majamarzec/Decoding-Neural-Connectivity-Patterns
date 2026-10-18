import config as c
import pandas as pd
import numpy as np
import seaborn as sns
import mne
import matplotlib.pyplot as plt
from sklearn.linear_model import MultiTaskLasso
import networkx as nx
import matplotlib as mpl
from mne_connectivity.viz import plot_connectivity_circle 
import logging
from pandas import DataFrame, read_csv

def read_metadata(metadata_path: str = c.metadata_path) -> DataFrame:
    return read_csv(metadata_path)

###
def load_eeg_memmap():
    mm = np.memmap(c.signals_path, 
                dtype = c.dtype, 
                mode = 'r', #only reading
                shape = (c.N_exams, c.N_windows, c.N_channels, c.Window_len*c.Fs))
    return mm

def load_s00_conn_memmap():
    meta = read_metadata()
    i_exams = np.where(np.asarray(meta['site']) == 'S00')[0]
    n_exams = len(i_exams)
    mm = np.memmap(c.s00_conn_path, 
                dtype = c.dtype, 
                mode = 'r', #only reading
                shape = (n_exams, c.N_windows, c.N_channels, c.N_channels, len(c.freq_bands)))
    return mm, i_exams, n_exams

def plot_epoch_eeg(exam, start_epoch, end_epoch, n_epochs_to_plot, scalings=dict(eeg=50e-6)):
    info = mne.create_info(c.channels, sfreq=c.Fs, ch_types="eeg")  
    epochs = mne.EpochsArray(exam, info, tmin=0, baseline=None) #type:ignore
    g = epochs[start_epoch:end_epoch].plot(n_epochs=n_epochs_to_plot, scalings = scalings)
    return g

def plot_one_window(window, scalings = dict(eeg=50e-6)):
    info = mne.create_info(c.channels, sfreq=c.Fs, ch_types="eeg")  
    epoch = mne.io.RawArray(window, info) 
    g = epoch.plot(scalings = scalings)
    return g

###
def zscore_per_channel(data: np.ndarray) -> np.ndarray:
    """Z-score normalization per channel; epsilon=1e-9 prevents division by zero.

    Parameters: 
    data : np.ndarray -> One frame (channels, samles in time).

    Returns:
    np.ndarray Data z-scored per channel of the same shape as input.
    """
    means = np.mean(data, axis=1, keepdims=True)
    stds  = np.std(data,  axis=1, keepdims=True)
    return (data - means) / (stds + 1e-9)

###
def countCorr(x, ip, iwhat = 2):
    '''
    Internal procedure building autocovariance & cross-correlation diagonal-constant matrix.

    Inputs
    ______
    x: np.ndarray
       eeg data of shape (channels, samples)
    
    ip: int
        model order, maximum lag

    iwhat: int = {1,2}
           stable biased or unstable unbiased estimator

    Returns
    _______
    rleft_tot (Block-Toeplitz Autocovariance Matrix / LHS): 
              Builds a massive symmetric block matrix where each block is the autocovariance at a specific lag. 
              This matrix has a Toeplitz structure (meaning diagonal blocks repeat), which is a mathematical property of stationary time series autocovariance.

    rright_tot (Cross-covariance / RHS): Packs the cross-covariances between current values and past lagged values across all requested lags (ip).

    r_tot (Zero-lag covariance): Computes the instantaneous covariance matrix across all channels at lag 0 (np.dot(x, x.T)).

    '''
    sdt = np.shape(x)
    assert sdt == (17, 6*128)
    m = sdt[0] #channels
    n = sdt[1] #samples

    mip = m * ip
    rleft = np.zeros((mip, mip))
    rright = np.zeros((mip, m))
    r = np.zeros((m, m))
    rleft_tot = np.zeros((mip, mip))
    rright_tot = np.zeros((mip, m))
    r_tot = np.zeros((m, m))


    for k in range(ip):
        if iwhat == 1: 
            corrscale = 1 / n # it guarantees that the resulting Block-Toeplitz covariance matrix (Rleft) is positive semi-definite
            nn = n - k - 1
            r[:, :] = np.dot(x[:, :nn], x[:, k + 1:k + nn + 1].T) * corrscale
        elif iwhat == 2:
            corrscale = 1 / (n - k)
            nn = n - k - 1
            r[:, :] = np.dot(x[:, :nn], x[:, k + 1:k + nn + 1].T) * corrscale
        
        rright[k * m:k * m + m, :] = r[:, :]
        
        if k < ip:
            for i in range(1, ip - k):
                rleft[(k + i) * m:(k + i) * m + m, (i - 1) * m:(i - 1) * m + m] = r[:, :]
                rleft[(i - 1) * m:(i - 1) * m + m, (k + i) * m:(k + i) * m + m] = r[:, :].T
        
        corrscale = 1 / n
        r[:, :] = np.dot(x[:, :], x[:, :].T) * corrscale
    
        for k in range(ip):
            rleft[k * m:k * m + m, k * m:k * m + m] = r[:, :]
    
        rleft_tot = rleft_tot + rleft
        rright_tot = rright_tot + rright
        r_tot = r_tot + r

    return rleft_tot, rright_tot, r_tot

###
def AR_coeff(dat):
    '''
    Estimates AR model coefficients for multivariate data.

    Inputs
    ______
    dat: np.ndarray 
         input time series epoch of shape (channels = 17,samples in time = 768)

    returns:
    ARr - model coefficients of size (channels_in,channels_out, p), here channels_in = channels_out = channels
    Vr - residual data variance matrix (p,channels_in,channels_out) which is a spectral power density of the noise
    '''

    rleft, rright, r = countCorr(dat, c.p) #constructing the covariance matrices
    xres = np.linalg.solve(rleft, rright)  
    xres= xres.T  
    Vr = -np.dot(xres, rright) + r
    AR = np.reshape(xres, (c.N_channels, c.p, c.N_channels)) # the reshape: (chans, p, chans)) ->  (chans, chans, p)) 
    AR = AR.transpose((0, 2, 1)) # (target_ch, source_ch, lag)
    return AR, Vr

#experiment, not successful
#better to enforce graph sparcity afterwards than during least squares optimization
def AR_coeff_lasso(dat: np.ndarray, p: int = 9, alpha: float = 0.01) -> tuple:
    """
    Estimates sparse MVAR model coefficients using MultiTaskLasso.

    Parameters:
    -----------
    dat : np.ndarray
         Input time series of shape (channels = 17, samples = 768) 
         or (channels, samples, trials).
    p : int
        Model order (number of lags). Default is 9.
    alpha : float
        L1 regularization penalty strength. 
        - Higher alpha = sparser network (more zero connections).
        - Lower alpha = closer to standard OLS. Try values like 0.001 to 0.1.

    Returns:
    --------
    AR : np.ndarray
        Coefficient tensor of shape (channels_out, channels_in, p) -> (17, 17, 9)
    Vr : np.ndarray
        Residual variance matrix of shape (channels_out, channels_out) -> (17, 17)
    """
    if dat.ndim == 3:
        chans, samples, trials = dat.shape
        dat_flattened = dat.reshape(chans, samples * trials)
    else:
        chans, samples = dat.shape
        dat_flattened = dat

    n_eff = samples - p
    Y = dat_flattened[:, p:].T  


    X = np.zeros((n_eff, chans * p))
    for lag in range(1, p + 1):
        X[:, (lag - 1) * chans : lag * chans] = dat_flattened[:, p - lag : samples - lag].T

    lasso = MultiTaskLasso(alpha=alpha, fit_intercept=False, max_iter=3000, tol=1e-4)
    lasso.fit(X, Y)

    coef_reshaped = lasso.coef_.reshape(chans, p, chans)
    AR = coef_reshaped.transpose(0, 2, 1)

    Y_pred = lasso.predict(X)
    residuals = Y - Y_pred
    Vr = np.dot(residuals.T, residuals) / n_eff

    return AR, Vr

###
def mvar_H(Ar):
    """
    Calculate the transfer function H from multivariate autoregressive coefficients Ar.
    
    Parameters:
    Ar : numpy.ndarray
         AR coefficient matrix with shape (chan, chan, p), where p is model order.

    Returns:
    H : numpy.ndarray
        Transfer function matrix with shape (chan, chan, len(f)).
    """

    m_vector = np.arange(1, c.p + 1).reshape(c.p, 1) #lag vector
    z = np.exp(-1j * 2 * np.pi * m_vector * c.f / c.Fs) # Z transform (p,1) -> (p, Nf) (into frequency domain)

    # Ar shape: (chan, chan, p), z shape: (p, Nf) -> sum_term shape: (chan, chan, Nf)
    # 'ijp' = target, source, lag | 'pf' = lag, frequency -> 'ijf' = target, source, frequency
    sum_term = np.einsum('ijp,pf->ijf', Ar, z) #Einstein convention summation
    I = np.eye(c.N_channels, dtype=complex)[..., np.newaxis]
    A_out = I - sum_term
    A_f = np.moveaxis(A_out, -1, 0)

    try:
        H_f = np.linalg.inv(A_f)
    except np.linalg.LinAlgError:
        H_f = np.full_like(A_f, np.nan)
        for fi in range(A_f.shape[0]):
            try:
                H_f[fi] = np.linalg.inv(A_f[fi])
            except np.linalg.LinAlgError:
                pass          # this frequency stays NaN
    return np.moveaxis(H_f, 0, -1) 

### from mtmvar package
def ffDTF(H):
    """
    Compute the full-frequency directed transfer function (ffDTF) for the multivariate case.
    
    The ffDTF modifies the standard DTF normalization to integrate over the entire frequency 
    range, making cross-frequency comparisons more interpretable. Unlike standard DTF which 
    normalizes by frequency-specific inflows, ffDTF normalizes by the sum over all frequencies.
    
    Mathematical formula:
    Standard DTF: DTF_ij(f) = |H_ij(f)|² / Σ_k |H_ik(f)|²
    Full-frequency DTF: ffDTF_ij(f) = |H_ij(f)|² / Σ_f Σ_k |H_ik(f)|²
    
    The normalization takes into account inflows to channel i across the entire frequency range,
    allowing for more meaningful comparison of information flow at different frequencies.
    
    Reference:
    Korzeniewska, A., Mańczak, M., Kamiński, M., Blinowska, K. J., & Kasicki, S. (2003). 
    Determination of information flow direction among brain structures by a modified directed transfer function (dDTF) method. 
    Journal of neuroscience methods, 125(1-2), 195-207.
    https://doi.org/10.1016/S0165-0270(03)00052-9
    
        
    Returns:
    np.ndarray
        Full-frequency DTF of shape (N_chan, N_chan, N_f).
    """

    DTF = np.abs(H)**2
    # Normalize NDTF to get full-frequency normalised DTF (ffDTF)
    ffDTF = DTF / DTF.sum(axis=(1, 2), keepdims=True)
    return ffDTF

###
#for plotting
def aggregate_bands(connectivity, pool=np.sum, transpose=True):
    """
    Collapse (N_chan, N_chan, N_f) into one (N_chan, N_chan) matrix per band.

    reduce    : np.sum or np.mean
    transpose : True -> entry [row, col] means row -> col (same as graph_plot)
    Returns   : {band title: np.ndarray (N_chan, N_chan)}
    """
    out = {}
    for title, (lo, hi) in c.freq_bands.items():
        idx = np.where((c.f >= lo) & (c.f <= hi))[0]
        if idx.size == 0:
            raise ValueError(f"No frequencies found in range {(lo, hi)} ({title}).")
        m = pool(connectivity[:, :, idx], axis=2)
        m = m.T.copy() if transpose else m.copy()
        np.fill_diagonal(m, 0)
        out[title] = m
    return out

#for features
def bands_to_image(band_mats):
    """{band: (N, N)} -> (N, N, N_bands), band order = c.freq_bands order."""
    return np.stack(list(band_mats.values()), axis=-1)
###
def feature_pipeline(window, feature_type = 'ffDTF', aggregation_method = np.sum):
    window_zscored = zscore_per_channel(window)
    AR, _ = AR_coeff(window_zscored)
    H = mvar_H(AR)
    if feature_type == 'NDTF':
        feature = np.abs(H)**2
    if feature_type == 'ffDTF':
        feature = ffDTF(H)
    feature_bands = aggregate_bands(feature, pool = aggregation_method) #type:ignore
    return bands_to_image(feature_bands)
###

def plot_one_flow(connectivity, from_channel, to_channel, ylabel = "|H_ij(f)|² / Σ_f Σ_k |H_ik(f)|²", conn_name = "ffDTF"):
    plt.figure(figsize=(10, 4))
    plt.plot(c.f, connectivity[from_channel, to_channel, :])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(ylabel)
    plt.title(f"{conn_name}: {c.channels[from_channel]} → {c.channels[to_channel]}")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_all_flows(connectivity, ylabel = "|H_ij(f)|² / Σ_f Σ_k |H_ik(f)|²", conn_name = "ffDTF"):
    fig, ax = plt.subplots(c.N_channels, c.N_channels, figsize=(24, 24), sharex=True, sharey=True)
    for i in range(c.N_channels):
        for j in range(c.N_channels):
            if i == j:
                ax[i, j].plot(c.f, np.zeros_like(c.f), color='black', linestyle='--')
            else: 
                ax[i, j].plot(c.f, connectivity[i, j, :])
            if i == 0:
                ax[i, j].set_title(c.channels[j], fontsize=12)
            if j == 0:
                ax[i, j].set_ylabel(c.channels[i], fontsize=12)
            ax[i, j].tick_params(labelsize=10)

    fig.suptitle(f"{conn_name} for an EEG window", fontsize=18, y=1.002)
    fig.supxlabel("Frequency (Hz)", fontsize = 14)
    fig.supylabel(ylabel, fontsize = 14)

    plt.tight_layout()
    plt.show()

###
def heatmap_bands(mats, cmap='viridis', conn_name = 'ffDTF'):
    fig, axes = plt.subplots(1, 5 , figsize=(15, 5), squeeze=False)
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        sns.heatmap(mats[:, :, i].T, ax=ax, xticklabels=c.channels, yticklabels=c.channels,
                    cmap=cmap, square=True, cbar=False)
        ax.set_title(list(c.freq_bands.keys())[i])
        ax.tick_params(axis='x', rotation=90, labelsize=8)
        ax.tick_params(axis='y', rotation=0, labelsize=8)

    sm = plt.cm.ScalarMappable(cmap=cmap) #type:ignore
    fig.colorbar(sm, ax=axes, shrink=0.65, pad=0.02, label='Connectivity strength')
    #fig.suptitle(f"Band-aggregated {conn_name} for an EEG window", fontsize=10, y=1)
    return fig

def heatmap_bands_dict(mats, cmap='viridis', conn_name = 'ffDTF'):
    titles = list(mats)
    vmin = min(m.min() for m in mats.values())
    vmax = max(m.max() for m in mats.values())

    fig, axes = plt.subplots(1, len(mats), figsize=(4.4 * len(mats), 5), squeeze=False)
    axes = axes.ravel()
    for ax, title in zip(axes, titles):
        sns.heatmap(mats[title].T, ax=ax, xticklabels=c.channels, yticklabels=c.channels,
                    cmap=cmap, vmin=vmin, vmax=vmax, square=True, cbar=False)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=90, labelsize=8)
        ax.tick_params(axis='y', rotation=0, labelsize=8)

    sm = plt.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax), cmap=cmap) #type:ignore
    fig.colorbar(sm, ax=axes, shrink=0.65, pad=0.02, label='Connectivity strength')
    fig.suptitle(f"Band-aggregated {conn_name} for an EEG window", fontsize=10, y=1.002)
    return fig


def circle_bands_dict(mats,cmap='viridis', conn_name = 'ffDTF'):
    titles = list(mats)
    vmin = min(m.min() for m in mats.values())
    vmax = max(m.max() for m in mats.values())

    fig, axes = plt.subplots(1, len(mats), figsize=(4 * len(mats), 5),
                             subplot_kw={'projection': 'polar'}, squeeze=False)
    axes = axes.ravel()
    for ax, title in zip(axes, titles):
        plot_connectivity_circle(
            con=mats[title], node_names=c.channels, title=title,
            colormap=cmap, vmin=vmin, vmax=vmax, ax=ax, show=False,
            colorbar=False, facecolor='white', textcolor='black',
        )

    sm = plt.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax), cmap=cmap) #type:ignore
    fig.colorbar(sm, ax=axes, shrink=0.65, pad=0.08, label='Connectivity strength')
    fig.suptitle(f"Band-aggregated {conn_name} for an EEG window", fontsize=18, y=1.002)
    return fig

### graphs
def montage_pos():
    """Return {node index: (x, y)} using real electrode positions (top-down view)."""
    montage = mne.channels.make_standard_montage(c.montage)
    ch_pos = montage.get_positions()['ch_pos']          
    pos = {}
    for i, name in enumerate(c.channels):
        if name not in ch_pos:
            raise KeyError(f"'{name}' not in montage '{c.montage}'")
        x, y, _ = ch_pos[name]
        pos[i] = (x, y)                                  # x = right, y = front -> nose points up
    return pos


def get_linewidths(G, vmax=None):
    weights = np.array([d['weight'] for _, _, d in G.edges(data=True)])
    if weights.size == 0: #no weights (for dropped too weak connections)
        return 1.0                      
    if vmax is None:
        vmax = weights.max()        
    return 5 * weights / vmax


def threshold_bands(mats, threshold=None, top_percent=None, mode='global'):
    """
    mode='per_band': strongest X% inside each band (same edge count in every band)
    mode='global'  : strongest X% over all bands pooled (edge count differs per band)
    threshold      : absolute cutoff, same for every band
    """
    n = next(iter(mats.values())).shape[0]
    off = ~np.eye(n, dtype=bool)

    if top_percent is not None and mode == 'global':
        pooled = np.concatenate([m[off] for m in mats.values()])
        global_cutoff = np.percentile(pooled, 100 - top_percent)

    out = {}
    for title, m in mats.items():
        m = m.copy()
        if top_percent is not None:
            cutoff = (global_cutoff if mode == 'global' #type:ignore
                      else np.percentile(m[off], 100 - top_percent))
            m[m < cutoff] = 0
        if threshold is not None:
            m[m < threshold] = 0
        out[title] = m
    return out


def graph_bands(mats, conn_name = 'ffDTF'):
    """
    Plot a connectivity matrix as one directed graph per frequency band,
    each in its own subplot.

    Parameters
    ----------
    connectivity : np.ndarray, shape (N_chan, N_chan, N_f)
        Connectivity (e.g. NDTF or ffDTF).
    seed : int
        Seed for the node layout, so the picture is reproducible.
    reduce : callable
        np.sum (default) or np.mean, used to collapse the frequencies in a band.

    Returns
    -------
    graphs : dict
        {band title: networkx.DiGraph}
    """
    labels = {i: name for i, name in enumerate(c.channels)}
    n = len(mats)
    pos = montage_pos() 

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 7), squeeze=False)
    axes = axes.ravel()

    graphs = {}
    for ax, (title, adj) in zip(axes, mats.items()):
        G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
        nx.draw_networkx(
            G, pos, ax=ax, with_labels=True, labels=labels,
            arrows=True, width=get_linewidths(G), node_size=500,
            arrowstyle='->', arrowsize=35, connectionstyle='arc3,rad=0.2',
        )
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

        graphs[title] = G

    fig.tight_layout()
    fig.suptitle(f"Band-aggregated {conn_name} directed graphs", fontsize=18, y=1.1, fontweight='bold')
    return graphs


def graph_bands_sparse(mats, top_percent=10, threshold=None, shared_scale=True, mode = 'global', conn_name = 'ffDTF'):
    """
    Plot one sparse directed graph per frequency band, each in its own subplot,
    keeping only the strongest connections.

    Parameters
    ----------
    mats : dict {band title: np.ndarray (N_chan, N_chan)}
        Output of aggregate_bands (row -> column, zero diagonal).
    top_percent : float or None
        Keep the strongest X% of edges in each band.
    threshold : float or None
        Absolute cutoff on the weight (applied after top_percent if both given).
    shared_scale : bool
        True  -> line widths comparable across panels (one global max).
        False -> each panel scaled to its own strongest edge.

    Returns
    -------
    graphs : dict
        {band title: networkx.DiGraph}
    """
    sparse = threshold_bands(mats, threshold, top_percent, mode=mode)


    labels = {i: name for i, name in enumerate(c.channels)}
    n = len(sparse)
    pos = montage_pos()
    vmax = max(m.max() for m in sparse.values()) if shared_scale else None

    #for plot visibility
    xs, ys = zip(*pos.values())
    mx, my = 0.15 * (max(xs) - min(xs)), 0.15 * (max(ys) - min(ys))
    xlim, ylim = (min(xs) - mx, max(xs) + mx), (min(ys) - my, max(ys) + my)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 7), squeeze=False)
    axes = axes.ravel()

    graphs = {}
    for ax, (title, adj) in zip(axes, sparse.items()):
        G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
        nx.draw_networkx(
            G, pos, ax=ax, with_labels=True, labels=labels,
            arrows=True, width=get_linewidths(G, vmax), node_size=500,
            arrowstyle='->', arrowsize=35, connectionstyle='arc3,rad=0.2',
        )
        ax.set_xlim(xlim)      
        ax.set_ylim(ylim)
        ax.set_title(f"{title}  ({G.number_of_edges()} edges)",
                     fontsize=12, fontweight='bold')
        ax.axis('off')

        graphs[title] = G

    fig.tight_layout()
    fig.suptitle(f"Band-aggregated {conn_name} directed sparse graphs in {mode} mode of connections choice", fontsize=18, y=1.1, fontweight='bold')
    return graphs

### from EEGraph
def compute_graph_metrics(G):
    """Compute graph-theoretic metrics on a single NetworkX graph.

    All metrics are computed on an undirected, self-loop-free copy of the
    graph regardless of whether the original is directed (e.g. DTF) or
    contains self-loops (e.g. single-channel measures).

    Parameters
    ----------
    G : NetworkX Graph or DiGraph
        A brain connectivity graph produced by EEGraph.

    Returns
    -------
    metrics : dict
        Dictionary with the following keys.

        ── Network-level scalars ─────────────────────────────────────────────

        density : float  [0, 1]
            Fraction of all possible electrode pairs that share an active
            connection above the chosen threshold.

            *Brain connectivity interpretation*: Overall level of
            synchronisation in the network. Higher density is observed during
            cognitive tasks or epileptic seizures (hyper-synchrony). Abnormally
            low density is reported in Alzheimer's disease and disorders of
            consciousness. Because density depends on the threshold, always
            compare recordings using the same threshold value.

        transitivity : float  [0, 1]
            Global clustering coefficient — ratio of closed triangles to all
            connected triplets across the whole network.

            *Brain connectivity interpretation*: Captures the tendency of
            electrodes to form tightly interconnected local clusters (functional
            modules). The healthy resting brain has high transitivity, reflecting
            local specialisation. Reduced transitivity has been linked to
            schizophrenia and TBI; elevated transitivity appears in some forms
            of epilepsy.

        average_clustering : float  [0, 1]
            Mean of each node's local clustering coefficient, weighted by edge
            strength.

            *Brain connectivity interpretation*: Similar to transitivity but
            gives equal weight to every electrode regardless of degree. Useful
            for detecting localised changes in functional modularity. Decreased
            average clustering in the alpha band has been reported in
            Alzheimer's disease and Parkinson's disease.

        global_efficiency : float  [0, 1]
            Average inverse shortest path length over all node pairs.
            Unlike average_path_length, this measure is well-defined for
            disconnected graphs (isolated pairs contribute 0).

            *Brain connectivity interpretation*: Reflects how efficiently
            information is integrated across the whole brain (parallel
            processing capacity). The healthy brain maintains high global
            efficiency. Reductions are a consistent finding in Alzheimer's
            disease, multiple sclerosis, and following stroke. An increase
            may indicate seizure-related hyper-connectivity.

        local_efficiency : float  [0, 1]
            Average global efficiency computed within each node's immediate
            neighbourhood (i.e. efficiency of information transfer when that
            node is removed).

            *Brain connectivity interpretation*: Measures the fault tolerance
            and local information-processing capacity of the network. High
            local efficiency means the network remains functional even if
            individual electrodes are compromised. Decreased local efficiency
            has been observed in Parkinson's disease (particularly in beta
            band) and in ageing.

        average_path_length : float  [≥ 1]
            Mean number of edges along the shortest path between all pairs of
            nodes. Computed on the largest connected component when the graph
            is disconnected; NaN if the graph has fewer than 2 connected nodes.

            *Brain connectivity interpretation*: Reflects global integration —
            how rapidly activity in one brain region can influence another.
            The healthy brain shows short path lengths (rapid integration).
            Increased path length indicates fragmented or less integrated
            connectivity and is reported in Alzheimer's disease, depression,
            and following white-matter lesions. Note: compare together with
            average_clustering via small_world_sigma.

        degree_assortativity : float  [-1, 1]
            Pearson correlation between the degrees of connected node pairs.
            Positive values: high-degree nodes tend to connect to other
            high-degree nodes. Negative values: hubs connect to low-degree
            nodes (disassortative). NaN if undefined.

            *Brain connectivity interpretation*: Most biological brain networks
            are disassortative (negative values) — hub electrodes connect to
            peripheral ones, which supports robustness and integration. A shift
            towards assortativity (hubs connecting only to hubs) may reflect
            pathological reorganisation, such as that seen in epilepsy or
            following focal lesions.

        small_world_sigma : float
            Small-world coefficient σ = (C / C_rand) / (L / L_rand), where
            C_rand ≈ ⟨k⟩/n and L_rand ≈ ln(n)/ln(⟨k⟩) are analytical
            approximations for a random graph with the same size and mean
            degree. σ > 1 indicates small-world organisation.
            NaN if not computable (e.g. too few edges or disconnected).

            *Brain connectivity interpretation*: The healthy brain operates as
            a small-world network (high clustering + short path lengths),
            striking an optimal balance between local specialisation and global
            integration. σ significantly above 1 has been confirmed in resting
            EEG across all frequency bands. Loss of small-world topology
            (σ approaching 1) is a robust biomarker reported in Alzheimer's
            disease, schizophrenia, epilepsy, and major depression.

        ── Node-level dicts  {electrode_label: value} ───────────────────────

        degree_centrality : dict  [0, 1]
            Fraction of other electrodes each node is directly connected to.

            *Brain connectivity interpretation*: Identifies electrodes with the
            largest number of active connections. Frontal and parietal hub
            electrodes typically show the highest degree centrality in the
            healthy resting brain. Shifts in degree centrality (e.g. from
            frontal to temporal regions) can indicate pathological
            reorganisation in epilepsy or dementia.

        betweenness_centrality : dict  [0, 1]
            Fraction of all shortest paths (across the full graph) that pass
            through each node, weighted by edge strength.

            *Brain connectivity interpretation*: Identifies bottleneck
            electrodes — those that mediate communication between otherwise
            distant brain regions. High betweenness centrality marks
            functional hubs. Loss of hub status in key regions (e.g. precuneus,
            posterior cingulate in Alzheimer's) or the emergence of new hubs
            (e.g. perilesional areas after stroke) are clinically meaningful
            signatures.

        eigenvector_centrality : dict  [0, 1]
            A node's influence weighted by the centrality of its neighbours —
            being connected to important nodes increases a node's own
            centrality. NaN per node if power iteration fails to converge.

            *Brain connectivity interpretation*: Captures global hub status
            more sensitively than degree centrality because it accounts for
            the quality, not just the quantity, of connections. Electrodes
            overlying the default mode network (medial frontal, posterior
            parietal) show high eigenvector centrality at rest. Reduced
            eigenvector centrality in these regions has been associated with
            cognitive decline and Alzheimer's disease.

    Notes
    -----
    When comparing metrics across subjects or conditions, ensure that the
    same connectivity measure, frequency band, and threshold are used
    throughout, as all graph metrics are sensitive to network density.
    Proportional thresholding (fixing density across recordings) is
    recommended for group-level analyses.

    References
    ----------
    Stam, C.J. et al. (2007). Small-world networks and functional connectivity
        in Alzheimer's disease. *Cerebral Cortex*, 17, 92–99.
    Bassett, D.S. & Bullmore, E. (2006). Small-world brain networks.
        *The Neuroscientist*, 12(6), 512–523.
    Rubinov, M. & Sporns, O. (2010). Complex network measures of brain
        connectivity: uses and interpretations. *NeuroImage*, 52(3), 1059–1069.
    Latora, V. & Marchiori, M. (2001). Efficient behavior of small-world
        networks. *Physical Review Letters*, 87(19), 198701.
    """
    # Work on an undirected, self-loop-free copy for all metrics
    G_und = G.to_undirected() if nx.is_directed(G) else G
    G_clean = nx.Graph(G_und)
    G_clean.remove_edges_from(nx.selfloop_edges(G_clean))

    metrics = {}

    # ------------------------------------------------------------------ #
    # Network-level scalars                                                #
    # ------------------------------------------------------------------ #
    metrics['density']             = nx.density(G_clean)
    metrics['transitivity']        = nx.transitivity(G_clean)
    metrics['average_clustering']  = nx.average_clustering(G_clean, weight='weight')
    metrics['global_efficiency']   = nx.global_efficiency(G_clean)
    metrics['local_efficiency']    = nx.local_efficiency(G_clean)

    # Average path length — requires a connected graph; use largest CC otherwise
    n = len(G_clean)
    if n > 1 and G_clean.number_of_edges() > 0:
        if nx.is_connected(G_clean):
            metrics['average_path_length'] = nx.average_shortest_path_length(G_clean, weight=None)
        else:
            largest_cc = G_clean.subgraph(
                max(nx.connected_components(G_clean), key=len)
            ).copy()
            if len(largest_cc) > 1:
                metrics['average_path_length'] = nx.average_shortest_path_length(largest_cc, weight=None)
                logging.warning(
                    'Graph is disconnected. average_path_length computed on '
                    'the largest connected component (%d / %d nodes).',
                    len(largest_cc), n
                )
            else:
                metrics['average_path_length'] = float('nan')
    else:
        metrics['average_path_length'] = float('nan')

    # Degree assortativity
    try:
        metrics['degree_assortativity'] = nx.degree_assortativity_coefficient(G_clean)
    except Exception:
        metrics['degree_assortativity'] = float('nan')

    # Small-world sigma: σ = (C / C_rand) / (L / L_rand)
    # Analytical approximations: C_rand ≈ <k> / n,  L_rand ≈ ln(n) / ln(<k>)
    if n > 1 and G_clean.number_of_edges() > 0:
        k_mean = np.mean([d for _, d in G_clean.degree()])
        C = metrics['average_clustering']
        L = metrics['average_path_length']
        if k_mean > 1 and not np.isnan(L) and L > 0:
            C_rand = k_mean / n
            L_rand = np.log(n) / np.log(k_mean)
            metrics['small_world_sigma'] = (C / C_rand) / (L / L_rand) if C_rand > 0 else float('nan')
        else:
            metrics['small_world_sigma'] = float('nan')
    else:
        metrics['small_world_sigma'] = float('nan')

    # ------------------------------------------------------------------ #
    # Node-level centrality dicts                                          #
    # ------------------------------------------------------------------ #
    metrics['degree_centrality']       = nx.degree_centrality(G_clean)
    metrics['betweenness_centrality']  = nx.betweenness_centrality(G_clean, weight='weight')

    try:
        metrics['eigenvector_centrality'] = nx.eigenvector_centrality(
            G_clean, weight='weight', max_iter=1000
        )
    except nx.PowerIterationFailedConvergence:
        logging.warning('Eigenvector centrality did not converge. Returning NaN for all nodes.')
        metrics['eigenvector_centrality'] = {node: float('nan') for node in G_clean.nodes()}

    metrics['closeness_centrality'] = nx.closeness_centrality(G_clean)

    metrics['node_strength'] = {
        node: float(sum(d.get('weight', 1.0) for _, d in G_clean[node].items()))
        for node in G_clean.nodes()
    }

    metrics['degree'] = dict(G_clean.degree())

    # ── Modularity ────────────────────────────────────────────────────────────
    if G_clean.number_of_edges() > 0:
        try:
            communities = nx.community.greedy_modularity_communities(
                G_clean, weight='weight'
            )
            metrics['modularity'] = float(
                nx.community.modularity(G_clean, communities, weight='weight')
            )
        except Exception:
            metrics['modularity'] = float('nan')
    else:
        metrics['modularity'] = 0.0

    # ── Rich-club coefficient ─────────────────────────────────────────────────
    if G_clean.number_of_nodes() >= 2:
        try:
            rcc = nx.rich_club_coefficient(G_clean, normalized=False)
            metrics['rich_club_coefficient'] = {int(k): float(v) for k, v in rcc.items()}
        except Exception:
            metrics['rich_club_coefficient'] = {}
    else:
        metrics['rich_club_coefficient'] = {}

    return metrics
def compute_metrics_all(graphs):
    """Compute graph-theoretic metrics for every graph in the EEGraph output dict.

    Parameters
    ----------
    graphs : dict
        Dictionary of NetworkX graphs as returned by ``Graph.modelate()``.

    Returns
    -------
    all_metrics : dict
        Dictionary ``{graph_index: metrics_dict}`` where each value is the
        output of :func:`compute_graph_metrics`.
    """
    return {k: compute_graph_metrics(G) for k, G in graphs.items()}
