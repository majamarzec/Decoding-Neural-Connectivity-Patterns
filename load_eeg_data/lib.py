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


###
def load_memmap():
    mm = np.memmap(c.signals_path, 
                dtype = c.dtype, 
                mode = 'r', #only reading
                shape = (c.N_exams, c.N_windows, c.N_channels, c.Window_len*c.Fs))
    return mm

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
def AR_coeff(dat, p = 9, chans = 17):
    '''
    Estimates AR model coefficients for multivariate data.

    Inputs
    ______
    dat: np.ndarray 
         input time series epoch of shape (channels = 17,samples in time = 768)
    p: int
        model order, as desribed, here equals 9

    returns:
    ARr - model coefficients of size (channels_in,channels_out, p), here channels_in = channels_out = channels
    Vr - residual data variance matrix (p,channels_in,channels_out) which is a spectral power density of the noise
    '''

    rleft, rright, r = countCorr(dat, p) #constructing the covariance matrices
    xres = np.linalg.solve(rleft, rright)  
    xres= xres.T  
    Vr = -np.dot(xres, rright) + r
    AR = np.reshape(xres, (chans, p,chans)) # the reshape: (chans, p, chans)) ->  (chans, chans, p)) 
    AR = AR.transpose((0, 2, 1)) # (target_ch, source_ch, lag)
    return AR, Vr

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
    # 1. Handle multi-trial vs single-trial data shapes
    if dat.ndim == 3:
        chans, samples, trials = dat.shape
        # Concatenate trials along the time axis to build a single large design matrix
        dat_flattened = dat.reshape(chans, samples * trials)
    else:
        chans, samples = dat.shape
        dat_flattened = dat

    n_eff = samples - p

    # 2. Build Target Matrix Y: values at time t, shape (n_eff, chans)
    Y = dat_flattened[:, p:].T  

    # 3. Build Lagged Feature Matrix X: shape (n_eff, chans * p)
    # Ordered by lag, then channel to match standard MVAR conventions
    X = np.zeros((n_eff, chans * p))
    for lag in range(1, p + 1):
        X[:, (lag - 1) * chans : lag * chans] = dat_flattened[:, p - lag : samples - lag].T

    # 4. Fit MultiTaskLasso
    # fit_intercept=False is standard for Yule-Walker/MVAR assuming mean-centered EEG
    lasso = MultiTaskLasso(alpha=alpha, fit_intercept=False, max_iter=3000, tol=1e-4)
    lasso.fit(X, Y)

    # 5. Reshape coefficients to match your pipeline format:
    # lasso.coef_ has shape (n_targets, n_features) -> (chans, chans * p)
    # Reshape to (chans, p, chans) [target_ch, lag, source_ch] then transpose to (chans, chans, p)
    coef_reshaped = lasso.coef_.reshape(chans, p, chans)
    AR = coef_reshaped.transpose(0, 2, 1)

    # 6. Compute Residual Variance Matrix Vr
    Y_pred = lasso.predict(X)
    residuals = Y - Y_pred
    Vr = np.dot(residuals.T, residuals) / n_eff

    return AR, Vr

###
def mvar_H(Ar, f = np.arange(1, 40, 1), Fs = 128):
    """
    Calculate the transfer function H from multivariate autoregressive coefficients Ar.
    
    Parameters:
    Ar : numpy.ndarray
        AR coefficient matrix with shape (chan, chan, p), where p is model order. Here set to 9.
    f : numpy.ndarray
        Frequency vector.
    Fs : float
        Sampling frequency.

    Returns:
    H : numpy.ndarray
        Transfer function matrix with shape (chan, chan, len(f)).
    """

    Nf = len(f)
    chan, _, p = Ar.shape
    
    H = np.zeros((chan, chan, Nf), dtype=complex)
    A_out = np.zeros((chan, chan, Nf), dtype=complex)

    #lag vector
    m_vector = np.arange(1, p + 1).reshape(p, 1)

    # Z transform (p,1) -> (p, Nf) (into frequency domain)
    z = np.exp(-1j * 2 * np.pi * m_vector * f / Fs) 

    # Ar shape: (chan, chan, p), z shape: (p, Nf) -> sum_term shape: (chan, chan, Nf)
    # 'ijp' = target, source, lag | 'pf' = lag, frequency -> 'ijf' = target, source, frequency
    sum_term = np.einsum('ijp,pf->ijf', Ar, z) #vectorized summation
    I = np.eye(chan, dtype=complex)[..., np.newaxis]
    A_out = I - sum_term

    H = np.zeros_like(A_out, dtype=complex)
    for fi in range(Nf):
        try:
            H[:, :, fi] = np.linalg.inv(A_out[:, :, fi])
        except np.linalg.LinAlgError:
            H[:, :, fi] = np.nan
            
    return H

###
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
    N_chan, _, N_f = DTF.shape
    # Normalize DTF to get full-frequency DTF (ffDTF)
    ffDTF = np.zeros((N_chan,N_chan, N_f))
    for i in range(N_chan):  # rows
        for j in range(N_chan):  # columns
            ffDTF[i,j,:] = DTF[i,j,:] / np.sum(DTF[i,:,:])
    return ffDTF

###
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

    fig.suptitle(f"{conn_name} for an exemplary EEG window", fontsize=18, y=1.002)
    fig.supxlabel("Frequency (Hz)", fontsize = 14)
    fig.supylabel(ylabel, fontsize = 14)

    plt.tight_layout()
    plt.show()

###
def heatmap_bands(mats, cmap='viridis'):
    titles = list(mats)
    vmin = min(m.min() for m in mats.values())
    vmax = max(m.max() for m in mats.values())

    fig, axes = plt.subplots(1, len(mats), figsize=(4.4 * len(mats), 5), squeeze=False)
    axes = axes.ravel()
    for ax, title in zip(axes, titles):
        sns.heatmap(mats[title], ax=ax, xticklabels=c.channels, yticklabels=c.channels,
                    cmap=cmap, vmin=vmin, vmax=vmax, square=True, cbar=False)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=90, labelsize=8)
        ax.tick_params(axis='y', rotation=0, labelsize=8)

    sm = plt.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax), cmap=cmap) #type:ignore
    fig.colorbar(sm, ax=axes, shrink=0.65, pad=0.02, label='Connectivity strength')
    return fig


def circle_bands(mats,cmap='viridis'):
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


def graph_bands(mats):
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

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6), squeeze=False)
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
    return graphs


def graph_bands_sparse(mats, top_percent=10, threshold=None, shared_scale=True, mode = 'global'):
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

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6), squeeze=False)
    axes = axes.ravel()

    graphs = {}
    for ax, (title, adj) in zip(axes, sparse.items()):
        G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
        nx.draw_networkx(
            G, pos, ax=ax, with_labels=True, labels=labels,
            arrows=True, width=get_linewidths(G, vmax), node_size=500,
            arrowstyle='->', arrowsize=35, connectionstyle='arc3,rad=0.2',
        )
        ax.set_title(f"{title}  ({G.number_of_edges()} edges)",
                     fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')

        graphs[title] = G

    fig.tight_layout()
    return graphs