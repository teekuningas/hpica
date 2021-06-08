##############
# In this example we will try the hierarchical ICA to a toy example.
# For comparison, we use temporal ICA with spatial concatenation as well.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.decomposition import FastICA

from hpica import HPICA


random_state = np.random.RandomState(5)
colors = ['red', 'steelblue', 'green', 'pink']

##################
# Generate temporally independent sources and mixings
n_samples = 200
time = np.linspace(0, 30, n_samples)

# Generate sine waves (they have two-peaked nongaussian distributions)
# and remove temporal correlations
s1 = np.sin(2 * time)
s2 = np.sin(3 * time)
s3 = np.sin(5 * time)
random_state.shuffle(s1)
random_state.shuffle(s2)
random_state.shuffle(s3)

S = np.c_[s1, s2, s3]
S = (S - S.mean(axis=0)) / S.std(axis=0)

# Plot scatters for each source pair to see
# that sources are nongaussian and  more or less independent
sns.pairplot(pd.DataFrame(S))

n_sources = S.shape[1]

n_subjects = 10
groups = 5*[0] + 5*[1]
subjects = []
for idx in range(n_subjects):

    template = np.array([[1.0, 1.0, 1.0],
                         [1.0, 2.0, 1.0],
                         [1.0, 1.0, 1.0]])

    # Vary mixing of s2 and s3 components by subject
    padded_template_s1 = np.pad(template, pad_width=[(1,1), (0, n_subjects+10)])
    padded_template_s2 = np.pad(template, pad_width=[(1,1), (idx+4, n_subjects+6-idx)])
    padded_template_s3 = np.pad(template, pad_width=[(1,1),(idx+7, n_subjects+3-idx)])

    spatial_shape = padded_template_s1.shape

    A = np.array([padded_template_s1.flatten(), 
                  padded_template_s2.flatten(),
                  padded_template_s3.flatten()]).T

    subject_s1 = s1.copy() 
    subject_s1[0:int(len(subject_s1)/2)] += 0.5*groups[idx] - 0.5*(1 - groups[idx])
    subject_s1[int(len(subject_s1)/2):] += -0.5*groups[idx] + 0.5*(1 - groups[idx])

    subject_s2 = s2.copy()
    subject_s3 = s3.copy()

    subject_S = np.c_[subject_s1, subject_s2, subject_s3]
    # subject_S = (subject_S - subject_S.mean(axis=0)) / subject_S.std(axis=0)

    # Generate observations from sources and mixing
    X = np.dot(subject_S, A.T)
    X += 0.2 * random_state.normal(size=X.shape)
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    subjects.append((X, A, subject_S))

##############################################
# Plot spatial heatmaps of the mixing matrices
fig, axes = plt.subplots(n_subjects, n_sources, constrained_layout=True)
fig.suptitle('Mixing maps for the sources')
for subject_idx, subject in enumerate(subjects):
    for source_idx in range(n_sources):
        ax = axes[subject_idx, source_idx]
        heatmap = subject[1][:, source_idx].reshape(spatial_shape)

        ax.imshow(heatmap, cmap='hot', interpolation='nearest')

        title = ('Subject ' + str(subject_idx+1) +
                 ', source ' + str(source_idx+1))
        ax.set_title(title, fontsize=7)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(7)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(7)


##################################
# Plot time courses of the sources
fig, axes = plt.subplots(n_subjects, n_sources, constrained_layout=True)
fig.suptitle('Time courses for the sources')
for subject_idx, subject in enumerate(subjects):
    for source_idx in range(n_sources):
        ax = axes[subject_idx, source_idx]

        ax.plot(time, subject[2][:, source_idx])

        ax.set_ylim(-1.5, 1.5)
        title = ('Subject ' + str(subject_idx+1) +
                 ', source ' + str(source_idx+1))
        ax.set_title(title, fontsize=7)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(7)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(7)

############################
# Helpers for plotting and computing scores

def corrcoef_with_shift(signal1, signal2):
    corrcoefs = []
    for idx in range(len(signal1)):
        shifted_signal1 = np.roll(signal1, shift=idx)
        corrcoefs.append(np.abs(np.corrcoef(shifted_signal1, signal2)[0,1]))
    return np.max(corrcoefs)


def compute_results(sources):
    results = []
    for subject_idx in range(n_subjects):
        subject_results = []
        for unmixed_ts in sources[subject_idx].T:
            # coefs = [corrcoef_with_shift(orig_ts, unmixed_ts) for orig_ts in S.T]
            coefs = [corrcoef_with_shift(orig_ts, unmixed_ts) for orig_ts in subjects[subject_idx][2].T]
            subject_results.append([np.argmax(coefs), np.max(coefs)])
        results.append(subject_results)
    return results

def plot_scores(results, title):

    ordering = np.argsort([[elem[0] for elem in res] for res in results])[0]
    coefs = np.array([[elem[1] for elem in res] for res in results])[:, ordering]

    labels = ['Source ' + str(idx+1) for idx in range(n_sources)]
    x = np.arange(len(labels))

    width = 0.5 / n_subjects
    min_ = 0.5

    fig, ax = plt.subplots()
    for subject_idx in range(n_subjects):
        locs = x - (n_subjects / 2)*width + subject_idx*width
        values = np.array(coefs[subject_idx]) - min_
        rects = ax.bar(locs, values, width, 
                       label=('Subject ' + str(subject_idx+1)), bottom=min_) 

    ax.set_ylim((min_, 1.0))
    ax.set_ylabel('Scores')
    ax.set_title(title + ' (' + str(np.round(np.mean(coefs), 3)) + ')')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

############################################
# Unmix the data with spatial concatenation and temporal ICA
concatenated = np.concatenate([data[0] for data in subjects], axis=1)

ica = FastICA(n_components=n_sources, random_state=random_state)
print("Fitting SC-TICA")
ica.fit(concatenated)

subjects_ts = []
subjects_maps = []
for subject_idx in range(n_subjects):
    start = subject_idx*int(concatenated.shape[1]/n_subjects)
    end = (subject_idx+1)*int(concatenated.shape[1]/n_subjects)

    mixing = ica.components_[:, start:end]
    sources = subjects[subject_idx][0] @ mixing.T

    subjects_ts.append(sources)
    subjects_maps.append(mixing.T)

results = compute_results(subjects_ts)
ordering = np.argsort([comp[0] for comp in results[0]])

subjects_ts = [sub[:, ordering] for sub in subjects_ts]
subjects_maps = [sub[:, ordering] for sub in subjects_maps]

fig, axes = plt.subplots(n_subjects, n_sources, constrained_layout=True)
fig.suptitle('SC-TICA')
for subject_idx in range(n_subjects):
    for source_idx in range(n_sources):
        ax = axes[subject_idx, source_idx]

        heatmap = subjects_maps[subject_idx][:, source_idx].reshape(spatial_shape)
        ax.imshow(heatmap, cmap='hot', interpolation='nearest')

        title = 'Subject ' + str(subject_idx+1) + ', source ' + str(source_idx+1)
        ax.set_title(title, fontsize=7)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(7)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(7)

fig, axes = plt.subplots(n_subjects, constrained_layout=True)
fig.suptitle('SC-TICA')

lim_min = np.min([subjects_ts[idx][:, 0] for idx in range(n_subjects)])
lim_max = np.max([subjects_ts[idx][:, 0] for idx in range(n_subjects)])
for subject_idx, source in enumerate(subjects_ts):
    ax = axes[subject_idx]

    # Plot first component
    ax.plot(source[:, 0], color=colors[0], lw=0.5)

    ax.set_ylim(lim_min, lim_max)
    title = 'Subject ' + str(subject_idx+1) + ', first source'
    ax.set_title(title, fontsize=7)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(7)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(7)

plot_scores(results, 'SC-TICA')

############################################
# Unmix the data with hierarchical probabilistic temporal ICA.
# Furthermore, do inference for group difference.
Y = [data[0].T for data in subjects]
X = np.array(groups)[:, np.newaxis]

print("Fitting hierarchical TICA")

# Fit the hierarchical ICA.
# Use exact algorithm as the (faster) subspace algorithm cannot recover three coactivated sine waves.
# In fMRI applications, as in the paper, the subspace algorithm probably works well.
# 
# Note that the EM algorithm is quite sensitive to initialization. To use only 2 gaussians, 
# you should help the algorithm with something like this: 
# init_values = {
#     'mus': np.tile([-1, 1], (n_sources, 1)),
#     'pis': np.tile([0.5, 0.5], (n_sources, 1)),
# }
# To more robustly model the sine waves, let's set n_gaussians to 3. 
# The default initialization works fine here.
init_values = {}
ica = HPICA(n_components=n_sources,
            random_state=random_state,
            n_iter=20,
            n_gaussians=3,
            init_values=init_values,
            algorithm='exact')

ica.fit(Y, X)

ica.plot_evolution()

ica_mixing = ica.mixing
pca_means = ica.wh_mean
pca_whitening = ica.wh_matrix

subjects_ts = []
subjects_maps = []
for subject_idx in range(n_subjects):
    demeaned = Y[subject_idx] - pca_means[subject_idx][:, np.newaxis]
    mixing = pca_whitening[subject_idx].T @ ica_mixing[subject_idx]
    unmixing = mixing.T
    sources = (unmixing @ demeaned) 
    
    subjects_ts.append(sources.T)
    subjects_maps.append(mixing)

results = compute_results(subjects_ts)
ordering = np.argsort([comp[0] for comp in results[0]])

subjects_ts = [sub[:, ordering] for sub in subjects_ts]
subjects_maps = [sub[:, ordering] for sub in subjects_maps]

fig, axes = plt.subplots(n_subjects, n_sources, constrained_layout=True)
fig.suptitle('Hierarchical TICA')
for subject_idx in range(n_subjects):
    for source_idx in range(n_sources):
        ax = axes[subject_idx, source_idx]
        heatmap = subjects_maps[subject_idx][:, source_idx].reshape(spatial_shape)

        ax.imshow(heatmap, cmap='hot', interpolation='nearest')

        title = 'Subject ' + str(subject_idx+1) + ', source ' + str(source_idx+1)
        ax.set_title(title, fontsize=7)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(7)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(7)

fig, axes = plt.subplots(n_subjects, constrained_layout=True)
fig.suptitle('Hierarchical TICA')

lim_min = np.min([subjects_ts[idx][:, 0] for idx in range(n_subjects)])
lim_max = np.max([subjects_ts[idx][:, 0] for idx in range(n_subjects)])
for subject_idx, source in enumerate(subjects_ts):
    ax = axes[subject_idx]

    # Plot first component
    ax.plot(source[:, 0], color=colors[0], lw=0.5)

    ax.set_ylim(lim_min, lim_max)
    title = 'Subject ' + str(subject_idx+1) + ', first source'
    ax.set_title(title, fontsize=7)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(7)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(7)

plot_scores(results, 'Hierarchical TICA')
