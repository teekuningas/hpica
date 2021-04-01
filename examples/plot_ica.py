##############
# In this example we will try the hierarchical ICA to a toy example.
# For comparison, we use temporal ICA with spatial concatenation as well.
# Both seem to work fine here.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.decomposition import FastICA

from hpica import compute_hpica


random_state = np.random.RandomState(4)
colors = ['red', 'steelblue', 'green', 'pink']

##################
# Generate temporally independent sources and mixings
n_samples = 500
time = np.linspace(0, 30, n_samples)

# Generate sine waves (they have two-peaked nongaussian distributions)
# and remove temporal correlations
s1 = np.sin(2 * time)
s2 = np.sin(3 * time)
s3 = np.sin(5 * time)
s4 = np.sin(7 * time)
random_state.shuffle(s1)
random_state.shuffle(s2)
random_state.shuffle(s3)
random_state.shuffle(s4)

S = np.c_[s1, s2, s3, s4]
S = (S - S.mean(axis=0)) / S.std(axis=0)

# Plot scatters for each source pair to see
# that sources are more or less independent
sns.pairplot(pd.DataFrame(S))
plt.show()

n_sources = S.shape[1]

n_subjects = 3
subjects = []
for idx in range(n_subjects):

    template = np.array([[1.0, 1.0, 1.0],
                         [1.0, 2.0, 1.0],
                         [1.0, 1.0, 1.0]])

    # Vary mixing of s2 and s3 components by subject
    padded_template_s1 = np.pad(template, pad_width=[(1,1), (0, n_subjects+10)])
    padded_template_s2 = np.pad(template, pad_width=[(1,1), (idx+4, n_subjects+6-idx)])
    padded_template_s3 = np.pad(template, pad_width=[(1,1),(idx+7, n_subjects+3-idx)])
    padded_template_s4 = np.pad(template, pad_width=[(1,1), (n_subjects+10,0)])

    spatial_shape = padded_template_s1.shape

    A = np.array([padded_template_s1.flatten(), 
                  padded_template_s2.flatten(),
                  padded_template_s3.flatten(),
                  padded_template_s4.flatten()]).T

    subject_s1 = s1.copy()
    subject_s2 = s2.copy()
    subject_s3 = s3.copy()
    subject_s4 = s4.copy()

    subject_S = np.c_[subject_s1, subject_s2, subject_s3, subject_s4]
    subject_S = (subject_S - subject_S.mean(axis=0)) / subject_S.std(axis=0)

    # Generate observations from sources and mixing
    X = np.dot(subject_S, A.T)
    X += 0.2 * random_state.normal(size=X.shape)
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    subjects.append((X, A))

##############################################
# Plot spatial heatmaps of the mixing matrices
fig, axes = plt.subplots(n_subjects, n_sources)
fig.suptitle('Mixing maps for the sources')
for subject_idx, subject in enumerate(subjects):
    for source_idx in range(n_sources):
        ax = axes[subject_idx, source_idx]
        heatmap = subject[1][:, source_idx].reshape(spatial_shape)
        title = ('Subject ' + str(subject_idx+1) +
                 ', source ' + str(source_idx+1))
        ax.set_title(title)
        ax.imshow(heatmap, cmap='hot', interpolation='nearest')

fig.tight_layout()
plt.show(block=False)


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
            coefs = [corrcoef_with_shift(orig_ts, unmixed_ts) for orig_ts in S.T]
            subject_results.append([np.argmax(coefs), np.max(coefs)])
        results.append(subject_results)
    return results

def plot_scores(results, title):

    ordering = np.argsort([[elem[0] for elem in res] for res in results])[0]
    coefs = np.array([[elem[1] for elem in res] for res in results])[:, ordering]

    labels = ['Source ' + str(idx+1) for idx in range(n_sources)]
    x = np.arange(len(labels))

    width = 0.1
    min_ = 0.5

    fig, ax = plt.subplots()
    for subject_idx in range(n_subjects):
        values = np.array(coefs[subject_idx]) - min_
        rects = ax.bar(x - width + width*subject_idx, values, width, 
                       label=('Subject ' + str(subject_idx+1)), bottom=min_) 

    ax.set_ylim((min_, 1.0))
    ax.set_ylabel('Scores')
    ax.set_title(title + ' (' + str(np.round(np.mean(coefs), 3)) + ')')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.show()

############################################
# Unmix the data with spatial concatenation and temporal ICA
concatenated = np.concatenate([data[0] for data in subjects], axis=1)

ica = FastICA(n_components=n_sources, random_state=random_state)
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

fig, axes = plt.subplots(n_subjects, n_sources)
fig.suptitle('SC-TICA')
for subject_idx in range(n_subjects):
    for source_idx in range(n_sources):
        ax = axes[subject_idx, source_idx]
        heatmap = subjects_maps[subject_idx][:, source_idx].reshape(spatial_shape)
        title = 'Subject ' + str(subject_idx+1) + ', source ' + str(source_idx+1)
        ax.set_title(title)
        ax.imshow(heatmap, cmap='hot', interpolation='nearest')

fig.tight_layout()
plt.show(block=False)

fig, axes = plt.subplots(n_subjects)
fig.suptitle('SC-TICA')
for subject_idx, source in enumerate(subjects_ts):
    ax = axes[subject_idx]
    ax.set_title('Subject ' + str(subject_idx+1))
    for ts_idx, ts in enumerate(source.T):
        ax.plot(ts, color=colors[ts_idx], lw=0.5)

fig.tight_layout()
plt.show(block=False)

title = "SC-TICA"
plot_scores(results, title)


############################################
# Unmix the data with hierarchical probabilistic temporal ICA
Y = [data[0].T for data in subjects]

results = compute_hpica(Y, n_components=n_sources, random_state=random_state, 
                        n_iter=20, n_gaussians=3, initial_guess='ica', algorithm='subspace')

ica_mixing = results[0]
pca_means = results[6]
pca_whitening = results[7]

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

fig, axes = plt.subplots(n_subjects, n_sources)
fig.suptitle('Hierarchical TICA')
for subject_idx in range(n_subjects):
    for source_idx in range(n_sources):
        ax = axes[subject_idx, source_idx]
        heatmap = subjects_maps[subject_idx][:, source_idx].reshape(spatial_shape)
        title = 'Subject ' + str(subject_idx+1) + ', source ' + str(source_idx+1)
        ax.set_title(title)
        ax.imshow(heatmap, cmap='hot', interpolation='nearest')

fig.tight_layout()
plt.show(block=False)

fig, axes = plt.subplots(n_subjects)
fig.suptitle('Hierarchical TICA')
for subject_idx, source in enumerate(subjects_ts):
    ax = axes[subject_idx]
    ax.set_title('Subject ' + str(subject_idx+1))
    for ts_idx, ts in enumerate(source.T):
        ax.plot(ts, color=colors[ts_idx], lw=0.5)

fig.tight_layout()
plt.show(block=False)

title = "Hierarchical TICA"
plot_scores(results, title)

