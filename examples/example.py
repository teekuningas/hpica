import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

from sklearn.decomposition import FastICA


random_state = 4
colors = ['red', 'steelblue', 'green', 'pink']

##################
# Generate sources and mixings
n_samples = 1000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)
s2 = np.sign(np.sin(3 * time))
s3 = scipy.signal.sawtooth(2 * np.pi * time)
s4 = np.sin(5 * time)

S = np.c_[s1, s2, s3, s4]
S = (S - S.mean(axis=0)) / S.std(axis=0)

n_sources = S.shape[1]

n_subjects = 3
subjects = []
for idx in range(n_subjects):

    template = np.array([[1.0, 1.0, 1.0],
                         [1.0, 2.0, 1.0],
                         [1.0, 1.0, 1.0]])

    # Vary location of s2 and s3 components by subject
    padded_template_s1 = np.pad(template, pad_width=[(1,1), (0, n_subjects+10)])
    padded_template_s2 = np.pad(template, pad_width=[(1,1), (idx+4, n_subjects+6-idx)])
    padded_template_s3 = np.pad(template, pad_width=[(1,1),(idx+7, n_subjects+3-idx)])
    padded_template_s4 = np.pad(template, pad_width=[(1,1), (n_subjects+10,0)])

    spatial_shape = padded_template_s1.shape

    A = np.array([padded_template_s1.flatten(), 
                  padded_template_s2.flatten(),
                  padded_template_s3.flatten(),
                  padded_template_s4.flatten()]).T

    # Add subject-specific delay two components 3 and 4
    subject_s1 = s1.copy()
    subject_s2 = s2.copy()
    subject_s3 = np.roll(s3, int(idx*n_samples/10))
    subject_s4 = np.roll(s4, int(idx*n_samples/10))

    subject_S = np.c_[subject_s1, subject_s2, subject_s3, subject_s4]
    subject_S = (subject_S - subject_S.mean(axis=0)) / subject_S.std(axis=0)

    # Generate observations from sources and mixing
    X = np.dot(subject_S, A.T)
    X += 1.0 * np.random.RandomState(random_state).normal(size=X.shape)
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
# Unmix the data with temporal concatenation and temporal ica
concatenated = np.concatenate([data[0] for data in subjects])

ica = FastICA(n_components=n_sources, random_state=random_state)

transformed = ica.fit_transform(concatenated)

subjects_sources = np.vsplit(transformed, n_subjects)
mixing = ica.components_.T

results = compute_results(subjects_sources)
ordering = np.argsort([comp[0] for comp in results[0]])

subjects_sources = [sub[:, ordering] for sub in subjects_sources]
mixing = mixing[:, ordering]

fig, axes = plt.subplots(1, n_sources)
fig.suptitle('TC-TICA')
for source_idx in range(n_sources):
    ax = axes[source_idx]
    heatmap = mixing[:, source_idx].reshape(spatial_shape)
    title = 'Source ' + str(source_idx+1)
    ax.set_title(title)
    ax.imshow(heatmap, cmap='hot', interpolation='nearest')

fig.tight_layout()
plt.show(block=False)

fig, axes = plt.subplots(n_subjects)
fig.suptitle('TC-TICA')
for subject_idx, source in enumerate(subjects_sources):
    ax = axes[subject_idx]
    ax.set_title('Subject ' + str(subject_idx+1))
    for ts_idx, ts in enumerate(source.T):
        ax.plot(ts, color=colors[ts_idx], lw=0.5)

fig.tight_layout()
plt.show(block=False)

title = 'TC-TICA'
plot_scores(results, title)


############################################
# Unmix the data with spatial concatenation and temporal ICA
concatenated = np.concatenate([data[0] for data in subjects], axis=1)

ica = FastICA(n_components=n_sources, random_state=random_state)
ica.fit(concatenated)

subjects_sources = []
subjects_mixing = []
for subject_idx in range(n_subjects):
    start = subject_idx*int(concatenated.shape[1]/n_subjects)
    end = (subject_idx+1)*int(concatenated.shape[1]/n_subjects)
    mixing = ica.components_[:, start:end]
    ts = subjects[subject_idx][0] @ mixing.T

    subjects_sources.append(ts)
    subjects_mixing.append(mixing.T)

results = compute_results(subjects_sources)
ordering = np.argsort([comp[0] for comp in results[0]])

subjects_sources = [sub[:, ordering] for sub in subjects_sources]
subjects_mixing = [sub[:, ordering] for sub in subjects_mixing]

fig, axes = plt.subplots(n_subjects, n_sources)
fig.suptitle('SC-TICA')
for subject_idx in range(n_subjects):
    for source_idx in range(n_sources):
        ax = axes[subject_idx, source_idx]
        heatmap = subjects_mixing[subject_idx][:, source_idx].reshape(spatial_shape)
        title = 'Subject ' + str(subject_idx+1) + ', source ' + str(source_idx+1)
        ax.set_title(title)
        ax.imshow(heatmap, cmap='hot', interpolation='nearest')

fig.tight_layout()
plt.show(block=False)

fig, axes = plt.subplots(n_subjects)
fig.suptitle('SC-TICA')
for subject_idx, source in enumerate(subjects_sources):
    ax = axes[subject_idx]
    ax.set_title('Subject ' + str(subject_idx+1))
    for ts_idx, ts in enumerate(source.T):
        ax.plot(ts, color=colors[ts_idx], lw=0.5)

fig.tight_layout()
plt.show(block=False)

title = "SC-TICA"
plot_scores(results, title)

############################################
# Unmix the data with temporal concatenation and spatial ICA
concatenated = np.concatenate([data[0] for data in subjects], axis=0)

ica = FastICA(n_components=n_sources, random_state=random_state)
ica.fit(concatenated.T)

subjects_sources = []
subjects_mixing = []
for subject_idx in range(n_subjects):
    start = subject_idx*int(concatenated.shape[0]/n_subjects)
    end = (subject_idx+1)*int(concatenated.shape[0]/n_subjects)
    ts = ica.components_[:, start:end].T
    mixing = ts.T @ subjects[subject_idx][0]

    subjects_sources.append(ts)
    subjects_mixing.append(mixing.T)

results = compute_results(subjects_sources)
ordering = np.argsort([comp[0] for comp in results[0]])

subjects_sources = [sub[:, ordering] for sub in subjects_sources]
subjects_mixing = [sub[:, ordering] for sub in subjects_mixing]

fig, axes = plt.subplots(n_subjects, n_sources)
fig.suptitle('TC-SICA')
for subject_idx in range(n_subjects):
    for source_idx in range(n_sources):
        ax = axes[subject_idx, source_idx]
        heatmap = subjects_mixing[subject_idx][:, source_idx].reshape(spatial_shape)
        title = 'Subject ' + str(subject_idx+1) + ', source ' + str(source_idx+1)
        ax.set_title(title)
        ax.imshow(heatmap, cmap='hot', interpolation='nearest')

fig.tight_layout()
plt.show(block=False)

fig, axes = plt.subplots(n_subjects)
fig.suptitle('TC-SICA')
for subject_idx, source in enumerate(subjects_sources):
    ax = axes[subject_idx]
    ax.set_title('Subject ' + str(subject_idx+1))
    for ts_idx, ts in enumerate(source.T):
        ax.plot(ts, color=colors[ts_idx], lw=0.5)

fig.tight_layout()
plt.show(block=False)

title = "TC-SICA"
plot_scores(results, title)


