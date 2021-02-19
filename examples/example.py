import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

from sklearn.decomposition import FastICA


random_state = 20
colors = ['red', 'steelblue', 'green']

##################
# Generate sources
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)
s2 = np.sign(np.sin(3 * time))
s3 = scipy.signal.sawtooth(2 * np.pi * time)

S = np.c_[s1, s2, s3]
S += 0.5 * np.random.RandomState(random_state).normal(size=S.shape)
S = (S - S.mean(axis=0)) / S.std(axis=0)

n_sources = S.shape[1]

###################################
# Generate subject-specific mixings
subjects = []
n_subjects = 3
for idx in range(n_subjects):

    template = np.array([[1.0, 1.0, 1.0],
                         [1.0, 2.0, 1.0],
                         [1.0, 1.0, 1.0]])

    # padded_template_s1 = np.pad(template, pad_width=[(1,1),(2, ((n_subjects-1)))])

    # padded_template_s1 = np.pad(template, pad_width=[(1,1),(idx+1, ((n_subjects-idx)))])
    # padded_template_s2 = np.pad(template, pad_width=[(1,1), (0, n_subjects+1)])
    # padded_template_s3 = np.pad(template, pad_width=[(1,1), (n_subjects+1,0)])

    padded_template_s1 = np.pad(template, pad_width=[(1,1),(idx+3, ((n_subjects+2-idx)))])
    padded_template_s2 = np.pad(template, pad_width=[(1,1), (0, n_subjects+5)])
    padded_template_s3 = np.pad(template, pad_width=[(1,1), (n_subjects+5,0)])

    spatial_shape = padded_template_s1.shape

    A = np.array([padded_template_s1.flatten(), 
                  padded_template_s2.flatten(),
                  padded_template_s3.flatten()]).T

    X = np.dot(S, A.T)
    X += 1.0 * np.random.RandomState(random_state).normal(size=X.shape)
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    subjects.append((X, A))

##############################################
# Plot spatial heatmaps of the mixing matrices
fig, axes = plt.subplots(n_subjects, n_sources)
fig.suptitle('Mixing heatmaps')
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
# Helper for plotting scores
def plot_scores(sources, title):

    results = []
    for subject_idx in range(n_subjects):
        subject_results = []
        for unmixed_ts in sources[subject_idx].T:
            coefs = [np.corrcoef(unmixed_ts, orig_ts)[0, 1] for orig_ts in S.T]
            subject_results.append(np.max(np.abs(coefs)))
        results.append(subject_results)

    labels = ['Source ' + str(idx+1) for idx in range(n_sources)]
    x = np.arange(len(labels))
    width = 0.1
    min_ = 0.8

    fig, ax = plt.subplots()
    for subject_idx in range(n_subjects):
        values = np.array(results[subject_idx]) - min_
        rects = ax.bar(x - width + width*subject_idx, values, width, 
                       label=('Subject ' + str(subject_idx+1)), bottom=min_) 

    ax.set_ylim((min_, 1.0))

    ax.set_ylabel('Scores')
    ax.set_title(title + ' (' + str(np.round(np.mean(results), 3)) + ')')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()


############################################
# Unmix the data with temporal concatenation and temporal ica
concatenated = np.concatenate([data[0] for data in subjects])

ica = FastICA(n_components=3)

sources = ica.fit_transform(concatenated)
subjects_sources = np.vsplit(sources, n_subjects)

mixing = ica.components_.T

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
plot_scores(subjects_sources, title)


############################################
# Unmix the data with spatial concatenation and temporal ICA
concatenated = np.concatenate([data[0] for data in subjects], axis=1)

ica = FastICA(n_components=3)
ica.fit(concatenated)

subjects_sources = []
subjects_mixing = []
for subject_idx in range(n_subjects):
    start = subject_idx*int(concatenated.shape[1]/3)
    end = (subject_idx+1)*int(concatenated.shape[1]/3)
    mixing = ica.components_[:, start:end]
    source = subjects[subject_idx][0] @ mixing.T
    subjects_sources.append(source)
    subjects_mixing.append(mixing.T)

fig, axes = plt.subplots(n_subjects, n_sources)
fig.suptitle('SC-TICA')
for subject_idx in range(n_subjects):
    for source_idx in range(n_sources):
        ax = axes[subject_idx, source_idx]
        heatmap = subjects_mixing[subject_idx][:, source_idx].reshape(spatial_shape)
        title = 'Source ' + str(source_idx+1)
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
plot_scores(subjects_sources, title)

