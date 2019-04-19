

import matplotlib.pyplot as plt

model = 'randomforest'
plt.figure(figsize=(13, 13))
plt.title(f"{model}", fontsize=16)

plt.xlabel("min_samples_split")
plt.ylabel("Score")

import maracatu as m


def plot_hist(mb, xlabel, ylabel, filepath, filename):
    par_base = get_par_base()
    par_base.update({
        'file_output': f'{filepath}/{filename}',
        'xlabel': f'{xlabel}',
        'ylabel': f'{ylabel}'
    })

    fig, ax = plt.subplots(1, 1)

    ax.hist(mb)

    plt_base = m.PlotBase()
    plt_base.configure_ax(ax, par_base)
    plt_base.configure_fig(fig, par_base)

    plt.show()

def get_values(results, scorer, sample, kfold):
    values = [results[f'split{i}_{sample}_{scorer}'] for i in range(0, kfold)]
    return np.ravel(values)




kfold = 10
sample = 'test' #train, test
scorer = 'f1_score'

values = get_values(results, scorer, sample, kfold)

plot_hist(values, 'F1_score (X)', 'P(X)', path_plots, f'{model}_{sample}_{scorer}.pdf')

X_axis = np.array(results['param_classifier__min_samples_split'].data, dtype=float)


fig, ax = plt.subplots(1, 1)

color = 'g'
for sample, style in (('train', '--'), ('test', '-')):
    sample_score_mean = path_results['mean_%s_%s' % (sample, scorer)]
    sample_score_std = path_results['std_%s_%s' % (sample, scorer)]
    ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                    sample_score_mean + sample_score_std,
                    alpha=0.1 if sample == 'test' else 0, color=color)
    ax.plot(X_axis, sample_score_mean, style, color=color,
            alpha=1 if sample == 'test' else 0.7,
            label="%s (%s)" % (scorer, sample))

    best_index = np.nonzero(path_results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = path_results['mean_test_%s' % scorer][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid('off')
plt.savefig(f'{path_plots}/{model}.pdf')