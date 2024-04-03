from matplotlib import pyplot as plt


def generated_spread_histogram(predicted_spread, true_spread):
    diff_spread = [y - x for x, y in zip(predicted_spread, true_spread)]

    fig, ax = plt.subplots(1, 1)
    plt.hist(diff_spread)

    ax.set_xlabel("spread diff (true - predicited)")
    ax.set_ylabel("freqency")
    ax.set_title("Baseline model performance")

    min_ylim, max_ylim = plt.ylim()
    min_xlim, max_xlim = plt.xlim()

    # plot the median
    median = sorted(diff_spread)[len(diff_spread) // 2]
    plt.axvline(
        median,
        color="black",
        linestyle="dashed",
        linewidth=1,
        marker="d",
    )
    plt.text(median * 1.1, max_ylim * 0.9, "Median: {:.2f}".format(median))

    plt.show()
