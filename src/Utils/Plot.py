import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, classes, k):
    import os
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    plt.clf()

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=10)
    plt.colorbar()

    # classes = np.unique(np.concatenate((class_list, labels)))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=15, horizontalalignment="center")
    plt.yticks(tick_marks, classes, fontsize=15, rotation=90, verticalalignment='center')

    plt.xlabel('Predicted Label', fontsize=10)
    plt.ylabel('True Label', fontsize=10)

    # 행렬의 값 표시
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     verticalalignment='center',
                     color="white" if cm[i, j] > cm.mean() else "black",
                     fontsize=25)

    plt.savefig(f'./cfm/CFM_{k}.png')


def plot_PRCurve(labels, results, k):
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    from itertools import cycle
    plt.clf()

    y_scores = np.zeros((len(labels), 10))
    y_scores[np.arange(len(labels)), results] = 1

    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(2):
        precision[i], recall[i], _ = precision_recall_curve((labels == i).astype(int), y_scores[:, i])
        average_precision[i] = average_precision_score((labels == i).astype(int), y_scores[:, i])

    colors = cycle(['navy', 'darkorange'])
    for i, color in zip(range(2), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2, label='Class {0} (AP = {1:0.2f})'.format(i, average_precision[i]))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve for Binary Classification')
    plt.legend(loc="upper right")
    plt.savefig(f'./PRC/PRC_{k}.png')


def plot_Hop_acc(acc, name, class_list):
    import matplotlib.pyplot as plt
    plt.clf()

    # labels = [1,2,3,4]
    labels = [1, 2, 3, 4, 5]
    plt.plot(labels, acc, marker='o', linestyle='-', color='blue')
    plt.xlabel('Number of Event-VoxelHop units', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.title(f'NMNIST ({class_list[0]} vs {class_list[1]})', fontsize=15)

    plt.ylim(0.5, 1.0)
    # 정확도 값 표시
    for i, accuracy in enumerate(acc):
        plt.text(i+1, accuracy + 0.01, f'{accuracy:.4f}', ha='center', va='bottom')

    plt.savefig(f'./Hop_acc/{name}_acc.png')

def plot_Energy_Log(i):
    import matplotlib.pyplot as plt
    import pickle
    from matplotlib.ticker import FuncFormatter
    plt.clf()

    fr = open(f'./weight/{i+1}-1_DVS_Hop.pkl' , 'rb')
    pca_params = pickle.load(fr)
    fr.close()

    percent_99 = np.argmax(np.cumsum(pca_params['energy'])>0.99)+1
    x_values = list(range(0, len(pca_params['energy'][:percent_99])))
    y_values = np.log10(pca_params['energy'][:percent_99])

    plt.plot(x_values, y_values, linestyle='-', color='black')

    plt.xlabel('Number of AC Components', fontsize=15)
    plt.ylabel('Log of Energy', fontsize=15)

    plt.title(f'VoxelHop Unit {i+1}', fontsize=15)


    cumulative_energy = np.cumsum(pca_params['energy'])
    total_energy = np.sum(pca_params['energy'])

    target_cumulative_energy = 0.95
    index_95_percent = next(i for i, value in enumerate(cumulative_energy) if value >= target_cumulative_energy)
    plt.scatter(x_values[index_95_percent], y_values[index_95_percent], color='red', label='95% Energy', zorder=5)    # 95% 누적 에너지 지점에 빨간색으로 점 찍기

    target_cumulative_energy = 0.96
    index_96_percent = next(i for i, value in enumerate(cumulative_energy) if value >= target_cumulative_energy)
    plt.scatter(x_values[index_96_percent], y_values[index_96_percent], color='orange', label='96% Energy', zorder=5)

    target_cumulative_energy = 0.97
    index_97_percent = next(i for i, value in enumerate(cumulative_energy) if value >= target_cumulative_energy)
    plt.scatter(x_values[index_97_percent], y_values[index_97_percent], color='yellow', label='97% Energy', zorder=5)

    target_cumulative_energy = 0.98
    index_98_percent = next(i for i, value in enumerate(cumulative_energy) if value >= target_cumulative_energy)
    plt.scatter(x_values[index_98_percent], y_values[index_98_percent], color='green', label='98% Energy', zorder=5)

    target_cumulative_energy = 0.99 * total_energy
    index_99_percent = next(i for i, value in enumerate(cumulative_energy) if value >= target_cumulative_energy)
    plt.scatter(x_values[index_99_percent], y_values[index_99_percent], color='blue', label='99% Energy', zorder=5)


    plt.grid(True)
    # plt.gca().yaxis.set_major_formatter('{log10(x):.0f}')
    # plt.yscale('log')


    plt.savefig(f'./Eng_Log/Eng_Log_{i}.png')