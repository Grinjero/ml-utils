import matplotlib.pyplot as plt


def plot_loss_function(history, y_limit=None, yscale="linear"):
    """
    Plotting of loss functions mainly for tensorflow models
    :param history: dict containing keys 'loss' and 'val_loss', (tensorflow history contains these)
    :param scale:
    :param y_limit:
    :param yscale:
    :return:
    """
    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(9, 4))
    plt.title("Loss over epochs")
    plt.plot(history.history['loss'], label='training')

    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='val')
    plt.yscale(yscale)
    plt.xlabel('Epoch')
    plt.ylabel('Loss function, {} scale'.format(yscale))
    if y_limit != None:
        plt.ylim(y_limit[0], y_limit[1])
    plt.grid(True)
    plt.legend(loc='upper right')
