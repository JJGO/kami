from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


def nb_plot_model(model):
    return SVG(model_to_dot(model, show_shapes=True, show_layer_names=True).create(prog='dot', format='svg'))