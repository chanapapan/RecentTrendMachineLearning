from visdom import Visdom

import numpy as np

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')


class VisdomImages(object):
    """Images to Visdom"""
    def __init__(self,var_name, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.win = var_name

    def show_images(self, images, title_name):
        self.viz.images(
            images,
            opts=dict(title=title_name),
            win = self.win
            )