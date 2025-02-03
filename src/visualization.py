import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go
import numpy as np

def show_first_image(image,show=False):
    image = image[0][0][0]
    plt.imshow(image,cmap='gray')
    plt.colorbar()
    if show:
        plt.show()

def show_first_image_from_loader(image_from_loader,path,name=""):
    image = image_from_loader.data[0].squeeze(0).numpy()
    plt.imshow(image,cmap='gray')
    print(path+name)
    if name!="":
        plt.savefig(path+name)
    plt.clf()


def show_loss(list_train_loss,list_val_loss,path,name=""):
    plt.plot(list_train_loss,label="train")
    plt.plot(list_val_loss,label="val")
    plt.legend()
    print(path+name)
    if name!="":
        plt.savefig(path+name)
    plt.clf()

def show_metric_latent_dim():
    pass

def show_latent_dim_2(class_1,class_2,show=False):
    fig = go.Figure()
    class_1 = np.array(class_1)
    class_2 = np.array(class_2)
    fig = fig.add_trace(
            go.Scatter3d(
                x=class_1[:, 0, 0, 0], # a
                y=class_1[:, 0, 0, 1], # b
                z=class_1[:, 0, 1, 1], # c
                mode="markers",
                name='right_hand',
                marker=dict(size=8, color="pink", opacity=0.9),
            )
        )
    fig = fig.add_trace(
            go.Scatter3d(
                x=class_2[:, 0, 0, 0], # a
                y=class_2[:, 0, 0, 1], # b
                z=class_2[:, 0, 1, 1], # c
                mode="markers",
                name='left_hand',
                marker=dict(size=8, color="green", opacity=0.9),
            )
        )
    fig.update_layout(
        title="Affichage de matrices SPD",
        scene=dict(xaxis_title="a", yaxis_title="b", zaxis_title="c"),
        width=900,
        height=700,
        autosize=False,
        margin=dict(t=30, b=0, l=0, r=0),
        template="plotly_white",
    )
    if show:
        fig.show()

if __name__ == '__main__':
    print(matplotlib.__version__)
    print(matplotlib.__file__)
