import matplotlib.pyplot as plt

def show_first_image(image):
    image = image[0][0][0]
    plt.imshow(image,cmap='gray')
    plt.colorbar()

def show_first_image_from_loader(image_from_loader):
    image = image_from_loader.data[0].squeeze(0).numpy()
    plt.imshow(image,cmap='gray')
    plt.colorbar()

def show_loss(list_train_loss,list_val_loss):
    plt.plot(list_train_loss,label="train")
    plt.plot(list_val_loss,label="val")
    plt.legend()
    plt.show()

def show_metric_latent_dim():
    pass

