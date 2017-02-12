from keras.models import load_model
import keras.backend as K

def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output])
    activations = get_activations([X_batch,0])
    return activations

X = np.array([img,])
model = load_model("model.h5")
output = np.array(get_activations(model, 1, X))
print(output.shape)
#plt.imshow(((output[0][0][:,:,:]+0.5)*255).astype(np.uint8),cmap='gray')
