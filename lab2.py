#import necessary moduls
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def main():
    # Build FCFNN
    inputs = Input((1,))
    h1 = Dense(4, activation='relu', name='hidden_layer1')(inputs)
    h2 = Dense(8, activation='relu', name='hidden_layer2')(h1)  
    h3 = Dense(4, activation='relu', name='hidden_layer3')(h2)
    outputs = Dense(1, activation= 'sigmoid', name = 'output_layer')(h3)
    model = Model(inputs, outputs)
    model.summary(show_trainable=True)

if __name__ == '__main__':
    main()