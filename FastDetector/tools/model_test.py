"""
Some model ideas for referencing

"""



def get_model():
    x = Input(shape=(200,200,3))
    #for _ in range(5):
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPooling2D((3,3))(x)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPooling2D((3,3))(x)
    x = Conv2D(64, (3,3), activation='relu')(x)    
    x = MaxPooling2D((3,3))(x)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = GlobalAveragePooling2D()(x)


    classifier = Dropout(0.3)(x)
    classifier = Dense(10, activation='softmax', name='label')(classifier)


    head = Dense(64, activation='relu')(x)
    head = Dense(32, activation='relu')(head)
    head = Dense(4, activation='sigmoid', name='bbox')(head)
    return Model(inputs=[inputs], outputs=[classifier, head])
    
def get_model():
    inputs = Input(shape=(224,224,3))
    x = Conv2D(64, (224,224), activation='relu')(inputs)
    x = Conv2D(64, (224,224), activation='relu')(x)
    x = MaxPooling2D(64, (224,224))(x)
    x = Conv2D(128, (112,112), activation='relu')(x)
    x = Conv2D(128, (112,112), activation='relu')(x)

    # TODO
    
    x = Conv2D(32, (3,3), activation='relu')(x)
    x = MaxPooling2D((3,3))(x)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    classifier = Dropout(0.3)(x)
    classifier = Dense(10, activation='softmax', name='label')(classifier)
    head = Dense(64, activation='relu')(x)
    head = Dense(32, activation='relu')(head)
    head = Dense(4, activation='sigmoid', name='bbox')(head)
    return Model(inputs=[inputs], outputs=[classifier, head])