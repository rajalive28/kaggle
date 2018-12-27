import tensorflow as tf
import numpy as np
import pandas as pd


# @author Abhishek Raj
class TitanicTrainingService:

    def train(self):
        tf.set_random_seed(123)
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(6, activation=tf.nn.relu),
            tf.keras.layers.Dense(6, activation=tf.nn.relu),
            tf.keras.layers.Dense(4, activation=tf.nn.relu),
            tf.keras.layers.Dense(2, activation=tf.nn.softmax)
        ])
        model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(np.asarray(self.train_x.values), np.asarray(self.train_y.values), batch_size=16, epochs=100)
        return model.predict(np.asarray(self.test_x.values))

    # Here I am considering only important features.
    # Converting those into categorical values
    def preprocess(self, df_train, df_test):
        trfm_trn_x = df_train[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
        encode = {"Sex": {"male": 1, "female": 2}, "Embarked": {"S": 1, "C": 2, "Q": 3}}
        self.train_x = trfm_trn_x.replace(encode)
        self.train_y = df_train[['Survived']]
        trfm_trn_y = df_test[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
        self.test_x = trfm_trn_y.replace(encode)

    def driver_program(self):
        df_train = pd.read_csv("./train.csv")
        df_test = pd.read_csv("./test.csv")
        df_train = df_train.fillna(0)
        df_test = df_test.fillna(0)
        print('###### Initiating Preprocessing ##########')
        # Call preprocess function
        self.preprocess(df_train, df_test)
        # Call the model
        res = self.train()
        print(res)
        res_df=pd.DataFrame({
            "label_0":res[:,0],
            "label_1":res[:,1]
        })
        res_df["label_2"]=np.where(res_df["label_1"]>res_df["label_0"],1,0)

        # write to csv file
        pd_df = pd.DataFrame({'PassengerId': self.test_x['PassengerId'], 'Survived':res_df["label_2"]})
        print(pd_df.head())
        pd_df.to_csv("submission.csv",index=False)


if __name__ == "__main__":
    tts = TitanicTrainingService()
    tts.driver_program()
