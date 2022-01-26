import numpy as np
import config
import xgboost

class Deploymodel:
    def __init__(
        self
    ):
        """
        We create an init function so we don't have to load the models
        each time we want to make a prediction, since it takes
        several seconds
        """
        self.model_ill = xgboost.Booster()
        self.model_ill.load_model(config.MODEL_ILL)
        self.model_dgp = xgboost.Booster()
        self.model_dgp.load_model(config.MODEL_DGP)

    def return_predictions(
        self,
        altitude=None,
        ibn=None,
        idh=None,
        reference=None,
        blind_angle_n=None,
        blind_angle_s=None,
        azimuth=None,
    ):

        # creating empty numpy array for feature values
        X = np.empty(8).reshape(1, 8)
        
        # calculate azimuth sin and azimuth cos
        # maximum azimuth in the dataset
        max_azimuth = 353.467
        azimuth_sin = np.sin((2*np.pi*azimuth/max_azimuth))
        azimuth_cos = np.cos((2*np.pi*azimuth/max_azimuth))
        
        # set vector values
        X[:,0] = altitude
        X[:,1] = ibn
        X[:,2] = idh
        X[:,3] = 0 # reference feature should always be zero
        X[:,4] = blind_angle_n
        X[:,5] = blind_angle_s
        X[:,6] = azimuth_sin
        X[:,7] = azimuth_cos

        # need to convert array to XGboost dataframe
        X = xgboost.DMatrix(X)

        ill = self.model_ill.predict(X)
        dgp = self.model_dgp.predict(X)

        return ill, dgp

if __name__ == "__main__":

    predict = Deploymodel()
    ill, dgp = predict.return_predictions(
        altitude=27.7491, 
        ibn=648.62, 
        idh=57, 
        blind_angle_n=55, 
        blind_angle_s=35, 
        azimuth=263.169,
    )

    print(ill, dgp)

    # preds for above features [192.51721, 0.05458343]
