import pandas as pd
import numpy as np
import config
import glob
import re

def prepare_dataset(

):

    """
    Reads raw data from RADIANCE sofware and weather data and 
    produces one full .csv file
    """

    cols = ['month', 'day', 
            'hour', 'azimuth', 
            'altitude', 'ibn', 
            'idh', 'reference', 
            'blind_angle_n', 'blind_angle_s',
            'ill', 'dgp']

    # data paths
    ill_path = config.ILL_RAW_DATA
    dgp_path = config.DGP_RAW_DATA
    weather_path = config.WEATHER_DATA

    # loading reference data
    ill_ref = pd.read_csv(ill_path + "/wr1_Ref.ill", sep='\s+', skiprows=13, header=None)
    dgp_ref = pd.read_csv(dgp_path + "/wr1-Ref.txt", sep=',| |{:*}', engine="python", header=None)
    radiation = pd.read_csv(weather_path + "/fribourg-solar-direct-diffuse.txt", sep='\t', engine="python", skiprows=2, index_col=0)
    sun_position = pd.read_csv(weather_path + "/fribourg-sun position.txt", sep='\t', skiprows=3, index_col=0, header=None)

    reference = np.ones([8760,1])
    blind_angle_north = np.zeros([8760,1])
    blind_angle_south = np.zeros([8760,1])

    #converting to arrays
    ill = np.mean(ill_ref.values, axis=1).reshape(8760,1)
    dgp = dgp_ref[6].values.reshape(8760,1)
    radiation = np.array(radiation)
    sun_position = np.array(sun_position)

    data = np.concatenate((sun_position, radiation, reference, blind_angle_north, blind_angle_south, ill, dgp), axis=1)

    ill_files = sorted(glob.glob(ill_path + "/*.ill"))
    dgp_files = sorted(glob.glob(dgp_path + "/*.txt"))

    frames = [data]
    counter = 1
    print(f"DataFrame {counter} has {data.shape[0]} rows and {data.shape[1]} columns")

    for ill_file, dgp_file in zip(ill_files, dgp_files):
        if ill_file.split('/')[-1] == 'wr1_Ref.ill' or dgp_file.split('/')[-1] == 'wr1-Ref.txt':
            assert ill_file.split('/')[-1] == 'wr1_Ref.ill' and dgp_file.split('/')[-1] == 'wr1-Ref.txt'
            continue
        south_angle_ill, north_angle_ill = re.split('St|-|Nt|.ill', ill_file)[1], re.split('St|-|Nt|.ill', ill_file)[3]
        south_angle_dgp, north_angle_dgp = re.split('St|-|Nt|.txt', dgp_file)[2], re.split('St|-|Nt|.txt', dgp_file)[4]
        assert south_angle_ill == south_angle_dgp
        assert north_angle_ill == north_angle_dgp
        
        ill = pd.read_csv(ill_file, sep='\s+', skiprows=13, header=None)
        ill = np.mean(ill.values, axis=1).reshape(8760,1)
        
        dgp = pd.read_csv(dgp_file, sep=',| |{:*}', engine="python", header=None)
        dgp = dgp[6].values.reshape(8760,1)
        
        blind_angle_south = np.empty([8760,1])
        blind_angle_north = np.empty([8760,1])
        ref = np.zeros([8760,1])
        
        blind_angle_south[:] = south_angle_ill
        blind_angle_south = blind_angle_south.astype('float64')

        blind_angle_north[:] = north_angle_ill
        blind_angle_north = blind_angle_north.astype('float64')
        
        next_data = np.concatenate((sun_position, radiation, ref, blind_angle_north, blind_angle_south, ill, dgp), axis=1)
        
        frames.append(next_data)
        counter += 1
        print(f"DataFrame {counter} has {next_data.shape[0]} rows and {next_data.shape[1]} columns")
        

    df = np.concatenate(frames)
    df = pd.DataFrame(df, columns=cols)
    df.to_csv(config.DATA_PATH + "data.csv", index=False)

def transform_azimuth(
    df,
):
    """
    Takes df and transforms the azimuth feature to sin and cosine azimuth
    Saves new file in data directory
    """

    df = pd.read_csv(df)

    df['azimuth_sin'] = np.sin((2*np.pi*df['azimuth'])/max(df['azimuth']))
    df['azimuth_cos'] = np.cos((2*np.pi*df['azimuth'])/max(df['azimuth']))
    print(max(df['azimuth']))

    df = df.drop('azimuth', axis=1)
    
    #df.to_csv(config.DATA_PATH + "data_transform.csv", index=False)

def shuffle_and_split_data(
    df,
):

    """
    Takes a DataFrame and shuffles along first axis
    then creates training and test sets
    We save the ordering so that we can reverse for plotting at a
    later stage
    """

    df = pd.read_csv(df)

    # for reproducibility
    np.random.seed(6)
    order = list(range(df.shape[0]))
    np.random.shuffle(order)
    # saving the shuffle order for later
    np.save(config.OUTPUTS_PATH + '/shuff_order.npy', order)

    # shuffling dataframe by numpy array index 
    df = df.loc[order].reset_index(drop=True)
    
    # splitting data into training and testing
    # representing 95% and 5% respectively the 
    # dataset is quite large, therefore 5% should be a 
    # large enough sample for test set (53,874) datapoints

    split_index = df.shape[0]
    train = df.loc[0:int(split_index*0.95)]
    test = df.loc[int(split_index*0.95) + 1:]

    print(f"Train has shape: {train.shape}")
    print(f"Test has shape: {test.shape}")

    train_ill = train.drop('dgp', axis=1)
    train_dgp = train.drop('ill', axis=1)

    test_ill = test.drop('dgp', axis=1)
    test_dgp = test.drop('ill', axis=1)

    # creating separate datasets for ill and dgp, so that
    # we can use stratified K-folds for each target
    train_ill.to_csv(config.DATA_PATH + "train_ill.csv", index=False)
    train_dgp.to_csv(config.DATA_PATH + "train_dgp.csv", index=False)
    test_ill.to_csv(config.DATA_PATH + "test_ill.csv", index=False)
    test_dgp.to_csv(config.DATA_PATH + "test_dgp.csv", index=False)


if __name__ == "__main__":
    #prepare_dataset()
    transform_azimuth(config.DATA_RAW)
    #shuffle_and_split_data(config.DATA)
