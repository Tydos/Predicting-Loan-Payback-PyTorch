from sklearn.model_selection import train_test_split

def split_dataset(dataset,config):
    config = config.data_split
    target = config.target_column
    trainset, tempset = train_test_split(
        dataset,
        test_size=config.test_size_1, 
        stratify=dataset[target] if config.stratify else None, # Stratify default is active
        random_state=config.random_state
        )
    
    valset, testset = train_test_split(
        tempset,
        test_size=config.test_size_2,
        stratify=tempset[target] if config.stratify else None,
        random_state=config.random_state
        )
    
    print(len(dataset))
    # print(len(trainset))
    # print(len(valset))
    # print(len(testset))
    
    return trainset, valset, testset

