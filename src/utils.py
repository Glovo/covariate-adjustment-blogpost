from sklearn.model_selection import train_test_split

def split_and_preprocess(data, treatment_effect):
    train_data, test_data = train_test_split(data, test_size=0.2)
    train_data.loc[train_data["T"] == 1, "Y"] -= treatment_effect
    train_data.loc[train_data["T"] == 1, "T"] = 0
    return train_data, test_data