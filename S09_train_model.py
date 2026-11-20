
from _model import *
import numpy as np

def split_train_test(df, test_ratio=0.1, random_seed=42):
    indices = np.arange(len(df))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    split_point = int(len(df) * (1 - test_ratio))
    return indices[:split_point].tolist(), indices[split_point:].tolist()

def prepare_datasets(df_merged, train_indices, test_indices):
    train = df_merged.iloc[train_indices]
    test = df_merged.iloc[test_indices]
    
    return (
        train["seqs"].tolist(), train["induced_mpra_means"].tolist(), 
        train["uninduced_mpra_means"].tolist(), train["lfc"].tolist(),
        
        test["seqs"].tolist(), test["induced_mpra_means"].tolist(), 
        test["uninduced_mpra_means"].tolist(), test["lfc"].tolist()
    )

def train_branch(task_name, seqs_file, labels_file, save_dir, length=252, epoch=400, patience=50):
    os.makedirs(save_dir, exist_ok=True)
    model = AttnBilstm_language(length=length, epoch=epoch, patience=patience, model_name=task_name)
    model.train(dataset=seqs_file, labels=labels_file, savepath=save_dir)

# part1: seqs, expr

df_induced = pd.read_csv("./data/7_exp_mean_FUS_TO_induced.tsv", sep="\t", header=None)
df_uninduced = pd.read_csv("./data/7_exp_mean_FUS_TO_uninduced.tsv", sep="\t", header=None)
df_induced.columns = ["seqs", "induced_mpra_means", "names"]
df_uninduced.columns = ["seqs", "uninduced_mpra_means", "names"]

df_merged = pd.merge(df_induced, df_uninduced, on=['names', 'seqs'], how='inner')
df_merged = df_merged[["seqs", "names", "induced_mpra_means", "uninduced_mpra_means"]]
df_merged["lfc"] = df_merged["induced_mpra_means"] / df_merged["uninduced_mpra_means"]


train_indices, test_indices = split_train_test(df_merged)
train_seqs, train_induced, train_uninduced, train_lfc, test_seqs, test_induced, test_uninduced, test_lfc = prepare_datasets(df_merged, train_indices, test_indices)

os.makedirs("./preds/S11_training_data/", exist_ok=True)
write_fa("./preds/S11_training_data/train_seqs.txt", train_seqs, [df_merged["names"][i] for i in train_indices])
write_exp("./preds/S11_training_data/train_induced.txt", train_induced)
write_exp("./preds/S11_training_data/train_uninduced.txt", train_uninduced)
write_exp("./preds/S11_training_data/train_lfc.txt", train_lfc)

write_fa("./preds/S11_training_data/test_seqs.txt", test_seqs, [df_merged["names"][i] for i in test_indices])
write_exp("./preds/S11_training_data/test_induced.txt", test_induced)
write_exp("./preds/S11_training_data/test_uninduced.txt", test_uninduced)
write_exp("./preds/S11_training_data/test_lfc.txt", test_lfc)

# part2: 训练模型

base_dir = './preds/S11_training_data/'
checkpoint_dir = './checks/round1_holdout/'
seqs_file = os.path.join(base_dir, 'train_seqs.txt')

train_branch(
    task_name="induced",
    seqs_file=seqs_file,
    labels_file=os.path.join(base_dir, 'train_induced.txt'),
    save_dir=checkpoint_dir
)

train_branch(
    task_name="uninduced",
    seqs_file=seqs_file,
    labels_file=os.path.join(base_dir, 'train_uninduced.txt'),
    save_dir=checkpoint_dir
)

train_branch(
    task_name="lfc",
    seqs_file=seqs_file,
    labels_file=os.path.join(base_dir, 'train_lfc.txt'),
    save_dir=checkpoint_dir
)

