import datetime as dt
import os
import shutil
import gc
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

# from torch.utils.tensorboard import SummaryWriter

import random
import time
import torch.nn.functional as F

import polars as pl
import numpy as np

from utils._constants import *
from utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    add_prediction_scores,
    truncate_history,
    ebnerd_from_path,
)
from evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore
from utils._python import (
    write_submission_file,
    rank_predictions_by_score,
    write_json_file,
)
from utils._articles import create_article_id_to_value_mapping
from utils._polars import split_df_chunks
from models.dataloader2 import NRMSDataLoader, NRMSDataLoaderPretransform
from models.model_config import (
    hparams_nrms_docvec,
    hparams_to_dict,
    print_hparams,
)
from models.nrms_docvec import NRMSDocVec
from args_nrms_docvec import get_args

os.environ["TOKENIZERS_PARALLELISM"] = "false"


args = get_args()
for arg, val in vars(args).items():
    print(f"{arg} : {val}")


def set_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

if args.seed is None:
    args.seed = int(time.time())
set_seed(args.seed)


SEED = args.seed
DATASPLIT = args.datasplit
DEBUG = args.debug
BS_TRAIN = args.bs_train
BS_TEST = args.bs_test
BATCH_SIZE_TEST_WO_B = args.batch_size_test_wo_b
BATCH_SIZE_TEST_W_B = args.batch_size_test_w_b
HISTORY_SIZE = args.history_size
NPRATIO = args.npratio
EPOCHS = args.epochs
TRAIN_FRACTION = args.train_fraction if not DEBUG else 0.0001
FRACTION_TEST = args.fraction_test if not DEBUG else 0.0001

NRMSLoader_training = (
    NRMSDataLoaderPretransform
    if args.nrms_loader == "NRMSDataLoaderPretransform"
    else NRMSDataLoader
)

model_func = NRMSDocVec
hparams = hparams_nrms_docvec
hparams.title_size = args.title_size
hparams.history_size = args.history_size
hparams.head_num = args.head_num
hparams.head_dim = args.head_dim
hparams.attention_hidden_dim = args.attention_hidden_dim
hparams.newsencoder_units_per_layer = args.newsencoder_units_per_layer
hparams.optimizer = args.optimizer
hparams.loss = args.loss
hparams.dropout = args.dropout
hparams.learning_rate = args.learning_rate
hparams.newsencoder_l2_regularization = args.newsencoder_l2_regularization

PATH = Path(args.data_path).resolve()  
DOC_VEC_PATH = Path(args.document_embeddings).resolve()  

print("Initiating articles...")
df_articles = pl.read_parquet(DOC_VEC_PATH)
article_mapping = create_article_id_to_value_mapping(
    df=df_articles, value_col=df_articles.columns[-1]
)

print_hparams(hparams)


CURRENT_DIR = Path(__file__).parent


DUMP_DIR = CURRENT_DIR / "Dataset" / "DUMP"
DUMP_DIR.mkdir(exist_ok=True, parents=True)

DT_NOW = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
MODEL_NAME = model_func.__name__
MODEL_OUTPUT_NAME = f"{MODEL_NAME}-{DT_NOW}"

ARTIFACT_DIR = DUMP_DIR.joinpath("test_predictions", MODEL_OUTPUT_NAME)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_WEIGHTS = DUMP_DIR.joinpath(f"state_dict/{MODEL_OUTPUT_NAME}/weights.pt")
MODEL_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
LOG_DIR = DUMP_DIR.joinpath(f"runs/{MODEL_OUTPUT_NAME}")

TEST_CHUNKS_DIR = ARTIFACT_DIR.joinpath("test_chunks")
TEST_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

N_CHUNKS_TEST = args.n_chunks_test
CHUNKS_DONE = args.chunks_done

COLUMNS = [
    DEFAULT_IMPRESSION_TIMESTAMP_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_USER_COL,
]

write_json_file(
    hparams_to_dict(hparams),
    ARTIFACT_DIR.joinpath(f"{MODEL_NAME}_hparams.json"),
)
write_json_file(vars(args), ARTIFACT_DIR.joinpath(f"{MODEL_NAME}_argparser.json"))

df = (
    pl.concat(
        [
            ebnerd_from_path(
                PATH.joinpath(DATASPLIT, "train"),
                history_size=HISTORY_SIZE,
                padding=0,
            ),
            ebnerd_from_path(
                PATH.joinpath(DATASPLIT, "validation"),
                history_size=HISTORY_SIZE,
                padding=0,
            ),
        ]
    )
    .sample(fraction=TRAIN_FRACTION, shuffle=True, seed=SEED)
    .select(COLUMNS)
    .pipe(
        sampling_strategy_wu2019,
        npratio=NPRATIO,
        shuffle=True,
        with_replacement=True,
        seed=SEED,
    )
    .pipe(create_binary_labels_column)
)

last_dt = df[DEFAULT_IMPRESSION_TIMESTAMP_COL].dt.date().max() - dt.timedelta(days=1)
df_train = df.filter(pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL).dt.date() < last_dt)
df_validation = df.filter(pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL).dt.date() >= last_dt)

print("Initiating training-dataloader")
train_dataset = NRMSLoader_training(
    behaviors=df_train,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=False,
    batch_size=BS_TRAIN,
    use_time_features=True
)
val_dataset = NRMSLoader_training(
    behaviors=df_validation,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=False,
    batch_size=BS_TRAIN,
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=None, shuffle=False)

model = model_func(hparams=hparams, seed=42) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.model.to(device)
model.scorer.to(device)
optimizer = model.optimizer 
criterion = model.criterion  

# writer = SummaryWriter(log_dir=str(LOG_DIR))


class EarlyStopping:
    def __init__(self, patience=4, mode='max', restore_best_weights=True):
        self.patience = patience
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_state_dict = None

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        improve = (score > self.best_score) if self.mode == 'max' else (score < self.best_score)
        if improve:
            self.best_score = score
            self.counter = 0
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_state_dict)
                return True
            return False
# early_stopping = EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True)
early_stopping = EarlyStopping(patience=4, mode="max", restore_best_weights=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=2, min_lr=1e-6)

def analyze_time_impact(model, test_loader, device):
    """time"""
    model.eval()
    time_effects = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch[0]) == 4:
                (his_input, pred_input, his_time, pred_time), _ = batch
                

                pred_with_time = model(his_input, pred_input, his_time, pred_time)
                pred_without_time = model(his_input, pred_input)
                
                time_effect = torch.abs(pred_with_time - pred_without_time).mean()
                time_effects.append(time_effect.item())
    
    return np.mean(time_effects)

def compute_auc(model, dataloader):
    from sklearn.metrics import roc_auc_score
    model.scorer.eval() 
    all_scores = []
    all_labels = []
    progress_bar = tqdm(total=len(dataloader), desc="Computing AUC", dynamic_ncols=True)
    
    with torch.no_grad():
        for batch_data in dataloader:

            if len(batch_data[0]) == 4:  # time
                (his_input_title, pred_input_title, his_time_delta, pred_time_delta), batch_y = batch_data
            else:  
                (his_input_title, pred_input_title), batch_y = batch_data
                his_time_delta, pred_time_delta = None, None


            if not isinstance(his_input_title, torch.Tensor):
                his_input_title = torch.from_numpy(his_input_title).float()
            his_input_title = his_input_title.to(device)

            if not isinstance(pred_input_title, torch.Tensor):
                pred_input_title = torch.from_numpy(pred_input_title).float()
            pred_input_title = pred_input_title.to(device)

            if his_time_delta is not None:
                if not isinstance(his_time_delta, torch.Tensor):
                    his_time_delta = torch.from_numpy(his_time_delta).float()
                if not isinstance(pred_time_delta, torch.Tensor):
                    pred_time_delta = torch.from_numpy(pred_time_delta).float()
                his_time_delta = his_time_delta.to(device)
                pred_time_delta = pred_time_delta.to(device)


            if his_time_delta is not None:
                scores = model.scorer(his_input_title, pred_input_title[:, 0:1, :], 
                                   his_time_delta, pred_time_delta[:, 0:1])
            else:
                scores = model.scorer(his_input_title, pred_input_title[:, 0:1, :])
            scores = scores.cpu().numpy()
            

            labels = batch_y[:, 0].cpu().numpy()
            
            all_scores.extend(scores)
            all_labels.extend(labels)
            
            progress_bar.update(1)

    progress_bar.close()
    return roc_auc_score(all_labels, all_scores)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.model.train()
    total_loss = 0
    count = 0
    progress_bar = tqdm(total=len(dataloader), desc="Training", dynamic_ncols=True)

    for batch_idx, batch_data in enumerate(dataloader):

        if len(batch_data[0]) == 4:  # time
            (his_input_title, pred_input_title, his_time_delta, pred_time_delta), batch_y = batch_data
        else:  
            (his_input_title, pred_input_title), batch_y = batch_data
            his_time_delta, pred_time_delta = None, None


        if not isinstance(his_input_title, torch.Tensor):
            his_input_title = torch.from_numpy(his_input_title).float()
        his_input_title = his_input_title.to(device)

        if not isinstance(pred_input_title, torch.Tensor):
            pred_input_title = torch.from_numpy(pred_input_title).float()
        pred_input_title = pred_input_title.to(device)

        if his_time_delta is not None:
            if not isinstance(his_time_delta, torch.Tensor):
                his_time_delta = torch.from_numpy(his_time_delta).float()
            if not isinstance(pred_time_delta, torch.Tensor):
                pred_time_delta = torch.from_numpy(pred_time_delta).float()
            his_time_delta = his_time_delta.to(device)
            pred_time_delta = pred_time_delta.to(device)

        if not isinstance(batch_y, torch.Tensor):
            batch_y = torch.from_numpy(batch_y).float()
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        

        if his_time_delta is not None:
            preds = model.model(his_input_title, pred_input_title, his_time_delta, pred_time_delta)
        else:
            preds = model.model(his_input_title, pred_input_title)

        # categorical_crossentropy
        loss = -torch.sum(batch_y * torch.log(preds + 1e-10)) / batch_y.size(0)
        loss.backward()


        grad_stats = []
        for name, param in model.model.named_parameters():
            if param.grad is not None:
                grad_stats.append(f"{name}: grad_mean={param.grad.mean().item():.6f}")

        optimizer.step()
        total_loss += loss.item() * len(batch_y)
        count += len(batch_y)


        progress_bar.set_postfix(
            loss=f"{loss.item():.6f}",
            grad_stats=" | ".join(grad_stats[:2])
        )
        progress_bar.update(1)

    progress_bar.close()
    return total_loss / count


def evaluate(model, dataloader, criterion, device):
    model.model.eval()
    total_loss = 0
    count = 0
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        progress_bar = tqdm(total=len(dataloader), desc="Evaluating", dynamic_ncols=True)
        for batch_idx, batch_data in enumerate(dataloader):

            if len(batch_data[0]) == 4:  # time
                (his_input_title, pred_input_title, his_time_delta, pred_time_delta), batch_y = batch_data
            else: 
                (his_input_title, pred_input_title), batch_y = batch_data
                his_time_delta, pred_time_delta = None, None


            if not isinstance(his_input_title, torch.Tensor):
                his_input_title = torch.from_numpy(his_input_title).float()
            his_input_title = his_input_title.to(device)

            if not isinstance(pred_input_title, torch.Tensor):
                pred_input_title = torch.from_numpy(pred_input_title).float()
            pred_input_title = pred_input_title.to(device)

            if his_time_delta is not None:
                if not isinstance(his_time_delta, torch.Tensor):
                    his_time_delta = torch.from_numpy(his_time_delta).float()
                if not isinstance(pred_time_delta, torch.Tensor):
                    pred_time_delta = torch.from_numpy(pred_time_delta).float()
                his_time_delta = his_time_delta.to(device)
                pred_time_delta = pred_time_delta.to(device)

            if not isinstance(batch_y, torch.Tensor):
                batch_y = torch.from_numpy(batch_y).float()
            batch_y = batch_y.to(device)


            if his_time_delta is not None:
                preds = model.model(his_input_title, pred_input_title, his_time_delta, pred_time_delta)
            else:
                preds = model.model(his_input_title, pred_input_title)


            loss = -torch.sum(batch_y * torch.log(preds + 1e-10)) / batch_y.size(0)
            total_loss += loss.item() * len(batch_y)
            count += len(batch_y)
            

            for i in range(batch_y.size(0)):
                all_labels.append(batch_y[i].cpu().numpy())
                all_scores.append(preds[i].cpu().numpy())
            
            progress_bar.set_postfix(loss=f"{loss.item():.6f}")
            progress_bar.update(1)
        progress_bar.close()


    print("Sample data shapes:")
    print("First label shape:", all_labels[0].shape if all_labels else "Empty")
    print("First score shape:", all_scores[0].shape if all_scores else "Empty")
    print("Number of samples:", len(all_labels))


    y_true = all_labels
    y_pred = all_scores


    metrics_dict = {}
    auc_metric = AucScore()
    mrr_metric = MrrScore()
    ndcg_5_metric = NdcgScore(k=5)
    ndcg_10_metric = NdcgScore(k=10)
    
    try:
        metrics_dict['auc'] = auc_metric.calculate(y_true, y_pred)
        metrics_dict['mrr'] = mrr_metric.calculate(y_true, y_pred)
        metrics_dict['ndcg@5'] = ndcg_5_metric.calculate(y_true, y_pred)
        metrics_dict['ndcg@10'] = ndcg_10_metric.calculate(y_true, y_pred)
    except Exception as e:
        print("Error in metric calculation:", str(e))
        print("Example y_true:", y_true[0])
        print("Example y_pred:", y_pred[0])

    return total_loss / count, metrics_dict

best_auc = -1
# Check for GPU availability and print device info
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("Using CPU for training.")
    

best_auc = -1
print(f"Initiating {MODEL_NAME}, start training...")
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    for param_group in optimizer.param_groups:
        print(f"Current learning rate: {param_group['lr']}")

    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
    
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val AUC: {val_metrics.get('auc', 0.0):.4f}")
    print(f"Val MRR: {val_metrics.get('mrr', 0.0):.4f}")
    print(f"Val NDCG@5: {val_metrics.get('ndcg@5', 0.0):.4f}")
    print(f"Val NDCG@10: {val_metrics.get('ndcg@10', 0.0):.4f}")
    
    if val_metrics.get('auc', 0.0) > best_auc:
        best_auc = val_metrics.get('auc', 0.0)
        torch.save(model.model.state_dict(), MODEL_WEIGHTS)
        print("Best model updated")

    scheduler.step(val_metrics.get('auc', 0.0))
    stop = early_stopping.step(val_metrics.get('auc', 0.0), model.model)
    if stop:
        print("Early stopping triggered.")
        break

print(f"loading model: {MODEL_WEIGHTS}")
model.model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
model.model.eval()
model.scorer.eval()

print("Initiating testset...")

print("Loading test data...")
with tqdm(total=2, desc="Loading data files") as pbar:
    test_behaviors = pl.read_parquet(PATH.joinpath("ebnerd_testset", "test", "behaviors.parquet"))
    pbar.update(1)
    test_history = pl.read_parquet(PATH.joinpath("ebnerd_testset", "test", "history.parquet"))
    pbar.update(1)

print(f"Loaded {len(test_behaviors)} test behaviors and {len(test_history)} history records")


print(f"Applying test fraction {FRACTION_TEST}")
if not DEBUG:
    test_behaviors = test_behaviors.sample(fraction=FRACTION_TEST, seed=SEED)
else:
    test_behaviors = test_behaviors.sample(n=min(100, len(test_behaviors)), seed=SEED)

print(f"After sampling: {len(test_behaviors)} test behaviors")


print("Merging and processing data...")
with tqdm(total=3, desc="Processing data") as pbar:

    df_test = test_behaviors.join(
        test_history,
        on="user_id",
        how="left"
    )
    pbar.update(1)
    

    print("Truncating history...")
    df_test = truncate_history(
        df=df_test,
        column="article_id_fixed",
        history_size=HISTORY_SIZE,
        padding_value=0
    )
    pbar.update(1)
    

    print("Processing columns...")
    if "article_id_fixed" not in df_test.columns:
        print("Warning: article_id_fixed column not found")
        print("Available columns:", df_test.columns)
    
    df_test = df_test.with_columns([

        pl.when(pl.col("article_id_fixed").is_null())
        .then(pl.lit([0] * HISTORY_SIZE))
        .otherwise(pl.col("article_id_fixed"))
        .alias(DEFAULT_HISTORY_ARTICLE_ID_COL),
        

        pl.col("article_ids_inview").alias(DEFAULT_INVIEW_ARTICLES_COL)
    ])
    pbar.update(1)

print(f"Processed test set shape: {df_test.shape}")
print("Sample of processed data:")
print(df_test.head(2))

print(f"Processed test set shape: {df_test.shape}")

print("Current PATH:", PATH)
print("Test data path:", PATH.joinpath("ebnerd_testset", "test"))
print(f"Path exists: {PATH.joinpath('ebnerd_testset', 'test').exists()}")


print("Splitting into chunks...")
df_test_chunks = split_df_chunks(df_test, n_chunks=N_CHUNKS_TEST)
print(f"Created {len(df_test_chunks)} chunks")
df_pred_chunks = []

@torch.no_grad()
def predict_scores(model_scorer, dataloader, device):
    model_scorer.eval()
    preds_all = []
    
    progress_bar = tqdm(
        total=len(dataloader), 
        desc="Predicting scores", 
        dynamic_ncols=True
    )
    
    for (his_input_title, pred_input_title_one), _ in dataloader:
        if not isinstance(his_input_title, torch.Tensor):
            his_input_title = torch.from_numpy(his_input_title).float()
        his_input_title = his_input_title.to(device)

        if not isinstance(pred_input_title_one, torch.Tensor):
            pred_input_title_one = torch.from_numpy(pred_input_title_one).float()
        pred_input_title_one = pred_input_title_one.to(device)

        scores = model_scorer(his_input_title, pred_input_title_one)
        preds_all.extend(scores.cpu().tolist())
        
        progress_bar.update(1)
    
    progress_bar.close()
    return np.array(preds_all, dtype=object)


print("Processing test chunks...")
for i, df_test_chunk in enumerate(df_test_chunks[CHUNKS_DONE:], start=1 + CHUNKS_DONE):
    print(f"\nProcessing chunk {i}/{len(df_test_chunks)}")
    print(f"Chunk size: {len(df_test_chunk)} records")
    
    test_dataloader = NRMSDataLoader(
        behaviors=df_test_chunk,
        article_dict=article_mapping,
        unknown_representation="zeros",
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        eval_mode=True,  
        batch_size=BATCH_SIZE_TEST_WO_B,
        history_size=HISTORY_SIZE
    )
    test_loader = torch.utils.data.DataLoader(test_dataloader, batch_size=None, shuffle=False)
    scores = predict_scores(model.scorer, test_loader, device)
    
    df_test_chunk = add_prediction_scores(df_test_chunk, scores.tolist())
    df_test_chunk = df_test_chunk.with_columns([
        pl.col("scores")
        .map_elements(lambda x: list(rank_predictions_by_score(x)), return_dtype=pl.List(pl.Float64))
        .alias("ranked_scores")
    ])
    
    df_test_chunk.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
        TEST_CHUNKS_DIR.joinpath(f"pred_{i}.parquet")
    )
    df_pred_chunks.append(df_test_chunk)
    del df_test_chunk, test_dataloader, scores
    gc.collect()


print("\nMerging prediction results...")
df_test = pl.concat(df_pred_chunks)

print("Saving final predictions...")
df_test.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
    ARTIFACT_DIR.joinpath("test_predictions.parquet")
)


if TEST_CHUNKS_DIR.exists() and TEST_CHUNKS_DIR.is_dir():
    shutil.rmtree(TEST_CHUNKS_DIR)


print("\nGenerating submission file...")
df_test = df_test.sort(DEFAULT_IMPRESSION_ID_COL)    
write_submission_file(
    impression_ids=df_test[DEFAULT_IMPRESSION_ID_COL],
    prediction_scores=df_test["ranked_scores"],
    path=ARTIFACT_DIR.joinpath("predictions.txt"),
    filename_zip=f"{MODEL_NAME}-{SEED}-{DATASPLIT}.zip",
)

print("\nTest set prediction completed!")