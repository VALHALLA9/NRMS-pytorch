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
import polars as pl
import numpy as np
import time
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
from models.dataloader import NRMSDataLoader, NRMSDataLoaderPretransform
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

PATH = Path(args.data_path).expanduser()
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

DOC_VEC_PATH = args.document_embeddings
# print("Train-fraction",TRAIN_FRACTION)
# print("FRACTION_TEST",FRACTION_TEST)
# FRACTION_TEST = 0.01
print("Initiating articles...")
df_articles = pl.read_parquet(DOC_VEC_PATH)
article_mapping = create_article_id_to_value_mapping(
    df=df_articles, value_col=df_articles.columns[-1]
)

print_hparams(hparams)
DUMP_DIR = PATH.joinpath(PATH, "DUMP")
# DUMP_DIR = Path("E:\\desktop\\test\\Dataset\\DUMP")
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

def compute_auc(model, dataloader):
    from sklearn.metrics import roc_auc_score
    model.scorer.eval()  
    all_scores = []
    all_labels = []
    # progress_bar = tqdm(total=len(dataloader), desc="Computing AUC", dynamic_ncols=True)
    
    with torch.no_grad():
        for (his_input_title, pred_input_title), batch_y in dataloader:
            if not isinstance(his_input_title, torch.Tensor):
                his_input_title = torch.from_numpy(his_input_title).float()
            his_input_title = his_input_title.to(device)

            if not isinstance(pred_input_title, torch.Tensor):
                pred_input_title = torch.from_numpy(pred_input_title).float()
            pred_input_title = pred_input_title.to(device)


            scores = model.scorer(his_input_title, pred_input_title[:, 0:1, :])
            scores = scores.cpu().numpy()
            

            labels = batch_y[:, 0].cpu().numpy()
            
            all_scores.extend(scores)
            all_labels.extend(labels)
            
    #         progress_bar.update(1)

    # progress_bar.close()
    return roc_auc_score(all_labels, all_scores)

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.model.train()
    total_loss = 0
    count = 0
    # progress_bar = tqdm(total=len(dataloader), desc="Training", dynamic_ncols=True)

    for batch_idx, ((his_input_title, pred_input_title), batch_y) in enumerate(dataloader):

        if not isinstance(his_input_title, torch.Tensor):
            his_input_title = torch.from_numpy(his_input_title).float()
        his_input_title = his_input_title.to(device)

        if not isinstance(pred_input_title, torch.Tensor):
            pred_input_title = torch.from_numpy(pred_input_title).float()
        pred_input_title = pred_input_title.to(device)

        if not isinstance(batch_y, torch.Tensor):
            batch_y = torch.from_numpy(batch_y).float()
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
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


        # progress_bar.set_postfix(
        #     loss=f"{loss.item():.6f}",
        #     grad_stats=" | ".join(grad_stats[:2])
        # )
        # progress_bar.update(1)

    # progress_bar.close()
    return total_loss / count


def evaluate(model, dataloader, criterion, device):
    model.model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        # progress_bar = tqdm(total=len(dataloader), desc="Evaluating", dynamic_ncols=True)
        for batch_idx, ((his_input_title, pred_input_title), batch_y) in enumerate(dataloader):
            if not isinstance(his_input_title, torch.Tensor):
                his_input_title = torch.from_numpy(his_input_title).float()
            his_input_title = his_input_title.to(device)

            if not isinstance(pred_input_title, torch.Tensor):
                pred_input_title = torch.from_numpy(pred_input_title).float()
            pred_input_title = pred_input_title.to(device)

            if not isinstance(batch_y, torch.Tensor):
                batch_y = torch.from_numpy(batch_y).float()
            batch_y = batch_y.to(device)

            preds = model.model(his_input_title, pred_input_title)

            loss = -torch.sum(batch_y * torch.log(preds + 1e-10)) / batch_y.size(0)
            
            total_loss += loss.item() * len(batch_y)
            count += len(batch_y)

        #     progress_bar.set_postfix(loss=f"{loss.item():.6f}")
        #     progress_bar.update(1)
        # progress_bar.close()

    return total_loss / count

best_auc = -1
# Check for GPU availability and print device info
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("Using CPU for training.")
    

print(f"Initiating {MODEL_NAME}, start training...")
for epoch in range(EPOCHS):

    print(f"Epoch {epoch+1}/{EPOCHS}")
    for param_group in optimizer.param_groups:
        print(f"Current learning rate: {param_group['lr']}")

    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader, criterion, device)
    val_auc = compute_auc(model, val_loader)
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
    
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.model.state_dict(), MODEL_WEIGHTS)
        print("Best model updated")

    scheduler.step(val_auc)
    stop = early_stopping.step(val_auc, model.model)
    if stop:
        print("Early stopping triggered.")
        break

print(f"loading model: {MODEL_WEIGHTS}")
model.model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
model.model.eval()
model.scorer.eval()

print("Initiating testset...")
print("DEFAULT_INVIEW_ARTICLES_COL",DEFAULT_INVIEW_ARTICLES_COL)
df_test = (
    ebnerd_from_path(
        PATH.joinpath(DATASPLIT, "validation"),
        history_size=HISTORY_SIZE,
        padding=0,
    )
    .sample(fraction=FRACTION_TEST)
    .with_columns([
        pl.col("article_ids_clicked").alias(DEFAULT_CLICKED_ARTICLES_COL),
        pl.col("article_ids_inview").alias(DEFAULT_INVIEW_ARTICLES_COL),
        pl.lit(False).alias(DEFAULT_IS_BEYOND_ACCURACY_COL)
    ])
    .select(COLUMNS + [DEFAULT_IS_BEYOND_ACCURACY_COL])
)

df_test = df_test.pipe(create_binary_labels_column)
# df_test = (
#     ebnerd_from_path(
#         PATH.joinpath("ebnerd_testset", "test"),
#         history_size=HISTORY_SIZE,
#         padding=0,
#     )
#     .sample(fraction=FRACTION_TEST)
#     .with_columns(
#         pl.col(DEFAULT_INVIEW_ARTICLES_COL)
#         .list.first()
#         .alias(DEFAULT_CLICKED_ARTICLES_COL)
#     )
#     .select(COLUMNS + [DEFAULT_IS_BEYOND_ACCURACY_COL])
#     .with_columns(
#         pl.col(DEFAULT_INVIEW_ARTICLES_COL)
#         .list.eval(pl.element() * 0)
#         .alias(DEFAULT_LABELS_COL)
#     )
# )


df_test_wo_beyond = df_test.filter(~pl.col(DEFAULT_IS_BEYOND_ACCURACY_COL))
df_test_w_beyond = df_test.filter(pl.col(DEFAULT_IS_BEYOND_ACCURACY_COL))

df_test_chunks = split_df_chunks(df_test_wo_beyond, n_chunks=N_CHUNKS_TEST)
df_pred_test_wo_beyond = []

@torch.no_grad()
def predict_scores(model_scorer, dataloader, device):
    print("predict_scores: Entering function.")
    model_scorer.eval()
    preds_all = []
    batch_counter = 0  
    # progress_bar = tqdm(total=len(dataloader), desc="Processing batches", dynamic_ncols=True)
    start_time = time.time()  
    for batch_idx, ((his_input_title, pred_input_title_one), _) in enumerate(dataloader):
        try:
            batch_start_time = time.time() 
            # print(f"Processing batch {batch_idx + 1}/{len(dataloader)}...")


            if not isinstance(his_input_title, torch.Tensor):
                his_input_title = torch.from_numpy(his_input_title).float()
            his_input_title = his_input_title.to(device)
            # print(f"his_input_title shape: {his_input_title.shape}")

            if not isinstance(pred_input_title_one, torch.Tensor):
                pred_input_title_one = torch.from_numpy(pred_input_title_one).float()
            pred_input_title_one = pred_input_title_one.to(device)
            # print(f"pred_input_title_one shape: {pred_input_title_one.shape}")


            scores = model_scorer(his_input_title, pred_input_title_one)
            # print(f"Scores shape: {scores.shape}")
            preds_all.extend(scores.cpu().tolist())


            # progress_bar.set_postfix(
            #     batch=batch_idx + 1,
            #     batch_size=his_input_title.shape[0]
            #     # scores_shape=scores.shape
            # )
            # progress_bar.update(1)


            # if batch_counter < 5:
            #     print(f"Batch {batch_idx + 1} first 5 scores: {scores[:5].cpu().tolist()}")

            batch_counter += 1
            # print(f"Batch {batch_idx + 1} processed in {time.time() - batch_start_time:.2f} seconds")
        except Exception as e:
            print(f"Error in batch {batch_idx + 1}: {e}")
            raise e

    print("predict_scores: Completed processing all batches.")
    print(f"Total batches processed: {batch_counter}")
    print(f"Total predictions: {len(preds_all)}")
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")
    return np.array(preds_all, dtype=object)


print("Initiating testset without beyond-accuracy...")
for i, df_test_chunk in enumerate(df_test_chunks[CHUNKS_DONE:], start=1 + CHUNKS_DONE):
    print(f"Test chunk: {i}/{len(df_test_chunks)}")
    

    print("Initializing NRMSDataLoader for test chunk...")
    try:
        test_dataloader_wo_b = NRMSDataLoader(
            behaviors=df_test_chunk,
            article_dict=article_mapping,
            unknown_representation="zeros",
            history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
            eval_mode=True,
            batch_size=BATCH_SIZE_TEST_WO_B,
        )
        print("NRMSDataLoader initialized successfully.")
    except Exception as e:
        print(f"Error initializing NRMSDataLoader: {e}")
        raise e


    print("Initializing torch DataLoader...")
    try:
        test_loader_wo_b = torch.utils.data.DataLoader(test_dataloader_wo_b, batch_size=None, shuffle=False)
        print("Torch DataLoader initialized successfully.")
    except Exception as e:
        print(f"Error initializing torch DataLoader: {e}")
        raise e


    print("Calling predict_scores...")
    try:
        scores = predict_scores(model.scorer, test_loader_wo_b, device)
        print("predict_scores completed successfully.")
        # print(f"First 5 scores: {scores[:5] if len(scores) > 5 else scores}")
    except Exception as e:
        print(f"Error during predict_scores: {e}")
        raise e


    # print("Debugging df_test_chunk before add_prediction_scores...")
    # print(f"Columns: {df_test_chunk.columns}")
    # print(f"Shape: {df_test_chunk.shape}")
    # print("First 5 rows of df_test_chunk:")
    # print(df_test_chunk.head(5))


    print("Calling add_prediction_scores...")
    try:
        # print("Debugging df_test_chunk before add_prediction_scores:")
        # print(df_test_chunk.head(5))
        # print("Columns:", df_test_chunk.columns)
        # print("Schema:", df_test_chunk.schema)

        # print("Debugging scores before add_prediction_scores:")
        # print("Scores type:", type(scores))
        # print("Scores shape:", np.array(scores).shape if isinstance(scores, (list, np.ndarray)) else "N/A")
        # print("First 5 scores:", scores[:5])



        df_test_chunk = add_prediction_scores(df_test_chunk, scores.tolist()).with_columns(
            pl.col("scores")
            .map_elements(lambda x: list(rank_predictions_by_score(x)), return_dtype=pl.List(pl.Int32))
            .alias("ranked_scores")
        )

        print("add_prediction_scores completed successfully.")
    except Exception as e:
        print(f"Error during add_prediction_scores: {e}")
        raise e


    print("Saving results to Parquet...")
    try:
        df_test_chunk.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
            TEST_CHUNKS_DIR.joinpath(f"pred_wo_ba_{i}.parquet")
        )
        print(f"Test chunk {i} processed and saved.")
    except Exception as e:
        print(f"Error saving results to Parquet: {e}")
        raise e


    df_pred_test_wo_beyond.append(df_test_chunk)
    del df_test_chunk, test_dataloader_wo_b, scores
    gc.collect()
    print(f"Memory cleaned for chunk {i}.")



df_pred_test_wo_beyond = pl.concat(df_pred_test_wo_beyond)
df_pred_test_wo_beyond.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
    TEST_CHUNKS_DIR.joinpath("pred_wo_ba.parquet")
)

print("Initiating testset with beyond-accuracy...")
test_dataloader_w_b = NRMSDataLoader(
    behaviors=df_test_w_beyond,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=True,
    batch_size=BATCH_SIZE_TEST_W_B,
)
test_loader_w_b = torch.utils.data.DataLoader(test_dataloader_w_b, batch_size=None, shuffle=False)
scores = predict_scores(model.scorer, test_loader_w_b, device)
df_pred_test_w_beyond = add_prediction_scores(df_test_w_beyond, scores.tolist()).with_columns(
    pl.col("scores")
    .map_elements(lambda x: list(rank_predictions_by_score(x)), return_dtype=pl.List(pl.Int32))  
    .alias("ranked_scores")
)
df_pred_test_w_beyond.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
    TEST_CHUNKS_DIR.joinpath("pred_w_ba.parquet")
)

print("Saving prediction results...")
df_test = pl.concat([df_pred_test_wo_beyond, df_pred_test_w_beyond])
df_test.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
    ARTIFACT_DIR.joinpath("test_predictions.parquet")
)
print("Evaluating test set predictions...")
labels = [np.array(label) for label in df_test["labels"].to_list()]
scores = [np.array(score) for score in df_test["scores"].to_list()]

print("Sample converted labels shape:", labels[0].shape)
print("Sample converted scores shape:", scores[0].shape)

test_metrics = MetricEvaluator(
    labels=labels,
    predictions=scores,
    metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
)
test_results = test_metrics.evaluate()
print("Test Set Results:")
print(f"AUC: {test_results.evaluations['auc']:.4f}")
print(f"MRR: {test_results.evaluations['mrr']:.4f}")
print(f"NDCG@5: {test_results.evaluations['ndcg@5']:.4f}")
print(f"NDCG@10: {test_results.evaluations['ndcg@10']:.4f}")


write_json_file(
    test_results.evaluations,
    ARTIFACT_DIR.joinpath("test_metrics.json"),
)

if TEST_CHUNKS_DIR.exists() and TEST_CHUNKS_DIR.is_dir():
    shutil.rmtree(TEST_CHUNKS_DIR)

write_submission_file(
    impression_ids=df_test[DEFAULT_IMPRESSION_ID_COL],
    prediction_scores=df_test["ranked_scores"],
    path=ARTIFACT_DIR.joinpath("predictions.txt"),
    filename_zip=f"{MODEL_NAME}-{SEED}-{DATASPLIT}.zip",
)
