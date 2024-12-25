from src.dataset.dataset import FireDataset
import torch
from tqdm import tqdm
import torch.optim as optim
from src.models.vision_transformer import VisionTransformer
from src.utils.arg_parser import arg_parser
from src.utils.early_stopper import EarlyStopper

args = arg_parser()

# Dataset initialization
fire_dataset = FireDataset(
        data_dir=args.data_dir,
        sequence_length=args.sequence_length,
        transform=None,
        split='train',
        weather=False,
        topological_features=False
    )
train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(fire_dataset, [0.8, 0.1, 0.1])
# Dataloader
generator = torch.Generator().manual_seed(123)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           generator=generator,
                                           num_workers=args.num_workers,
                                           drop_last=args.drop_last)

validation_loader = torch.utils.data.DataLoader(validation_dataset,
                                                batch_size=args.batch_size,
                                                generator=generator,
                                                num_workers=args.num_workers,
                                                drop_last=args.drop_last)

model = VisionTransformer(
    in_chans=1
)

if args.device == 'cuda' and torch.cuda.is_available():
    device = torch.device(f'cuda:{args.gpu}')
else:
    device = torch.device('cpu')

model.setup_metrics(threshold=args.metrics_threshold)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# Training
num_epochs = args.num_epochs
early_stopper = EarlyStopper(patience=args.early_stop_patience, min_delta=args.early_stop_minimum_delta)
# Time metrics
training_time_per_epoch_array = []
for epoch in tqdm(range(num_epochs), desc="Epochs".ljust(10), dynamic_ncols=True):
    # Training loop
    model.train_model(train_loader, optimizer, device)

    # Validation loop
    model.validate_model(validation_loader, device)

    # Early stopping
    if early_stopper.early_stop(model.validation_epoch_loss, epoch):
        print('Early stopping at epoch:', epoch)
        break

    # Reset the loss for the next epoch
    model.reset_loss()

# Save the model
model.save_loss_plot('results', num_epochs)
if args.save_weights:
    model.save_weights('results', num_epochs)

# Evaluation
results = model.evaluate(train_loader, device)
print('Evaluation results', results)
# Metadata
print(f"Evaluation results: {results}")
model.save_results(results, "results", num_epochs)
