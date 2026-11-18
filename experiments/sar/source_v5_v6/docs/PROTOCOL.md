# New experiment protocol

## Data

Preserve original dataset test splits. Choose any resource-bounded test subset
once using a fixed seed; record that it is a subset benchmark.

Derive validation only from the original training pool. Freeze train, validation
and test image lists and content fingerprints; use exactly the same lists for
all models. Check image overlap and duplicates, and group related scenes where
metadata permits. Do not reuse an old cache solely because its directory exists.

The HRSID pool limit of 1,000 is not a promise of 1,000 training images plus extra
validation images: the chosen validation fraction must be deducted from the
pool unless that protocol is explicitly revised.

## Training and validation

Record training loss and validation loss for each epoch, including meaningful
components. Implement and test validation-loss calculation per framework: no
gradients, optimizer updates, BatchNorm-statistic updates or training augmentation
on validation data. Define loss reductions consistently within each model.
Raw loss magnitudes are not directly comparable across different architectures.

Compute validation mask AP50:95 on a consistent scale. Select the best checkpoint
and apply patience using that validation metric, not the test score. Record
maximum epochs, actual epochs, stopping reason, minimum improvement, batch size,
optimizer, schedule, precision, input size and pretrained initialization.

Use a shared maximum epoch budget and stopping rule, not a claim that all models
necessarily complete the same number of epochs. Framework-specific implementations
must demonstrate the same stopping semantics before final training.

## Persistence

Each run has a unique path:

`runs/<dataset>/<model>/<run_id>/`

Required outputs:

- `config.json`, environment/package versions and GPU identity.
- `data.json`, a reference to the persisted split manifest and its fingerprint.
- `history.csv` and atomic `history.json` containing epoch, train loss,
  validation loss, validation AP and loss components.
- `checkpoints/best.*` for validation-selected weights.
- `checkpoints/last.*` containing full resumption state, including optimizer,
  scheduler, epoch, random-number state and early-stopping state as applicable.
- `metrics/test.json` and separately scoped predictions.
- `metrics/ls_ssdd.json` where applicable.

Train from local runtime storage; durably synchronize history and checkpoints
to Drive during training. Report the same scalar values to a unique ClearML
task. Record sync failures and retry without silently dropping local history.
Recovery must distinguish 'resume unfinished training' from 'evaluate completed
training'; the presence of any checkpoint is not proof that training completed.

## Evaluation

Evaluate the selected checkpoint on test data only after selection is complete.
Choose confidence thresholds on validation data and freeze them for test
precision/recall/F1. Never tune the threshold on test labels while describing
the result as untouched-test performance.

Report segmentation AP50, AP50:95, precision, recall, F1 and small-object AP where
annotations support them. Keep LS-SSD box metrics distinct from mask metrics.
SAMSAR's ground-truth-prompted protocol must be labelled and not presented as an
equivalent fully automatic detection pipeline.

Measure FPS with a documented common hardware/timing protocol. Do not compare
FPS across unlike GPUs as if only model architecture changed.

Aggregate results by dataset, model and run ID, validate coverage and retain
source provenance. Missing results must stay missing, not become zero or a
copied row from a different dataset.
