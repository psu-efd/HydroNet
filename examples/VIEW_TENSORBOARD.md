# How to View TensorBoard Results

After training your ML model, TensorBoard logs are automatically saved. Here's how to view them:

## Step 1: Navigate to the Project Directory

Open a terminal/PowerShell and navigate to the example directory, .e.g., for the Sacramento River steady example:
```bash
cd "path\to\HydroNet\examples\PI_DeepONet\SacramentoRiver_steady"
```

## Step 2: Launch TensorBoard

You have two options:

### Option A: View the Latest Training Run (Recommended)
The training script creates timestamped log directories in `logs/tensorboard/`. To view the most recent run:

```bash
tensorboard --logdir=logs
```

This will show all runs in the `logs/` directory, allowing you to compare different training sessions.

### Option B: View a Specific Training Run
If you want to view a specific run, specify the exact directory:

```bash
tensorboard --logdir=logs/swe_deeponet_20251114-165625
```

Replace `swe_deeponet_20251114-165625` with your actual log directory name.

## Step 3: Open TensorBoard in Browser

After running the command, TensorBoard will start and display a URL like:
```
http://localhost:6006
```

Open this URL in your web browser to view:
- **Loss/train**: Training loss over epochs
- **Loss/val**: Validation loss over epochs

## Step 4: Viewing Multiple Runs

If you have multiple training runs, TensorBoard will show them all in the left sidebar. You can:
- Toggle runs on/off to compare them
- Use the slider to adjust smoothing
- Click on specific data points to see exact values

## Troubleshooting

### If TensorBoard is not installed:
```bash
pip install tensorboard
```

### If port 6006 is already in use:
```bash
tensorboard --logdir=logs --port=6007
```

### To view TensorBoard on a remote server:
If you're running on a remote server, you may need to use port forwarding:
```bash
ssh -L 6006:localhost:6006 user@remote-server
```

Then access TensorBoard at `http://localhost:6006` on your local machine.

## Quick Reference

- **Log directory location**: `logs/swe_deeponet_YYYYMMDD-HHMMSS/`
- **Default TensorBoard port**: 6006
- **Metrics logged**: Training loss and validation loss per epoch

