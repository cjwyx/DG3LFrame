# Usage Example

## Step 1: Clone the Repositories

First, clone the BasicTS and D3L code repositories:

```bash
git clone https://github.com/GestaltCogTeam/BasicTS.git
git clone https://github.com/cjwyx/DG3LFrame.git
```

## Step 2: Set Up the Python Environment

Navigate to the `DG3LFrame` directory and install the required Python packages:

```bash
cd DG3LFrame
pip install -r requirements.txt
cd ..
```

## Step 3: Integrate DG3LFrame into BasicTS

Move the `DG3LFrame` code into the `baselines` folder of BasicTS:

```bash
mv DG3LFrame ./BasicTS/baselines/
```

## Step 4: Run Experiments

Follow the BasicTS tutorial to run experiments. For example, to run the METR-LA experiment:

```bash
python experiments/train.py -c baselines/DG3LFrame/METR-LA.py --gpus 0
```

This will start the training process using the specified configuration file and GPU.