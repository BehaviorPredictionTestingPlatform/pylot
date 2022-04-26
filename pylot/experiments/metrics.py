import absl.app
import csv
import erdos
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pylot.prediction.flags import flags
from pylot.prediction.from_csv_operator import FromCsvOperator
from pylot.prediction.linear_predictor_operator import LinearPredictorOperator
# from pylot.prediction.r2p2_predictor_operator import R2P2PredictorOperator
from pylot.prediction.to_csv_operator import ToCsvOperator

class Simulation():
    def __init__(self):
        self.worker_num = 0
        self.trajectory = (
                            ((0.0, 0.0, 0), (0.0, 5.0, 0)),
                            ((0.0, 0.1, 0), (0.0, 5.1, 0)),
                            ((0.0, 0.2, 0), (0.0, 5.2, 0)),
                            ((0.0, 0.3, 0), (0.0, 5.3, 0)),
                            ((0.0, 0.4, 0), (0.0, 5.4, 0)),
                            ((0.0, 0.5, 0), (0.0, 5.5, 0)),
                            ((0.0, 0.6, 0), (0.0, 5.6, 0)),
                            ((0.0, 0.7, 0), (0.0, 5.7, 0)),
                            ((0.0, 0.8, 0), (0.0, 5.8, 0)),
                            ((0.0, 0.9, 0), (0.0, 5.9, 0)),
                            ((0.0, 1.0, 0), (0.0, 6.0, 0)),
                            ((0.0, 1.1, 0), (0.0, 6.1, 0)),
                            ((0.0, 1.2, 0), (0.0, 6.2, 0)),
                            ((0.0, 1.3, 0), (0.0, 6.3, 0)),
                            ((0.0, 1.4, 0), (0.0, 6.4, 0)),
                            ((0.0, 1.5, 0), (0.0, 6.5, 0)),
                            ((0.0, 1.6, 0), (0.0, 6.6, 0)),
                            ((0.0, 1.7, 0), (0.0, 6.7, 0)),
                            ((0.0, 1.8, 0), (0.0, 6.8, 0)),
                            ((0.0, 1.9, 0), (0.0, 6.9, 0)),
                            ((0.0, 2.0, 0), (0.0, 7.0, 0)),
                            ((0.0, 2.1, 0), (0.0, 7.1, 0)),
                            ((0.0, 2.2, 0), (0.0, 7.2, 0)),
                            ((0.0, 2.3, 0), (0.0, 7.3, 0)),
                            ((0.0, 2.4, 0), (0.0, 7.4, 0)),
                            ((0.0, 2.5, 0), (0.0, 7.5, 0)),
                            ((0.0, 2.6, 0), (0.0, 7.6, 0)),
                            ((0.0, 2.7, 0), (0.0, 7.7, 0)),
                            ((0.0, 2.8, 0), (0.0, 7.8, 0)),
                            ((0.0, 2.9, 0), (0.0, 7.9, 0)),
                            ((0.0, 3.0, 0), (0.0, 8.0, 0)),
                            ((0.0, 3.1, 0), (0.0, 8.1, 0)),
                            ((0.0, 3.2, 0), (0.0, 8.2, 0)),
                            ((0.0, 3.3, 0), (0.0, 8.3, 0)),
                            ((0.0, 3.4, 0), (0.0, 8.4, 0)),
                            ((0.0, 3.5, 0), (0.0, 8.5, 0)),
                            ((0.0, 3.6, 0), (0.0, 8.6, 0)),
                            ((0.0, 3.7, 0), (0.0, 8.7, 0)),
                            ((0.0, 3.8, 0), (0.0, 8.8, 0)),
                            ((0.0, 3.9, 0), (0.0, 8.9, 0))
                          )

def main(argv):
    simulation = Simulation()

    in_dir = '/Users/francisindaheng/Desktop/Francis/Academics/Current_Classes/EECS_219C/proj/pylot/pylot/experiments/input'
    out_dir = '/Users/francisindaheng/Desktop/Francis/Academics/Current_Classes/EECS_219C/proj/pylot/pylot/experiments/output'
    threshADE = 0.5
    threshFDE = 1.0
    timepoint = 25
    past_steps = 20
    future_steps = 15
    parallel = False
    debug = False

    assert timepoint >= past_steps, 'Timepoint must be at least the number of past steps!'
    assert past_steps >= future_steps, 'Must track at least as many steps as we predict!'

    flags.FLAGS.__delattr__('prediction_num_past_steps')
    flags.FLAGS.__delattr__('prediction_num_future_steps')
    flags.DEFINE_integer('prediction_num_past_steps', past_steps, '')
    flags.DEFINE_integer('prediction_num_future_steps', future_steps, '')

    worker_num = simulation.worker_num if parallel else 0
    traj = simulation.trajectory
    past_trajs = traj[timepoint-past_steps:timepoint]
    gt_trajs = traj[timepoint:timepoint+future_steps]
    gt_len = len(gt_trajs)

    # Dictionary mapping agent IDs to ground truth trajectories
    gts = {}
    for gt_traj in gt_trajs:
        for agent_id, transform in enumerate(gt_traj):
            if agent_id not in gts:
                gts[agent_id] = []
            gts[agent_id].append(transform)
    for agent_id, gt in gts.items():
        gts[agent_id] = np.array(gt)

    if debug:
        print(f'ADE Threshold: {threshADE}, FDE Threshold: {threshFDE}')
        plt.plot([gt[-1][0] for gt in traj], [gt[-1][1] for gt in traj], color='black')
        plt.plot([gt[-1][0] for gt in past_trajs], [gt[-1][1] for gt in past_trajs], color='blue')
        plt.plot([gt[-1][0] for gt in gt_trajs], [gt[-1][1] for gt in gt_trajs], color='yellow')

    # Write past trajectories to CSV file
    input_csv_path = f'{in_dir}/past_{worker_num}.csv'
    csv_file = open(input_csv_path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['timestamp', 'agent_id', 'x', 'y', 'yaw'])
    for timestamp in range(timepoint-past_steps, timepoint):
        for agent_id, transform in enumerate(traj[timestamp]):
            x, y, yaw = transform
            writer.writerow([timestamp, agent_id, x, y, yaw])
    csv_file.close()

    # Run behavior prediction model
    [tracking_stream] = erdos.connect(FromCsvOperator, erdos.OperatorConfig(name='from_csv_operator'), [], input_csv_path)
    [prediction_stream] = erdos.connect(LinearPredictorOperator, erdos.OperatorConfig(name='linear_predictor_operator'), [tracking_stream], flags.FLAGS)
    # [prediction_stream] = erdos.connect(R2P2PredictorOperator, erdos.OperatorConfig(name='r2p2_predictor_operator'), [erdos.ReadStream(), tracking_stream], flags.FLAGS, None)
    erdos.connect(ToCsvOperator, erdos.OperatorConfig(name='to_csv_operator'), [prediction_stream], out_dir, worker_num)
    erdos.run()

    # Extract predicted trajectories from CSV file
    output_csv_path = f'{out_dir}/pred_{worker_num}.csv'
    pred_trajs = np.genfromtxt(output_csv_path, delimiter=',', skip_header=1)
    pred_len = pred_trajs.shape[0]
    if gt_len < pred_len:
        pred_trajs = pred_trajs[:gt_len]

    # Sort by timestamp
    pred_trajs = pred_trajs[pred_trajs[:, 0].argsort()]

    # Dictionary mapping agent IDs to predicted trajectories
    preds = {}
    for pred_traj in pred_trajs:
        _, agent_id, x, y, yaw = pred_traj
        if agent_id not in preds:
            preds[agent_id] = []
        preds[agent_id].append((x, y, yaw))

    # Dictionary mapping agent IDs to ADEs/FDEs
    ADEs, FDEs = {}, {}
    for agent_id, pred in preds.items():
        pred = np.array(pred)
        gt = gts[agent_id]
        ADEs[agent_id] = float(
            sum(
                math.sqrt(
                    (pred[i, 0] - gt[i, 0]) ** 2
                    + (pred[i, 1] - gt[i, 1]) ** 2
                )
                for i in range(min(pred_len, gt_len))
            ) / pred_len
        )
        FDEs[agent_id] = math.sqrt(
            (pred[-1, 0] - gt[-1, 0]) ** 2
            + (pred[-1, 1] - gt[-1, 1]) ** 2
        )

    if debug:
        print(f'ADE: {ADE}, FDE: {FDE}')
        p = pd.read_csv(output_csv_path)
        plt.plot(p['X'], p['Y'], color='green')

    minADE, minFDE = min(ADEs.values()), min(FDEs.values())
    print(f'minADE: {minADE}, minFDE: {minFDE}')
    rho = (threshADE - minADE, threshFDE - minFDE)

    if debug:
        plt.show()

if __name__ == '__main__':
    absl.app.run(main)
