import numpy as np


def run_configuration(algorithm, hparams, data):
    if hparams.show_training:
        print('Running configuration:')
        for param_key, param_value in hparams.values().items():
            print('\t', param_key, '=', param_value)
        print()

    data.positive_reward = hparams.positive_reward
    data.negative_reward = hparams.negative_reward

    sampling = algorithm(hparams, data)
    last_update = 0
    last_mean_cost = float('inf')
    stop = False
    batch_sum_cost = 0
    for step in range(hparams.num_steps):
        if step >= hparams.positive_start:
            action_i = sampling.action()
        else:
            action_i = step
            opt_r = sampling.data.rewards[action_i]
            sampling.h_data.add(action_i, opt_r, opt_r)
        sampling.update(action_i)

        if hparams.early_stopping:
            if last_mean_cost > sampling.h_data.mean_cost:
                last_mean_cost = sampling.h_data.mean_cost
                last_update = 0
            elif last_update < hparams.patience:
                last_update += 1
            else:
                stop = True

        cost = sampling.h_data.cost(-1)
        batch_sum_cost += cost

        if stop or ((step + 1) % hparams.freq_summary == 0):
            if hparams.show_training:
                print(
                    '{} | {:4} | pred_r: {:13} | opt_r: {:3} | cost: {:13} | bmean_cost: {:17} | precision: {:6} | recall: {:6} | pos: {:4}'.format(
                        hparams.name,
                        step + 1,
                        round(sampling.h_data.pred_rewards[-1], 10),
                        int(sampling.h_data.opt_rewards[-1]),
                        round(cost, 10),
                        round(batch_sum_cost / hparams.freq_summary, 10),
                        round(sampling.h_data.precision[-1] * 100, 3),
                        round(sampling.h_data.recall[-1] * 100, 3),
                        sampling.h_data.tp + sampling.h_data.fn
                    ))
            batch_sum_cost = 0

        if stop:
            if hparams.show_training:
                print('Early stopping | {:4} | last_mean_cost: {}'.format(step + 1, last_mean_cost))
            break

    if hparams.show_training:
        costs = sampling.h_data.cost()
        print('Cost stats:')
        print('Max:', np.max(costs))
        print('Min:', np.min(costs))
        print('Mean:', np.mean(costs))
        print('Median:', np.median(costs))
        print('\n\n')

    return sampling.h_data
