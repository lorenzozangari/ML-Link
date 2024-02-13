import torch


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            if result.dim() > 1:
                argmax = result[:, 0].argmax().item()
                return result[argmax, 0], result[argmax, 1]
            else:
                return result.max()
        else:
            if len(self.results[0]) == 0:
                result = 100 * torch.tensor(self.results[1:])
            else:
                result = 100 * torch.tensor(self.results)
            best_results = []
            for r in result:
                if r.dim() > 1:
                    valid = r[:, 0].max().item()
                    test = r[r[:, 0].argmax(), 1].item()
                    best_results.append((valid, test))
                else:
                    best_results.append(r.max().item())
            best_result = torch.tensor(best_results)
            if best_result.dim() > 1:
                r_valid = best_result[:, 0]
                r_test = best_result[:, 1]
                v_mean, v_std = r_valid.mean(), r_valid.std()
                t_mean, t_std = r_test.mean(), r_test.std()

                return v_mean, v_std, t_mean, t_std
            else:
                r_mean, r_std = best_result.mean(), best_result.std()
                return r_mean, r_std
