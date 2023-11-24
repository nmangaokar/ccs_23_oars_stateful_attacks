from IPython import embed
from abc import abstractmethod
import torch
from tqdm.auto import tqdm
from IPython import embed
from torchvision import transforms
import torchvision
import random
import numpy as np
from attacks.Attack import Attack
import torch_dct as dct
import torch.nn.functional as F


class QEBA(Attack):
    def __init__(self, model, model_config, attack_config):
        super().__init__(model, model_config, attack_config)

    def phi(self, x, y, targeted):
        x = torch.clamp(x, 0, 1)
        logits, is_cache = [], []
        for x_i in x:
            logits_i, is_cache_i = self.model(x_i.unsqueeze(0))
            logits.append(logits_i.cpu())
            is_cache.extend(is_cache_i)
        logits = torch.cat(logits, dim=0)
        if targeted:
            return (logits.argmax(dim=1) == y).float(), is_cache
        else:
            return (logits.argmax(dim=1) != y).float(), is_cache

    def binary_search_to_boundary(self, x, y, x_adv, threshold, targeted):
        alpha_low = 0
        alpha_high = 1
        while alpha_high - alpha_low > threshold:
            alpha_middle = (alpha_low + alpha_high) / 2
            interpolated = (1 - alpha_middle) * x_adv + alpha_middle * x
            decision, is_cache = self.phi(interpolated, y, targeted)
            if is_cache[0] and not self.attack_config["adaptive"]["bs_boundary_end_on_hit"]:
                break
            elif is_cache[0] and self.attack_config["adaptive"]["bs_boundary_end_on_hit"]:
                self.end("Boundary search failure.")
            if decision == 0:
                alpha_high = alpha_middle
            else:
                alpha_low = alpha_middle
        interpolated = (1 - alpha_low) * x_adv + alpha_low * x
        return interpolated

    def binary_search_gradient_estimation_variance(self, x):
        lower = self.attack_config["adaptive"]["bs_grad_var_lower"]
        upper = self.attack_config["adaptive"]["bs_grad_var_upper"]
        var = upper
        for _ in range(self.attack_config["adaptive"]["bs_grad_var_steps"]):
            mid = (lower + upper) / 2
            cache_hits = 0
            for _ in range(self.attack_config["adaptive"]["bs_grad_var_sample_size"]):
                noise = torch.randn_like(x).to(x.device)
                noise = noise / torch.norm(noise)
                noise = noise * mid
                noisy_img = x + noise
                noisy_img = torch.clamp(noisy_img, min=0, max=1)
                probs, is_cache = self.model(noisy_img)
                if is_cache[0]:
                    cache_hits += 1
            if cache_hits / self.attack_config["adaptive"]["bs_grad_var_sample_size"] \
                    <= self.attack_config["adaptive"]["bs_grad_var_hit_rate"]:
                var = mid
                upper = mid
            else:
                lower = mid
            print(
                f"Var : {var:.6f} | Cache Hits : {cache_hits}/{self.attack_config['adaptive']['bs_grad_var_sample_size']}")
        return var

    def attack_untargeted(self, x, y):
        dim = torch.prod(torch.tensor(x.shape[1:]))
        theta = 1 / (torch.sqrt(dim) * dim)

        # initialize
        x_adv = torch.rand_like(x)
        while self.phi(x_adv, y, targeted=False)[0] == 0:
            x_adv = torch.rand_like(x)
        x_adv = self.binary_search_to_boundary(x, y, x_adv, 0.001, targeted=False)
        x_adv_prev = None
        step_attempts = 0
        rollback = False

        if self.attack_config["adaptive"]["bs_grad_var"]:
            delta = self.binary_search_gradient_estimation_variance(x)

        # attack
        pbar = tqdm(range(self.attack_config["max_iter"]))
        for t in pbar:
            # 1. compute new delta
            if not self.attack_config["adaptive"]["bs_grad_var"]:
                if t == 0:
                    delta = 0.1
                else:
                    delta = torch.sqrt(dim) * theta * torch.linalg.norm(x_adv_prev - x)

            # 2. compute number of directions
            num_dirs_goal = min(int(self.attack_config["num_dirs"] * np.sqrt(t + 1)),
                                self.attack_config["max_num_dirs"])
            num_dirs_ = num_dirs_goal

            # 3. estimate gradient
            fval_obtained = torch.zeros(0, 1, 1, 1).to(x.device)
            dirs_obtained = x_adv.repeat(0, 1, 1, 1).to(x.device)
            for _ in range(self.attack_config["adaptive"]["grad_max_attempts"]):
                dirs_low_dim = torch.randn(num_dirs_,
                                           x_adv.shape[1],
                                           int(x_adv.shape[2] / self.attack_config["dim_reduction_factor"]),
                                           int(x_adv.shape[3] / self.attack_config["dim_reduction_factor"]))
                dirs_low_dim = torch.linalg.qr(dirs_low_dim, mode='complete')[0]
                dirs = torch.zeros_like(x_adv.repeat(num_dirs_, 1, 1, 1))
                dirs[:, :, :int(x_adv.shape[2] / self.attack_config["dim_reduction_factor"]), :int(
                    x_adv.shape[3] / self.attack_config["dim_reduction_factor"])] = dirs_low_dim
                dirs = dct.idct_2d(dirs)
                dirs = dirs / torch.linalg.norm(torch.flatten(dirs, start_dim=1), dim=1).reshape(-1, 1, 1, 1)
                perturbed = x_adv.repeat(num_dirs_, 1, 1, 1) + delta * dirs
                perturbed = torch.clamp(perturbed, 0, 1)
                dirs = (perturbed - x_adv.repeat(num_dirs_, 1, 1, 1)) / delta
                decision, is_cache = self.phi(perturbed, y, targeted=False)
                fval = 2 * decision.reshape(num_dirs_, 1, 1, 1) - 1

                dirs = dirs[~np.array(is_cache)]
                fval = fval[~np.array(is_cache)]
                dirs_obtained = torch.cat((dirs_obtained, dirs), dim=0)
                fval_obtained = torch.cat((fval_obtained, fval), dim=0)

                if len(dirs_obtained) == num_dirs_goal:
                    break
                else:
                    num_dirs_ = num_dirs_goal - len(dirs_obtained)
            dirs = dirs_obtained
            fval = fval_obtained
            if len(dirs) != num_dirs_goal and not self.attack_config["adaptive"]["grad_est_accept_partial"]:
                self.end("Gradient estimation failure.")
            if len(dirs) == 0:
                self.end("Gradient estimation failure. Literally zero directions.")

            if torch.mean(fval) == 1:
                grad = torch.mean(dirs, dim=0)
            elif torch.mean(fval) == -1:
                grad = -torch.mean(dirs, dim=0)
            else:
                fval -= torch.mean(fval)
                grad = torch.mean(fval * dirs, dim=0)
            grad = grad / torch.linalg.norm(grad)

            # 4. step size search
            step_attempts += 1
            eta = torch.linalg.norm(x_adv - x) / np.sqrt(t + 1)
            while True:
                decision, is_cache = self.phi(x_adv + eta * grad, y, targeted=False)
                if is_cache[0] and step_attempts < self.attack_config["adaptive"]["step_max_attempts"]:
                    print("step cache hit")
                    rollback = True
                    break
                elif is_cache[0] and step_attempts >= self.attack_config["adaptive"]["step_max_attempts"]:
                    self.end("Step movement failure.")
                if decision == 1:
                    rollback = False
                    break
                eta /= 2
            if rollback:
                continue
            step_attempts = 0

            # 5. update
            x_adv = torch.clamp(x_adv + eta * grad, 0, 1)
            x_adv_prev = x_adv.clone()

            # 6. binary search to return to the boundary
            x_adv = self.binary_search_to_boundary(x, y, x_adv, threshold=theta, targeted=False)

            # 7. check budget and log progress
            norm_dist = torch.linalg.norm(x_adv - x) / (x.shape[-1] * x.shape[-2] * x.shape[-3]) ** 0.5
            pbar.set_description(
                f"Iter {t} | L2_normalized={norm_dist:.4f} | Cache Hits : {self.get_cache_hits()}/{self.get_total_queries()} | delta={delta:.4f}")
            if norm_dist <= self.attack_config["eps"]:
                return x_adv
        return x

    def attack_targeted(self, x, y, x_adv):
        x = x.cpu()
        y = y.cpu()
        x_adv = x_adv.cpu()
        dim = torch.prod(torch.tensor(x.shape[1:]))
        theta = 1 / (torch.sqrt(dim) * dim)

        # initialize
        x_adv = self.binary_search_to_boundary(x, y, x_adv, 0.001, targeted=True)
        x_adv_prev = None
        step_attempts = 0
        rollback = False

        if self.attack_config["adaptive"]["bs_grad_var"]:
            delta = self.binary_search_gradient_estimation_variance(x)

        # attack
        pbar = tqdm(range(self.attack_config["max_iter"]))
        for t in pbar:
            # 1. compute new delta
            if not self.attack_config["adaptive"]["bs_grad_var"]:
                if t == 0 or x_adv_prev is None:
                    delta = 0.1
                else:
                    delta = torch.sqrt(dim) * theta * torch.linalg.norm(x_adv_prev - x)

            # 2. compute number of directions
            num_dirs_goal = min(int(self.attack_config["num_dirs"] * np.sqrt(t + 1)),
                                self.attack_config["max_num_dirs"])
            num_dirs_ = num_dirs_goal

            # 3. estimate gradient
            fval_obtained = torch.zeros(0, 1, 1, 1).to(x.device)
            dirs_obtained = x_adv.repeat(0, 1, 1, 1).to(x.device)
            for _ in range(self.attack_config["adaptive"]["grad_max_attempts"]):
                dirs_low_dim = torch.randn(num_dirs_,
                                           x_adv.shape[1],
                                           int(x_adv.shape[2] / self.attack_config["dim_reduction_factor"]),
                                           int(x_adv.shape[3] / self.attack_config["dim_reduction_factor"]))
                dirs = F.interpolate(dirs_low_dim, size=(x_adv.shape[2], x_adv.shape[3]), mode='bilinear',
                                     align_corners=False).to(x.device)
                dirs = dirs / torch.linalg.norm(torch.flatten(dirs, start_dim=1), dim=1).reshape(-1, 1, 1, 1)
                perturbed = x_adv.repeat(num_dirs_, 1, 1, 1) + delta * dirs
                perturbed = torch.clamp(perturbed, 0, 1)
                dirs = (perturbed - x_adv.repeat(num_dirs_, 1, 1, 1)) / delta
                decision, is_cache = self.phi(perturbed, y, targeted=True)
                fval = 2 * decision.reshape(num_dirs_, 1, 1, 1) - 1

                dirs = dirs[~np.array(is_cache)]
                fval = fval[~np.array(is_cache)]
                dirs_obtained = torch.cat((dirs_obtained, dirs), dim=0)
                fval_obtained = torch.cat((fval_obtained, fval), dim=0)

                if len(dirs_obtained) == num_dirs_goal:
                    break
                else:
                    num_dirs_ = num_dirs_goal - len(dirs_obtained)
                    print(f"Got {len(dirs_obtained)} directions. Trying {num_dirs_} more until {num_dirs_goal}.")
            dirs = dirs_obtained
            fval = fval_obtained
            if len(dirs) != num_dirs_goal and not self.attack_config["adaptive"]["grad_est_accept_partial"]:
                self.end("Gradient estimation failure.")
            if len(dirs) == 0:
                self.end("Gradient estimation failure. Literally zero directions.")

            if torch.mean(fval) == 1:
                grad = torch.mean(dirs, dim=0)
            elif torch.mean(fval) == -1:
                grad = -torch.mean(dirs, dim=0)
            else:
                fval -= torch.mean(fval)
                grad = torch.mean(fval * dirs, dim=0)
            grad = grad / torch.linalg.norm(grad)

            # 4. step size search
            step_attempts += 1
            eta = torch.linalg.norm(x_adv - x) / np.sqrt(t + 1)
            while True:
                decision, is_cache = self.phi(x_adv + eta * grad, y, targeted=True)
                if is_cache[0] and step_attempts < self.attack_config["adaptive"]["step_max_attempts"]:
                    print("step cache hit")
                    rollback = True
                    break
                elif is_cache[0] and step_attempts >= self.attack_config["adaptive"]["step_max_attempts"]:
                    self.end("Step movement failure.")
                if decision == 1:
                    rollback = False
                    break
                eta /= 2
            if rollback:
                continue
            step_attempts = 0

            # 5. update
            x_adv = torch.clamp(x_adv + eta * grad, 0, 1)
            x_adv_prev = x_adv.clone()

            # 6. binary search to return to the boundary
            x_adv = self.binary_search_to_boundary(x, y, x_adv, threshold=theta, targeted=True)

            # 7. check budget and log progress
            norm_dist = torch.linalg.norm(x_adv - x) / (x.shape[-1] * x.shape[-2] * x.shape[-3]) ** 0.5
            pbar.set_description(
                f"Iter {t} | L2_normalized={norm_dist:.4f} | Cache Hits : {self.get_cache_hits()}/{self.get_total_queries()} | delta={delta:.4f}")
            if norm_dist <= self.attack_config["eps"]:
                return x_adv
        return x
