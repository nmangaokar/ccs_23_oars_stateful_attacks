from IPython import embed
from abc import abstractmethod
import torch
import math
from tqdm.auto import tqdm
from IPython import embed
from attacks.Attack import Attack
import numpy as np


class NESScore(Attack):
    def __init__(self, model, model_config, attack_config):
        super().__init__(model, model_config, attack_config)

    def attack_untargeted(self, x, y):
        # original image loss
        probs_orig, is_cache = self.model(x)
        loss_orig = self.loss(probs_orig, y)

        # initialize
        x_adv = x.detach()
        x_adv = x_adv + torch.FloatTensor(*x.shape).uniform_(-self.attack_config["eps"],
                                                             self.attack_config["eps"]).cuda()

        # variables and bookkeeping
        if self.attack_config["adaptive"]["bs_grad_var"]:
            var = self.binary_search_gradient_estimation_variance(x)
        else:
            var = self.attack_config["var"]
        if self.attack_config["adaptive"]["bs_min_ss"]:
            bs_min_step_size = self.binary_search_minimum_step_size(x)
            # bs_min_step_size = self.interval_search_minimum_step_size(x)
        else:
            bs_min_step_size = 0
        step_size = self.attack_config["step_size"]
        prev_grad_est = None
        prev_x_adv = None
        loss_history = []
        step_attempts = 0

        # attack loop
        pbar = tqdm(range(self.attack_config["max_iter"]), colour="red", leave=True)
        for _ in pbar:
            # estimate gradient
            avg_loss, grad_est = self.estimate_gradient(x_adv, y, self.attack_config["num_dirs"], var)

            # gradient momentum
            if step_attempts == 0:
                if prev_grad_est is not None:
                    grad_est = self.attack_config["momentum"] * prev_grad_est + (
                            1 - self.attack_config["momentum"]) * grad_est
            prev_grad_est = grad_est
            prev_x_adv = x_adv

            # anneal step size
            loss_history.append(avg_loss)
            loss_history = loss_history[-self.attack_config["plateau_length"]:]
            if loss_history[-1] > loss_history[0] and \
                    len(loss_history) == self.attack_config["plateau_length"]:
                if step_size > self.attack_config["min_step_size"]:
                    step_size = max(step_size / self.attack_config["plateau_drop"], self.attack_config["min_step_size"])
                loss_history = []

            if self.attack_config["adaptive"]["bs_min_ss"]:
                step_size = max(step_size, bs_min_step_size)

            # step
            x_adv = x_adv + step_size * grad_est.sign()
            eta = torch.clamp(x_adv - x, min=-self.attack_config["eps"], max=self.attack_config["eps"])
            x_adv = torch.clamp(x + eta, min=0, max=1).detach_()
            step_attempts += 1
            curr_probs, is_cache = self.model(x_adv)
            if is_cache[0] and step_attempts < self.attack_config["adaptive"]["step_max_attempts"]:
                print("Step movement failure. Retrying.", torch.sum(grad_est))
                x_adv = prev_x_adv
                continue
            elif is_cache[0] and step_attempts >= self.attack_config["adaptive"]["step_max_attempts"]:
                self.end("Step movement failure.")
            step_attempts = 0
            curr_label = torch.argmax(curr_probs, dim=1)
            curr_loss = self.loss(curr_probs, y)

            # logging
            pbar.set_description(
                f"Label : {curr_label.item()}/{y.item()}| Loss : {loss_orig:.8f}/{curr_loss:.8f} | Cache Hits : {self.get_cache_hits()}/{self.get_total_queries()} | Step Size : {step_size:.6f}")
            if curr_label.item() != y.item():
                break
        return x_adv

    def attack_targeted(self, x, y, x_adv):
        # original image loss
        probs_orig, is_cache = self.model(x)
        loss_orig = self.loss(probs_orig, y)

        # initialize
        x_adv = x.detach()
        x_adv = x_adv + torch.FloatTensor(*x.shape).uniform_(-self.attack_config["eps"],
                                                             self.attack_config["eps"]).cuda()

        # variables and bookkeeping
        if self.attack_config["adaptive"]["bs_grad_var"]:
            var = self.binary_search_gradient_estimation_variance(x)
        else:
            var = self.attack_config["var"]
        if self.attack_config["adaptive"]["bs_min_ss"]:
            bs_min_step_size = self.binary_search_minimum_step_size(x)
            step_query_interval = bs_min_step_size / self.attack_config["min_step_size"]
            step_query_interval *= self.attack_config["adaptive"]["bs_min_ss_hit_rate"]
            step_query_interval = math.ceil(step_query_interval)
        else:
            step_query_interval = 1
        step_size = self.attack_config["step_size"]
        prev_grad_est = None
        loss_history = []
        step_attempts = 0

        # attack loop
        pbar = tqdm(range(self.attack_config["max_iter"]), colour="red", leave=True)
        for _ in pbar:
            # estimate gradient
            avg_loss, grad_est = self.estimate_gradient(x_adv, y, self.attack_config["num_dirs"], var)

            # gradient momentum
            if step_attempts == 0:
                if prev_grad_est is not None:
                    grad_est = self.attack_config["momentum"] * prev_grad_est + (
                            1 - self.attack_config["momentum"]) * grad_est
            prev_grad_est = grad_est
            prev_x_adv = x_adv

            # anneal step size
            loss_history.append(avg_loss)
            loss_history = loss_history[-self.attack_config["plateau_length"]:]
            if loss_history[-1] > loss_history[0] and \
                    len(loss_history) == self.attack_config["plateau_length"]:
                if step_size > self.attack_config["min_step_size"]:
                    step_size = max(step_size / self.attack_config["plateau_drop"], self.attack_config["min_step_size"])
                loss_history = []

            # step
            x_adv = x_adv - step_size * grad_est.sign()
            eta = torch.clamp(x_adv - x, min=-self.attack_config["eps"], max=self.attack_config["eps"])
            x_adv = torch.clamp(x + eta, min=0, max=1).detach_()

            if _ % step_query_interval == 0:
                step_attempts += 1
                curr_probs, is_cache = self.model(x_adv)
                if is_cache[0] and step_attempts < self.attack_config["adaptive"]["step_max_attempts"]:
                    print("Step movement failure.")
                    x_adv = prev_x_adv
                    continue
                elif is_cache[0] and step_attempts >= self.attack_config["adaptive"]["step_max_attempts"]:
                    self.end("Step movement failure.")
                step_attempts = 0
                curr_label = torch.argmax(curr_probs, dim=1)
                curr_loss = self.loss(curr_probs, y)

                # logging
                pbar.set_description(
                    f"Label : {y.item()}/{curr_label.item()}| Loss : {loss_orig:.8f}/{curr_loss:.8f} "
                    f"| Cache Hits : {self.get_cache_hits()}/{self.get_total_queries()} "
                    f"| Var : {var:.6f} | Step Size : {step_size:.6f} | Step Query Interval : {step_query_interval}"
                    f"| Step Attempts : {step_attempts}")
                if curr_label.item() == y.item():
                    break
        return x_adv

    def loss(self, probs, y):
        loss = torch.nn.functional.nll_loss(torch.log(probs), y)
        return loss

    def binary_search_minimum_step_size(self, x):
        lower = self.attack_config["adaptive"]["bs_min_ss_lower"]
        upper = self.attack_config["adaptive"]["bs_min_ss_upper"]
        ss = upper
        for _ in range(self.attack_config["adaptive"]["bs_min_ss_steps"]):
            mid = (lower + upper) / 2
            cache_hits = 0
            for _ in range(self.attack_config["adaptive"]["bs_min_ss_sample_size"]):
                step = torch.where(torch.rand(*x.shape).to(x.device) < 0.5, -1,
                                   1)  # 2 * torch.rand(*x.shape).to(x.device) - 1 #
                noisy_img = x + mid * step
                noisy_img = torch.clamp(noisy_img, min=0, max=1)
                probs, is_cache = self.model(noisy_img)
                if is_cache[0]:
                    cache_hits += 1
            if cache_hits / self.attack_config["adaptive"]["bs_min_ss_sample_size"] \
                    <= self.attack_config["adaptive"]["bs_min_ss_hit_rate"]:
                ss = mid
                upper = mid
            else:
                lower = mid
                # upper = upper * 1.5
            print(
                f"Step Size : {ss:.6f} | Cache Hits : {cache_hits}/{self.attack_config['adaptive']['bs_min_ss_sample_size']}, upper : {upper:.6f}, lower : {lower:.6f}")
        return ss

    def smooth_list(self, lst):
        smoothed = []
        for i in range(len(lst)):
            if i == 0:  # First element
                smoothed.append((lst[i] + lst[i + 1]) / 2)
            elif i == len(lst) - 1:  # Last element
                smoothed.append((lst[i - 1] + lst[i]) / 2)
            else:
                smoothed.append((lst[i - 1] + lst[i] + lst[i + 1]) / 3)
        return smoothed

    def binary_search_gradient_estimation_variance(self, x):
        lower = self.attack_config["adaptive"]["bs_grad_var_lower"]
        upper = self.attack_config["adaptive"]["bs_grad_var_upper"]
        var = upper
        for _ in range(self.attack_config["adaptive"]["bs_grad_var_steps"]):
            mid = (lower + upper) / 2
            cache_hits = 0
            for _ in range(self.attack_config["adaptive"]["bs_grad_var_sample_size"]):
                noise = torch.randn_like(x).to(x.device)
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

    def estimate_gradient(self, x, y, num_dirs, var):
        grad_est = torch.zeros_like(x)
        losses = []
        num_dirs_goal = num_dirs
        for _ in range(self.attack_config["adaptive"]["grad_max_attempts"]):
            for _ in range(int(num_dirs / 2)):
                dir = torch.randn_like(x) * var
                x_pert = x + dir
                probs, is_cache = self.model(x_pert)
                if is_cache[0]:
                    continue
                neg_dir = -dir
                x_pert = x + neg_dir
                neg_probs, neg_is_cache = self.model(x_pert)
                if neg_is_cache[0]:
                    continue
                loss = self.loss(probs, y)
                neg_loss = self.loss(neg_probs, y)
                losses.append(loss)
                losses.append(neg_loss)
                grad_est += loss * dir + neg_loss * neg_dir
            num_dirs = num_dirs_goal - len(losses)
        if len(losses) != num_dirs_goal and not self.attack_config["adaptive"]["grad_est_accept_partial"]:
            self.end("Failure in gradient estimation, not enough directions.")
        if len(losses) == 0:
            self.end("Failure in gradient estimation, literally zero directions.")
        grad_est /= len(losses)
        return torch.stack(losses).mean(), grad_est
