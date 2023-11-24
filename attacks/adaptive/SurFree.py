from IPython import embed
from abc import abstractmethod
import torch
from tqdm.auto import tqdm
from torchvision import transforms
import torchvision
import random
import numpy as np
from torch_dct import dct_2d, idct_2d
from attacks.Attack import Attack


class SurFree(Attack):
    def __init__(self, model, model_config, attack_config):
        super().__init__(model, model_config, attack_config)

    def is_adversarial(self, x, y, targeted):
        logits, is_cache = self.model(x)
        if targeted:
            return (logits.argmax(dim=1) == y).float(), is_cache
        else:
            return (logits.argmax(dim=1) != y).float(), is_cache

    def binary_search_to_boundary(self, x, y, x_adv, targeted):
        dim = torch.prod(torch.tensor(x.shape[1:]))
        high = 1
        low = 0
        threshold = self.attack_config["bs_gamma"] / (dim * torch.sqrt(dim))
        boost_start = torch.clamp(0.2 * x + 0.8 * x_adv, 0, 1)
        is_adv, is_cache = self.is_adversarial(boost_start, y, targeted)
        if is_adv == 1 and not is_cache[0]:
            x_adv = boost_start
        iters = 0
        while high - low > threshold and iters < self.attack_config["bs_max_iter"]:
            middle = (high + low) / 2
            interpolated = (1 - middle) * x + middle * x_adv
            is_adv, is_cache = self.is_adversarial(interpolated, y, targeted)
            if is_cache[0] and not self.attack_config["adaptive"]["bs_boundary_end_on_hit"]:
                break
            elif is_cache[0] and self.attack_config["adaptive"]["bs_boundary_end_on_hit"]:
                self.end("Boundary search failure.")
            if is_adv == 1:
                high = middle
            else:
                low = middle
            iters += 1
        interpolated = (1 - high) * x + high * x_adv
        return interpolated

    def step_in_circular_direction(self, dir1, dir2, r, degree):
        degree = degree.reshape(degree.shape + (1,) * (len(dir1.shape) - len(degree.shape)))
        r = r.reshape(r.shape + (1,) * (len(dir1.shape) - len(r.shape)))
        result = dir1 * torch.cos(degree * np.pi / 180) + dir2 * torch.sin(degree * np.pi / 180)
        result = result * r * torch.cos(degree * np.pi / 180)
        return result

    def gram_schmidt(self, v, orthogonal_with):
        v_repeated = torch.cat([v] * len(orthogonal_with), axis=0)
        gs_coeff = (orthogonal_with * v_repeated).flatten(1).sum(1)
        proj = gs_coeff.reshape(
            gs_coeff.shape + (1,) * (len(orthogonal_with.shape) - len(gs_coeff.shape))) * orthogonal_with
        v = v - proj.sum(0)
        return v

    def get_zig_zag_mask(self, x, mask_size):
        total_components = mask_size[0] * mask_size[1]
        n_coeff_kept = int(total_components * min(1, self.attack_config["freq_range"][1]))
        n_coeff_to_start = int(total_components * max(0, self.attack_config["freq_range"][0]))
        mask_size = (x.shape[0], x.shape[1], mask_size[0], mask_size[1])
        zig_zag_mask = torch.zeros(mask_size)
        s = 0
        while n_coeff_kept > 0:
            for i in range(min(s + 1, mask_size[2])):
                for j in range(min(s + 1, mask_size[3])):
                    if i + j == s:
                        if n_coeff_to_start > 0:
                            n_coeff_to_start -= 1
                            continue
                        if s % 2:
                            zig_zag_mask[:, :, i, j] = 1
                        else:
                            zig_zag_mask[:, :, j, i] = 1
                        n_coeff_kept -= 1
                        if n_coeff_kept == 0:
                            return zig_zag_mask
            s += 1
        return zig_zag_mask

    def dct2_8_8(self, image, mask):
        assert mask.shape[-2:] == (8, 8)
        imsize = image.shape
        dct = torch.zeros_like(image)
        for i in np.r_[:imsize[2]:8]:
            for j in np.r_[:imsize[3]:8]:
                dct_i_j = dct_2d(image[:, :, i:(i + 8), j:(j + 8)])
                dct[:, :, i:(i + 8), j:(j + 8)] = dct_i_j * mask  # [:dct_i_j.shape[0], :dct_i_j.shape[1]]
        return dct

    def idct2_8_8(self, dct):
        im_dct = torch.zeros_like(dct)
        for i in np.r_[:dct.shape[2]:8]:
            for j in np.r_[:dct.shape[3]:8]:
                im_dct[:, :, i:(i + 8), j:(j + 8)] = idct_2d(dct[:, :, i:(i + 8), j:(j + 8)])
        return im_dct

    def attack_untargeted(self, x, y):
        # Initialize
        x_adv = x.clone()
        while True:
            x_adv = torch.clamp(x_adv + torch.randn_like(x_adv) * 0.5, 0, 1)
            is_adv, is_cache = self.is_adversarial(x_adv, y, targeted=False)
            if is_adv == 1:
                break
        x_adv = self.binary_search_to_boundary(x, y, x_adv, targeted=False)
        norm_dist = torch.linalg.norm(x_adv - x) / (x.shape[-1] * x.shape[-2] * x.shape[-3]) ** 0.5
        explored_orthogonal_directions = ((x_adv - x) / torch.linalg.norm(x_adv - x))
        theta_max = self.attack_config["theta_max"]
        if self.attack_config["adaptive"]["bs_min_angle"]:
            theta_min = self.binary_search_min_angle(x, x_adv)
        else:
            theta_min = 0

        step_attempts = 0
        rollback = False

        # Attack
        pbar = tqdm(range(self.attack_config["max_iter"]), colour="red")
        for t in pbar:
            # Get new orthogonal direction and corresponding best angle for a circular step
            epsilon = 0
            while epsilon == 0:
                theta_max = max(theta_max, theta_min)
                probs = torch.FloatTensor(size=x.shape).uniform_(0, 3).long().to(x.device) - 1
                dcts = torch.tanh(self.dct2_8_8(x, self.get_zig_zag_mask(x, (8, 8)).to(x.device)))
                new_direction = self.idct2_8_8(dcts * probs) + torch.FloatTensor(size=x.shape).normal_(std=0).to(
                    x.device)
                new_direction = self.gram_schmidt(new_direction, explored_orthogonal_directions)
                new_direction = new_direction / torch.linalg.norm(new_direction)

                explored_orthogonal_directions = torch.cat((
                    explored_orthogonal_directions[:1],
                    explored_orthogonal_directions[
                    1 + len(explored_orthogonal_directions) - self.attack_config["n_ortho"]:],
                    new_direction), dim=0)
                # get best angle
                direction = ((x_adv - x) / torch.linalg.norm(x_adv - x))
                evolution_function = lambda degree: torch.clamp(
                    x + self.step_in_circular_direction(direction, new_direction, torch.linalg.norm(x_adv - x), degree),
                    0, 1)
                coefficients = torch.zeros(2 * self.attack_config["eval_per_direction"]).to(x.device)
                for i in range(0, self.attack_config["eval_per_direction"]):
                    coefficients[2 * i] = 1 - (i / self.attack_config["eval_per_direction"])
                    coefficients[2 * i + 1] = - coefficients[2 * i]
                best_epsilon = 0

                step_attempts += 1
                for coeff in coefficients:
                    possible_best_epsilon = coeff * theta_max
                    x_evolved = evolution_function(possible_best_epsilon)
                    is_adv, is_cache = self.is_adversarial(x_evolved, y, targeted=False)
                    if is_cache[0] and step_attempts < self.attack_config["adaptive"]["step_max_attempts"]:
                        rollback = True
                        print("Step movement failure. Rolling back.", coeff * theta_max, step_attempts)
                        break
                    elif is_cache[0] and step_attempts >= self.attack_config["adaptive"]["step_max_attempts"]:
                        self.end("Step movement failure.")
                    if best_epsilon == 0 and is_adv == 1:
                        best_epsilon = possible_best_epsilon
                    if best_epsilon != 0:
                        break
                if rollback:
                    continue
                step_attempts = 0

                if best_epsilon == 0:
                    theta_max = theta_max * self.attack_config["rho"]
                if best_epsilon != 0 and epsilon == 0:
                    theta_max = theta_max / self.attack_config["rho"]
                    epsilon = best_epsilon

                pbar.set_description(
                    f"Step {t}: Norm distance: {norm_dist} | Cache Hits : {self._model.cache_hits}/{self._model.total}")
            evolution_function = lambda degree: torch.clamp(
                x + self.step_in_circular_direction(direction, new_direction, torch.linalg.norm(x_adv - x), degree), 0,
                1)

            # # alpha binary search
            # check_opposite = epsilon > 0
            # lower = epsilon
            # if abs(lower) != theta_max:
            #     upper = lower + torch.sign(lower) * theta_max / self.attack_config["eval_per_direction"]
            # else:
            #     upper = 0
            # max_angle = 180
            # keep_going = upper == 0
            # while keep_going:
            #     new_upper = lower + torch.sign(lower) * theta_max / self.attack_config["eval_per_direction"]
            #     new_upper = min(new_upper, max_angle)
            #     x_evolved_new_upper = evolution_function(new_upper)
            #     is_adv, is_cache = self.is_adversarial(x_evolved_new_upper, y, targeted=False)
            #     if is_adv == 1:
            #         lower = new_upper
            #     else:
            #         upper = new_upper
            #         keep_going = False
            # step = 0
            # over_gamma = abs(torch.cos(lower * np.pi / 180) - torch.cos(upper * np.pi / 180)) > self.attack_config[
            #     "bs_gamma"]
            # while step < self.attack_config["bs_max_iter"] and over_gamma:
            #     mid_bound = (upper + lower) / 2
            #     if mid_bound != 0:
            #         mid = evolution_function(mid_bound)
            #     else:
            #         mid = torch.zeros_like(x)
            #     is_mid_adversarial, is_cache = self.is_adversarial(mid, y, targeted=False)
            #     if check_opposite:
            #         mid_opp = evolution_function(-mid_bound)
            #     else:
            #         mid_opp = torch.zeros_like(x)
            #     is_mid_opp_adversarial, is_cache = self.is_adversarial(mid_opp, y, targeted=False)
            #     if is_mid_adversarial:
            #         lower = mid_bound
            #     elif not is_mid_adversarial and check_opposite and is_mid_opp_adversarial:
            #         lower = -mid_bound
            #     if not is_mid_adversarial and check_opposite and is_mid_opp_adversarial:
            #         upper = -upper
            #     if abs(lower) != abs(mid_bound):
            #         upper = mid_bound
            #     check_opposite = check_opposite and is_mid_opp_adversarial and lower > 0
            #     over_gamma = abs(torch.cos(lower * np.pi / 180) - torch.cos(upper * np.pi / 180)) > self.attack_config[
            #         "bs_gamma"]
            #     step += 1
            # epsilon = lower
            # # end alpha binary search

            candidate = evolution_function(epsilon)
            if torch.linalg.norm(candidate - x) < torch.linalg.norm(x_adv - x):
                x_adv = candidate

            # Logging current progress with normalized L2 distance
            norm_dist = torch.linalg.norm(x_adv - x) / (x.shape[-1] * x.shape[-2] * x.shape[-3]) ** 0.5
            pbar.set_description(
                f"Step {t}: Norm distance: {norm_dist} | Cache Hits : {self._model.cache_hits}/{self._model.total}")
            if norm_dist < self.attack_config["eps"]:
                return x_adv

        # final binary search
        x_adv = self.binary_search_to_boundary(x, y, x_adv, targeted=False)
        norm_dist = torch.linalg.norm(x_adv - x) / (x.shape[-1] * x.shape[-2] * x.shape[-3]) ** 0.5
        if norm_dist < self.attack_config["eps"]:
            return x_adv
        else:
            return x

    def binary_search_min_angle(self, x, x_adv):
        direction = (x_adv - x) / torch.linalg.norm(x_adv - x)
        explored_orthogonal_directions = (x_adv - x) / torch.linalg.norm(x_adv - x)

        lower = self.attack_config["adaptive"]["bs_min_angle_lower"]
        upper = self.attack_config["adaptive"]["bs_min_angle_upper"]
        angle = upper
        for _ in range(self.attack_config["adaptive"]["bs_min_angle_steps"]):
            mid = (lower + upper) / 2
            cache_hits = 0
            for _ in range(self.attack_config["adaptive"]["bs_min_angle_sample_size"]):
                probs = torch.FloatTensor(size=x.shape).uniform_(0, 3).long().to(x.device) - 1
                dcts = torch.tanh(self.dct2_8_8(x, self.get_zig_zag_mask(x, (8, 8)).to(x.device)))
                new_direction = self.idct2_8_8(dcts * probs) + torch.FloatTensor(size=x.shape).normal_(std=0).to(
                    x.device)
                new_direction = self.gram_schmidt(new_direction, explored_orthogonal_directions)
                new_direction = new_direction / torch.linalg.norm(new_direction)
                explored_orthogonal_directions = torch.cat((
                    explored_orthogonal_directions[:1],
                    explored_orthogonal_directions[
                    1 + len(explored_orthogonal_directions) - self.attack_config["n_ortho"]:],
                    new_direction), dim=0)
                evolution_function = lambda degree: torch.clamp(
                    x + self.step_in_circular_direction(direction, new_direction, torch.linalg.norm(x_adv - x), degree),
                    0, 1)
                noisy_img = evolution_function(torch.tensor(mid).to(x.device))
                probs, is_cache = self.model(noisy_img)
                if is_cache[0]:
                    cache_hits += 1
            if cache_hits / self.attack_config["adaptive"]["bs_min_angle_sample_size"] \
                    <= self.attack_config["adaptive"]["bs_min_angle_hit_rate"]:
                angle = mid
                upper = mid
            else:
                lower = mid
            print(
                f"Step Size : {angle:.6f} | Cache Hits : {cache_hits}/{self.attack_config['adaptive']['bs_min_angle_sample_size']}, upper : {upper:.6f}, lower : {lower:.6f}")
        return angle

    def attack_targeted(self, x, y, x_adv):
        # Initialize
        y = torch.argmax(self._model.model(x)).unsqueeze(0)
        x_adv = self.binary_search_to_boundary(x, y, x_adv, targeted=False)
        norm_dist = torch.linalg.norm(x_adv - x) / (x.shape[-1] * x.shape[-2] * x.shape[-3]) ** 0.5
        explored_orthogonal_directions = ((x_adv - x) / torch.linalg.norm(x_adv - x))
        theta_max = self.attack_config["theta_max"]
        if self.attack_config["adaptive"]["bs_min_angle"]:
            theta_min = self.binary_search_min_angle(x, x_adv)
        else:
            theta_min = 0

        step_attempts = 0
        rollback = False

        # Attack
        pbar = tqdm(range(self.attack_config["max_iter"]), colour="red")
        for t in pbar:
            # Get new orthogonal direction and corresponding best angle for a circular step
            epsilon = 0
            while epsilon == 0:
                theta_max = max(theta_max, theta_min)
                probs = torch.FloatTensor(size=x.shape).uniform_(0, 3).long().to(x.device) - 1
                dcts = torch.tanh(self.dct2_8_8(x, self.get_zig_zag_mask(x, (8, 8)).to(x.device)))
                new_direction = self.idct2_8_8(dcts * probs) + torch.FloatTensor(size=x.shape).normal_(std=0).to(
                    x.device)
                new_direction = self.gram_schmidt(new_direction, explored_orthogonal_directions)
                new_direction = new_direction / torch.linalg.norm(new_direction)

                explored_orthogonal_directions = torch.cat((
                    explored_orthogonal_directions[:1],
                    explored_orthogonal_directions[
                    1 + len(explored_orthogonal_directions) - self.attack_config["n_ortho"]:],
                    new_direction), dim=0)
                # get best angle
                direction = ((x_adv - x) / torch.linalg.norm(x_adv - x))
                evolution_function = lambda degree: torch.clamp(
                    x + self.step_in_circular_direction(direction, new_direction, torch.linalg.norm(x_adv - x), degree),
                    0, 1)
                coefficients = torch.zeros(2 * self.attack_config["eval_per_direction"]).to(x.device)
                for i in range(0, self.attack_config["eval_per_direction"]):
                    coefficients[2 * i] = 1 - (i / self.attack_config["eval_per_direction"])
                    coefficients[2 * i + 1] = - coefficients[2 * i]
                best_epsilon = 0

                step_attempts += 1
                for coeff in coefficients:
                    possible_best_epsilon = coeff * theta_max
                    x_evolved = evolution_function(possible_best_epsilon)
                    is_adv, is_cache = self.is_adversarial(x_evolved, y, targeted=False)
                    if is_cache[0] and step_attempts < self.attack_config["adaptive"]["step_max_attempts"]:
                        rollback = True
                        print("Step movement failure. Rolling back.", coeff * theta_max, step_attempts)
                        break
                    elif is_cache[0] and step_attempts >= self.attack_config["adaptive"]["step_max_attempts"]:
                        self.end("Step movement failure.")
                    if best_epsilon == 0 and is_adv == 1:
                        best_epsilon = possible_best_epsilon
                    if best_epsilon != 0:
                        break
                if rollback:
                    continue
                step_attempts = 0

                if best_epsilon == 0:
                    theta_max = theta_max * self.attack_config["rho"]
                if best_epsilon != 0 and epsilon == 0:
                    theta_max = theta_max / self.attack_config["rho"]
                    epsilon = best_epsilon

                pbar.set_description(
                    f"Step {t}: Norm distance: {norm_dist} | Cache Hits : {self._model.cache_hits}/{self._model.total}")
            evolution_function = lambda degree: torch.clamp(
                x + self.step_in_circular_direction(direction, new_direction, torch.linalg.norm(x_adv - x), degree), 0,
                1)

            # # alpha binary search
            # check_opposite = epsilon > 0
            # lower = epsilon
            # if abs(lower) != theta_max:
            #     upper = lower + torch.sign(lower) * theta_max / self.attack_config["eval_per_direction"]
            # else:
            #     upper = 0
            # max_angle = 180
            # keep_going = upper == 0
            # while keep_going:
            #     new_upper = lower + torch.sign(lower) * theta_max / self.attack_config["eval_per_direction"]
            #     new_upper = min(new_upper, max_angle)
            #     x_evolved_new_upper = evolution_function(new_upper)
            #     is_adv, is_cache = self.is_adversarial(x_evolved_new_upper, y, targeted=False)
            #     if is_adv == 1:
            #         lower = new_upper
            #     else:
            #         upper = new_upper
            #         keep_going = False
            # step = 0
            # over_gamma = abs(torch.cos(lower * np.pi / 180) - torch.cos(upper * np.pi / 180)) > self.attack_config[
            #     "bs_gamma"]
            # while step < self.attack_config["bs_max_iter"] and over_gamma:
            #     mid_bound = (upper + lower) / 2
            #     if mid_bound != 0:
            #         mid = evolution_function(mid_bound)
            #     else:
            #         mid = torch.zeros_like(x)
            #     is_mid_adversarial, is_cache = self.is_adversarial(mid, y, targeted=False)
            #     if check_opposite:
            #         mid_opp = evolution_function(-mid_bound)
            #     else:
            #         mid_opp = torch.zeros_like(x)
            #     is_mid_opp_adversarial, is_cache = self.is_adversarial(mid_opp, y, targeted=False)
            #     if is_mid_adversarial:
            #         lower = mid_bound
            #     elif not is_mid_adversarial and check_opposite and is_mid_opp_adversarial:
            #         lower = -mid_bound
            #     if not is_mid_adversarial and check_opposite and is_mid_opp_adversarial:
            #         upper = -upper
            #     if abs(lower) != abs(mid_bound):
            #         upper = mid_bound
            #     check_opposite = check_opposite and is_mid_opp_adversarial and lower > 0
            #     over_gamma = abs(torch.cos(lower * np.pi / 180) - torch.cos(upper * np.pi / 180)) > self.attack_config[
            #         "bs_gamma"]
            #     step += 1
            # epsilon = lower
            # # end alpha binary search

            candidate = evolution_function(epsilon)
            if torch.linalg.norm(candidate - x) < torch.linalg.norm(x_adv - x):
                x_adv = candidate

            # Logging current progress with normalized L2 distance
            norm_dist = torch.linalg.norm(x_adv - x) / (x.shape[-1] * x.shape[-2] * x.shape[-3]) ** 0.5
            pbar.set_description(
                f"Step {t}: Norm distance: {norm_dist} | Cache Hits : {self._model.cache_hits}/{self._model.total} | Theta : {theta_max}")
            if norm_dist < self.attack_config["eps"]:
                return x_adv

        # final binary search
        x_adv = self.binary_search_to_boundary(x, y, x_adv, targeted=False)
        norm_dist = torch.linalg.norm(x_adv - x) / (x.shape[-1] * x.shape[-2] * x.shape[-3]) ** 0.5
        if norm_dist < self.attack_config["eps"]:
            return x_adv
        else:
            return x
